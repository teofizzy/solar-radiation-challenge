"""
Training loop for the Physics-Informed BiLSTM.
Temporal CV: odd months train, even months validate (mirrors competition structure).
Supports mixed precision, gradient clipping, early stopping.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.config import HPARAMS, PATHS, SEED, seed_everything, ensure_dirs, get_n_stations
from src.model_lstm import PhysicsInformedBiLSTM
from src.loss import ZindiSolarLoss, compute_zindi_score
from src.utils import timer, get_device, clean_memory


def get_train_val_indices(dataset, val_year: int = 2017):
    """
    Split dataset indices into train and validation sets.

    Train: samples where target is valid AND month is odd (or year != val_year)
    Val: samples where target is valid AND month is even AND year == val_year

    Parameters
    ----------
    dataset : SolarDataset
        Full dataset.
    val_year : int
        Year for validation.

    Returns
    -------
    train_indices, val_indices : list of int
    """
    train_indices = []
    val_indices = []

    for i, sample in enumerate(dataset.samples):
        # Skip test samples (no target)
        if sample['is_test'] == 1 or np.isnan(sample['target_kt']):
            continue

        # Parse month from sample_id if available, otherwise skip
        sid = sample['sample_id']
        if isinstance(sid, str) and '_' in sid:
            parts = sid.split('_')
            if len(parts) >= 2:
                try:
                    year_month = parts[1]
                    year = int(year_month.split('-')[0])
                    month = int(year_month.split('-')[1])

                    if year == val_year and month % 2 == 0:
                        val_indices.append(i)
                    else:
                        train_indices.append(i)
                    continue
                except (ValueError, IndexError):
                    pass

        # Fallback: put in training
        train_indices.append(i)

    return train_indices, val_indices


def train_model(dataset, feature_cols: list, val_year: int = 2017,
                model_save_dir: str = None):
    """
    Train the Physics-Informed BiLSTM with temporal CV.

    Parameters
    ----------
    dataset : SolarDataset
        Full dataset with all samples.
    feature_cols : list
        Feature column names.
    val_year : int
        Year to use for validation split.
    model_save_dir : str
        Directory to save model checkpoints.

    Returns
    -------
    model : PhysicsInformedBiLSTM
        Trained model.
    history : dict
        Training history with losses and metrics.
    """
    seed_everything(SEED)
    ensure_dirs()
    device = get_device()

    if model_save_dir is None:
        model_save_dir = PATHS['experiments_dir']
    os.makedirs(model_save_dir, exist_ok=True)

    # Split into train/val
    train_indices, val_indices = get_train_val_indices(dataset, val_year)
    print(f"\n[TRAIN] Split: {len(train_indices)} train, {len(val_indices)} val")

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=HPARAMS['batch_size'],
        shuffle=True,
        num_workers=0,     # Colab-safe
        pin_memory=False,  # Colab-safe
        drop_last=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=HPARAMS['batch_size'] * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Model
    n_features = len(feature_cols)
    n_stations = get_n_stations()
    model = PhysicsInformedBiLSTM(
        n_features=n_features,
        n_stations=n_stations,
        hidden_dim=HPARAMS['hidden_dim'],
        n_layers=HPARAMS['n_layers'],
        embed_dim=HPARAMS['embed_dim'],
        dropout=HPARAMS['dropout'],
    ).to(device)

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HPARAMS['lr'],
        weight_decay=HPARAMS['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=HPARAMS['epochs'], eta_min=1e-6
    )

    # Loss function
    criterion = ZindiSolarLoss(
        mbe_weight=HPARAMS['mbe_weight'],
        rmse_weight=HPARAMS['rmse_weight'],
        smoothness_weight=HPARAMS['smoothness_weight'],
        night_penalty_weight=HPARAMS['night_penalty_weight'],
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'val_mbe': [], 'val_rmse': [], 'val_zindi': [],
        'lr': [],
    }

    best_val_score = float('inf')
    patience_counter = 0

    print(f"\n[TRAIN] Starting training for {HPARAMS['epochs']} epochs...")
    print(f"  Device: {device}")
    print(f"  Batch size: {HPARAMS['batch_size']}")
    print(f"  Learning rate: {HPARAMS['lr']}")

    for epoch in range(HPARAMS['epochs']):
        # ---- Training ----
        model.train()
        train_losses = []

        for batch in train_loader:
            x = batch['x'].to(device)
            station_idx = batch['station_idx'].to(device)
            clear_sky = batch['clear_sky_ghi'].to(device)
            is_night = batch['is_night'].to(device)
            target_ghi = batch['target_ghi'].to(device)
            target_kt = batch['target_kt'].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                kt_pred, ghi_pred = model(x, station_idx, clear_sky, is_night)
                loss, loss_dict = criterion(ghi_pred, target_ghi, kt_pred, is_night)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), HPARAMS['grad_clip']
            )
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss_dict['total'])

        scheduler.step()
        avg_train_loss = np.mean(train_losses)

        # ---- Validation ----
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                station_idx = batch['station_idx'].to(device)
                clear_sky = batch['clear_sky_ghi'].to(device)
                is_night = batch['is_night'].to(device)
                target_ghi = batch['target_ghi']

                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    kt_pred, ghi_pred = model(x, station_idx, clear_sky, is_night)

                val_preds.extend(ghi_pred.cpu().numpy())
                val_targets.extend(target_ghi.numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)

        # Compute Zindi metrics
        valid = ~np.isnan(val_targets)
        if valid.sum() > 0:
            residuals = val_preds[valid] - val_targets[valid]
            val_mbe = np.abs(np.mean(residuals))
            val_rmse = np.sqrt(np.mean(residuals ** 2))
            val_zindi = 0.5 * val_mbe + 0.5 * val_rmse
        else:
            val_mbe = val_rmse = val_zindi = float('inf')

        # Record history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_zindi)
        history['val_mbe'].append(val_mbe)
        history['val_rmse'].append(val_rmse)
        history['val_zindi'].append(val_zindi)
        history['lr'].append(current_lr)

        # Logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{HPARAMS['epochs']} | "
                  f"Train: {avg_train_loss:.4f} | "
                  f"Val MBE: {val_mbe:.2f} | Val RMSE: {val_rmse:.2f} | "
                  f"Val Zindi: {val_zindi:.2f} | LR: {current_lr:.6f}")

        # Early stopping + best model saving
        if val_zindi < best_val_score:
            best_val_score = val_zindi
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_zindi': val_zindi,
                'val_mbe': val_mbe,
                'val_rmse': val_rmse,
                'feature_cols': feature_cols,
                'hparams': HPARAMS,
            }, os.path.join(model_save_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= HPARAMS['patience']:
                print(f"\n  Early stopping at epoch {epoch+1} "
                      f"(best Zindi: {best_val_score:.2f})")
                break

    # Load best model
    best_ckpt = torch.load(
        os.path.join(model_save_dir, 'best_model.pt'),
        map_location=device, weights_only=False
    )
    model.load_state_dict(best_ckpt['model_state_dict'])
    print(f"\n[TRAIN] Best model at epoch {best_ckpt['epoch']+1}: "
          f"Zindi={best_ckpt['val_zindi']:.2f}, "
          f"MBE={best_ckpt['val_mbe']:.2f}, "
          f"RMSE={best_ckpt['val_rmse']:.2f}")

    clean_memory()
    return model, history
