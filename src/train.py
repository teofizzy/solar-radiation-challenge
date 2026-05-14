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

from src.config import HPARAMS, PATHS, SEED, WANDB_CONFIG, seed_everything, ensure_dirs, get_n_stations
from src.model_patch import PhysicsInformedPatchTransformer
from src.loss import ZindiSolarLoss, compute_zindi_score
from src.utils import timer, get_device, clean_memory

try:
    import wandb
except ImportError:
    wandb = None


def get_train_val_indices(dataset, val_months: list = None):
    """
    Split dataset indices into train and validation sets.
    Validation: Samples where the month is in val_months.
    """
    if val_months is None:
        val_months = [3, 7, 11]

    train_indices = []
    val_indices = []

    for i, sample in enumerate(dataset.samples):
        # Skip test samples (no target)
        if sample['is_test'] == 1 or np.isnan(sample['target_delta_kt']):
            continue

        month = sample['month']

        # Zindi strategy: Train on odd, Validate on odd (since even is hidden)
        if month in val_months:
            val_indices.append(i)
        else:
            train_indices.append(i)

    return train_indices, val_indices


def train_model(dataset, feature_cols: list, val_months: list = None,
                model_save_dir: str = None, use_wandb: bool = False):
    """
    Train the Physics-Informed BiLSTM with temporal CV.

    Parameters
    ----------
    dataset : SolarDataset
        Full dataset with all samples.
    feature_cols : list
        Feature column names.
    val_months : list
        Months to use for validation split.
    model_save_dir : str
        Directory to save model checkpoints.
    use_wandb : bool
        If True, log metrics to Weights & Biases.

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

    if use_wandb and wandb is not None:
        # If wandb.run is None, it means the sweep or pipeline hasn't initialized it yet.
        if wandb.run is None:
            # Try to get API key from Colab userdata
            try:
                from google.colab import userdata
                wandb_api_key = userdata.get('WANDB_API_KEY')
                if wandb_api_key:
                    wandb.login(key=wandb_api_key)
            except ImportError:
                pass
            
            wandb.init(
                project=WANDB_CONFIG['project'],
                entity=WANDB_CONFIG['entity'],
                config=HPARAMS,
                reinit='allow'
            )
        # Update HPARAMS if wandb sweep modified them.
        # wandb.config is not a plain dict; wrap in dict() to avoid TypeError.
        HPARAMS.update(dict(wandb.config))

    # Split into train/val
    train_indices, val_indices = get_train_val_indices(dataset, val_months)
    print(f"\n[TRAIN] Split: {len(train_indices)} train, {len(val_indices)} val")

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=HPARAMS['batch_size'],
        shuffle=True,
        num_workers=2,     # Optimized for Colab stability
        pin_memory=True,   # Accelerated
        drop_last=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=HPARAMS['batch_size'] * 2,
        shuffle=False,
        num_workers=2,     # Optimized for Colab stability
        pin_memory=True,
    )

    # Model selection
    n_features = len(feature_cols)
    n_stations = get_n_stations()

    model = PhysicsInformedPatchTransformer(
        n_features=n_features,
        n_stations=n_stations,
        d_model=HPARAMS['hidden_dim'],
        nhead=HPARAMS.get('transformer_heads', 8),
        num_layers=HPARAMS['n_layers'],
        patch_len=HPARAMS['patch_len'],
        stride=HPARAMS['stride'],
        dropout=HPARAMS['dropout'],
    ).to(device)
    print(f"  Model: PhysicsInformedPatchTransformer (Delta kt)")

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HPARAMS['lr'],
        weight_decay=HPARAMS['weight_decay'],
    )

    # LR Scheduler: OneCycleLR (stepped per batch) or CosineAnnealing (fallback)
    scheduler_type = HPARAMS.get('scheduler', 'onecycle')
    steps_per_epoch = max(len(train_loader), 1)

    if scheduler_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=HPARAMS['lr'],
            epochs=HPARAMS['epochs'],
            steps_per_epoch=steps_per_epoch,
            pct_start=HPARAMS.get('onecycle_pct_start', 0.12),
            anneal_strategy='cos',
            div_factor=HPARAMS.get('onecycle_div_factor', 25),
            final_div_factor=HPARAMS.get('onecycle_final_div', 1e4),
        )
        step_scheduler_per_batch = True
        print(f"  Scheduler: OneCycleLR (max_lr={HPARAMS['lr']}, pct_start={HPARAMS.get('onecycle_pct_start', 0.12)})")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=HPARAMS['epochs'], eta_min=1e-6
        )
        step_scheduler_per_batch = False
        print(f"  Scheduler: CosineAnnealingLR")

    # Loss function (Zindi-aligned multi-task)
    criterion = ZindiSolarLoss(
        dkt_weight=HPARAMS.get('dkt_weight', 0.4),
        zindi_weight=HPARAMS.get('zindi_weight', 0.6),
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_mbe': [], 'val_rmse': [], 'val_zindi': [], 'val_zindi_ema': [], 'lr': []}
    best_val_score = float('inf')
    val_zindi_ema = None
    ema_alpha = 0.3 # Smoothing factor for sweep signal
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
            target_delta_kt = batch['target_delta_kt'].to(device)
            center_kt_landsaf = batch['center_kt_landsaf'].to(device)

            diag_vector = batch['diag_vector'].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                delta_kt_pred, ghi_pred = model(x, station_idx, diag_vector, clear_sky, is_night, center_kt_landsaf)
                loss, loss_dict = criterion(delta_kt_pred, ghi_pred, target_delta_kt, target_ghi, is_night, clear_sky)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), HPARAMS['grad_clip']
            )
            scaler.step(optimizer)
            scaler.update()

            # OneCycleLR steps per batch, not per epoch
            if step_scheduler_per_batch:
                scheduler.step()

            train_losses.append(loss_dict['loss'])

        # CosineAnnealing steps per epoch
        if not step_scheduler_per_batch:
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
                center_kt_landsaf = batch['center_kt_landsaf'].to(device)

                diag_vector = batch['diag_vector'].to(device)

                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    delta_kt_pred, ghi_pred = model(x, station_idx, diag_vector, clear_sky, is_night, center_kt_landsaf)

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

        # Compute EMA for stable sweep signal
        if val_zindi_ema is None:
            val_zindi_ema = val_zindi
        else:
            val_zindi_ema = (1 - ema_alpha) * val_zindi_ema + ema_alpha * val_zindi

        # Record history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_zindi)
        history['val_mbe'].append(val_mbe)
        history['val_rmse'].append(val_rmse)
        history['val_zindi'].append(val_zindi)
        history['val_zindi_ema'].append(val_zindi_ema)
        history['lr'].append(current_lr)

        # Logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{HPARAMS['epochs']} | "
                  f"Train: {avg_train_loss:.4f} | "
                  f"Val MBE: {val_mbe:.2f} | Val RMSE: {val_rmse:.2f} | "
                  f"Val Zindi: {val_zindi:.2f} | LR: {current_lr:.6f}")

        if use_wandb and wandb is not None:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': avg_train_loss,
                'val/mbe': val_mbe,
                'val/rmse': val_rmse,
                'val/zindi_score': val_zindi,
                'val/zindi_score_ema': val_zindi_ema,
                'lr': current_lr
            })

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
    best_model_path = os.path.join(model_save_dir, 'best_model.pt')
    best_ckpt = torch.load(
        best_model_path,
        map_location=device, weights_only=False
    )
    model.load_state_dict(best_ckpt['model_state_dict'])
    print(f"\n[TRAIN] Best model at epoch {best_ckpt['epoch']+1}: "
          f"Zindi={best_ckpt['val_zindi']:.2f}, "
          f"MBE={best_ckpt['val_mbe']:.2f}, "
          f"RMSE={best_ckpt['val_rmse']:.2f}")

    if use_wandb and wandb is not None:
        artifact = wandb.Artifact('best-model', type='model')
        artifact.add_file(best_model_path)
        wandb.log_artifact(artifact)
        
        # We don't finish the run here if it's a sweep, sweep agent handles it.
        # But if it was init manually, we can finish it. 
        # Actually it's safer to let the caller handle wandb.finish() if needed.

    clean_memory()
    return model, history, best_model_path
