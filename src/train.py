"""
Training loop for the Hybrid BiLSTM (residual prediction).
Temporal CV: odd months train, specified months validate.

Architecture (Hybrid V2):
  - FP32 training (no AMP autocast swings)
  - Huber loss on GHI (not Zindi loss -- prevents MBE/RMSE oscillation)
  - ReduceLROnPlateau (not CosineAnnealing -- better for noisy data)
  - AdamW + early stopping + gradient clipping
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.config import HPARAMS, PATHS, SEED, WANDB_CONFIG, seed_everything, ensure_dirs, get_n_stations
from src.model_lstm import PhysicsInformedBiLSTM
from src.loss import SolarHuberLoss, compute_zindi_score
from src.utils import timer, get_device, clean_memory

try:
    import wandb
except ImportError:
    wandb = None


# ------------------------------------------------------------------
# Train/Val Split (temporal: odd months train, specified months val)
# ------------------------------------------------------------------
def get_train_val_indices(dataset, val_months: list = None):
    """Split dataset indices into train and validation sets."""
    if val_months is None:
        val_months = [3, 7, 11]

    train_indices = []
    val_indices = []

    for i, sample in enumerate(dataset.samples):
        if sample['is_test'] == 1 or np.isnan(sample['target_ghi']):
            continue
        month = sample['month']
        if month in val_months:
            val_indices.append(i)
        else:
            train_indices.append(i)

    return train_indices, val_indices


# ------------------------------------------------------------------
# Main Training Function (Hybrid V2: residual + Huber + ReduceLROnPlateau)
# ------------------------------------------------------------------
def train_model(dataset, feature_cols: list, val_months: list = None,
                model_save_dir: str = None, use_wandb: bool = False):
    """
    Train the Physics-Informed BiLSTM with Huber loss on GHI residuals.

    FP32 training, AdamW + ReduceLROnPlateau, early stopping.

    Returns: model, history, best_model_path
    """
    seed_everything(SEED)
    ensure_dirs()
    device = get_device()

    if model_save_dir is None:
        model_save_dir = PATHS['experiments_dir']
    os.makedirs(model_save_dir, exist_ok=True)

    if use_wandb and wandb is not None:
        if wandb.run is None:
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
        HPARAMS.update(dict(wandb.config))

    # Split
    train_indices, val_indices = get_train_val_indices(dataset, val_months)
    print(f"\n[TRAIN] Split: {len(train_indices)} train, {len(val_indices)} val")

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=HPARAMS['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=HPARAMS['batch_size'] * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Model
    n_features = len(feature_cols)
    n_stations = get_n_stations()

    model = PhysicsInformedBiLSTM(
        n_features=n_features,
        n_stations=n_stations,
        hidden_dim=HPARAMS['hidden_dim'],
        n_layers=HPARAMS['n_layers'],
        embed_dim=HPARAMS.get('station_embed_dim', 16),
        dropout=HPARAMS['dropout'],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model: PhysicsInformedBiLSTM (residual prediction)")
    print(f"  Parameters: {param_count:,}")
    print(f"  Param/sample ratio: {param_count / max(len(train_indices), 1):.2f}")

    # Optimizer: AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HPARAMS['lr'],
        weight_decay=HPARAMS['weight_decay'],
    )

    # LR Scheduler: ReduceLROnPlateau (multi-AI consensus: better for noisy data)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )

    # Loss: Huber on GHI (NOT Zindi loss)
    huber_delta = HPARAMS.get('huber_delta', 50.0)
    criterion = SolarHuberLoss(
        delta=huber_delta,
        lambda_night=0.1
    )

    # History
    history = {'train_loss': [], 'val_loss': [], 'val_mbe': [], 'val_rmse': [],
               'val_zindi': [], 'lr': []}
    best_val_score = float('inf')
    patience_counter = 0
    total_epochs = HPARAMS['epochs']

    print(f"\n[TRAIN] Starting {total_epochs} epochs "
          f"(FP32, Huber delta={huber_delta}, ReduceLROnPlateau)")

    for epoch in range(total_epochs):
        # ---- Training ----
        model.train()
        train_losses = []

        for batch in train_loader:
            x = batch['x'].to(device)
            station_idx = batch['station_idx'].to(device)
            mdssf_ghi = batch['mdssf_ghi'].to(device)
            clear_sky = batch['clear_sky_ghi'].to(device)
            is_night = batch['is_night'].to(device)
            target_ghi = batch['target_ghi'].to(device)

            optimizer.zero_grad(set_to_none=True)

            # Forward (FP32, residual prediction)
            residual_pred, ghi_pred = model(x, station_idx, mdssf_ghi, clear_sky, is_night)

            # Huber loss on GHI
            loss, loss_dict = criterion(residual_pred, ghi_pred, target_ghi, is_night)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), HPARAMS['grad_clip'])
            optimizer.step()

            train_losses.append(loss_dict['loss'])

        avg_train_loss = np.mean(train_losses)

        # ---- Validation ----
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                station_idx = batch['station_idx'].to(device)
                mdssf_ghi = batch['mdssf_ghi'].to(device)
                clear_sky = batch['clear_sky_ghi'].to(device)
                is_night = batch['is_night'].to(device)
                target_ghi = batch['target_ghi']

                _, ghi_pred = model(x, station_idx, mdssf_ghi, clear_sky, is_night)

                val_preds.extend(ghi_pred.cpu().numpy())
                val_targets.extend(target_ghi.numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)

        # Zindi metrics (numpy) -- for evaluation only
        valid = ~np.isnan(val_targets)
        if valid.sum() > 0:
            residuals = val_preds[valid] - val_targets[valid]
            val_mbe = np.abs(np.mean(residuals))
            val_rmse = np.sqrt(np.mean(residuals ** 2))
            val_zindi = 0.5 * val_mbe + 0.5 * val_rmse
        else:
            val_mbe = val_rmse = val_zindi = float('inf')

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_zindi)
        history['val_mbe'].append(val_mbe)
        history['val_rmse'].append(val_rmse)
        history['val_zindi'].append(val_zindi)
        history['lr'].append(current_lr)

        # Step scheduler on validation Zindi score
        scheduler.step(val_zindi)

        # Logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{total_epochs} | "
                  f"Train: {avg_train_loss:.4f} | "
                  f"Val MBE: {val_mbe:.2f} | Val RMSE: {val_rmse:.2f} | "
                  f"Zindi: {val_zindi:.2f} | LR: {current_lr:.6f}")

        if use_wandb and wandb is not None:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': avg_train_loss,
                'val/mbe': val_mbe,
                'val/rmse': val_rmse,
                'val/zindi_score': val_zindi,
                'lr': current_lr,
            })

        # Save best model
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
                'hparams': dict(HPARAMS),
            }, os.path.join(model_save_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= HPARAMS['patience']:
                print(f"\n  Early stopping at epoch {epoch+1} "
                      f"(best Zindi: {best_val_score:.2f})")
                break

    # Load best model
    best_model_path = os.path.join(model_save_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        best_ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
        print(f"\n[TRAIN] Best model at epoch {best_ckpt['epoch']+1}: "
              f"Zindi={best_ckpt['val_zindi']:.2f}, "
              f"MBE={best_ckpt['val_mbe']:.2f}, "
              f"RMSE={best_ckpt['val_rmse']:.2f}")

    if use_wandb and wandb is not None:
        artifact = wandb.Artifact('best-model', type='model')
        artifact.add_file(best_model_path)
        wandb.log_artifact(artifact)

    clean_memory()
    return model, history, best_model_path
