"""
Training loop for the Physics-Informed BiLSTM (direct kt prediction).
Temporal CV: odd months train, specified months validate.

Architecture (solar-sweep-1 proven, Zindi=45.48):
  - FP32 training (no AMP autocast swings)
  - ZindiLoss (0.5*|MBE| + 0.5*RMSE) -- direct leaderboard metric
  - CosineAnnealingLR (handles noisy ZindiLoss better than Plateau)
  - AdamW + early stopping + gradient clipping
  - Model predicts kt -> GHI = kt * clear_sky_ghi
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.config import HPARAMS, PATHS, SEED, WANDB_CONFIG, seed_everything, ensure_dirs, get_n_stations
from src.model_lstm import PhysicsInformedBiLSTM
from src.loss import ZindiLoss, compute_zindi_score
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
# Main Training Function (solar-sweep-1: kt + ZindiLoss + CosineAnnealing)
# ------------------------------------------------------------------
def train_model(dataset, feature_cols: list, val_months: list = None,
                model_save_dir: str = None, use_wandb: bool = False,
                collect_oof: bool = False):
    """
    Train the Physics-Informed BiLSTM with ZindiLoss on GHI.

    FP32 training, AdamW + CosineAnnealingLR, early stopping.

    Parameters
    ----------
    dataset : SolarDataset
    feature_cols : list of feature column names
    val_months : list of int for validation months
    model_save_dir : str
    use_wandb : bool
    collect_oof : bool
        If True, collect out-of-fold predictions for LightGBM Stage 2.

    Returns
    -------
    model, history, best_model_path, oof_data (if collect_oof=True)
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
        use_attention=HPARAMS.get('use_attention', False),
        attn_dropout=HPARAMS.get('attn_dropout', 0.1),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model: PhysicsInformedBiLSTM (direct kt prediction)")
    print(f"  Parameters: {param_count:,}")
    print(f"  Param/sample ratio: {param_count / max(len(train_indices), 1):.2f}")

    # Optimizer: AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HPARAMS['lr'],
        weight_decay=HPARAMS['weight_decay'],
    )

    # LR Scheduler: CosineAnnealingLR (proven in solar-sweep-1)
    total_epochs = HPARAMS['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=1e-6,
    )

    # Loss: ZindiLoss (direct leaderboard metric)
    lambda_smooth = HPARAMS.get('lambda_smooth', 0.001)
    criterion = ZindiLoss(
        lambda_smooth=lambda_smooth,
        lambda_night=0.1
    )

    # History
    history = {'train_loss': [], 'val_loss': [], 'val_mbe': [], 'val_rmse': [],
               'val_zindi': [], 'lr': []}
    best_val_score = float('inf')
    patience_counter = 0

    print(f"\n[TRAIN] Starting {total_epochs} epochs "
          f"(FP32, ZindiLoss lambda_smooth={lambda_smooth}, CosineAnnealingLR)")

    for epoch in range(total_epochs):
        # ---- Training ----
        model.train()
        train_losses = []

        for batch in train_loader:
            x = batch['x'].to(device)
            station_idx = batch['station_idx'].to(device)
            clear_sky = batch['clear_sky_ghi'].to(device)
            is_night = batch['is_night'].to(device)
            target_ghi = batch['target_ghi'].to(device)

            optimizer.zero_grad(set_to_none=True)

            # Forward (FP32, direct kt prediction)
            kt_pred, ghi_pred = model(x, station_idx, clear_sky, is_night)

            # ZindiLoss on GHI
            loss, loss_dict = criterion(kt_pred, ghi_pred, target_ghi, is_night)

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
                clear_sky = batch['clear_sky_ghi'].to(device)
                is_night = batch['is_night'].to(device)
                target_ghi = batch['target_ghi']

                _, ghi_pred = model(x, station_idx, clear_sky, is_night)

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

        # Step CosineAnnealing scheduler (epoch-based, no val metric needed)
        scheduler.step()

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

    # ---- Collect OOF predictions for LightGBM Stage 2 ----
    oof_data = None
    if collect_oof:
        print("\n[TRAIN] Collecting OOF predictions for Stage 2...")
        model.eval()
        oof_records = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                station_idx = batch['station_idx'].to(device)
                clear_sky = batch['clear_sky_ghi'].to(device)
                is_night = batch['is_night'].to(device)
                target_ghi = batch['target_ghi'].numpy()
                sample_ids = batch['sample_id']
                mdssf_ghi = batch['mdssf_ghi'].numpy()
                station_np = batch['station_idx'].numpy()

                _, ghi_pred = model(x, station_idx, clear_sky, is_night)
                ghi_pred_np = ghi_pred.cpu().numpy()

                for j in range(len(ghi_pred_np)):
                    oof_records.append({
                        'sample_id': sample_ids[j],
                        'ghi_pred': float(ghi_pred_np[j]),
                        'ghi_true': float(target_ghi[j]),
                        'mdssf_ghi': float(mdssf_ghi[j]),
                        'station_idx': int(station_np[j]),
                    })

        oof_data = oof_records
        print(f"  Collected {len(oof_data)} OOF samples")

    clean_memory()
    if collect_oof:
        return model, history, best_model_path, oof_data
    return model, history, best_model_path
