"""
Distributed Data Parallel (DDP) Training for the Physics-Informed Patch Transformer.
Supports 4 GPUs using torchrun.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler

from src.config import HPARAMS, PATHS, SEED, WANDB_CONFIG, seed_everything, ensure_dirs, get_n_stations
from src.model_patch import PhysicsInformedPatchTransformer
from src.loss import ZindiSolarLoss, compute_zindi_score
from src.utils import timer, clean_memory

try:
    import wandb
except ImportError:
    wandb = None

def setup_ddp():
    """Initialize distributed process group."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """Destroy distributed process group."""
    dist.destroy_process_group()

def reduce_tensor(tensor):
    """Average a tensor across all processes."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def get_train_val_indices(dataset, val_months: list = None):
    """
    Split dataset indices into train and validation sets.
    """
    if val_months is None:
        val_months = [3, 7, 11]

    train_indices = []
    val_indices = []

    for i, sample in enumerate(dataset.samples):
        if sample['is_test'] == 1 or np.isnan(sample['target_delta_kt']):
            continue
        month = sample['month']
        if month in val_months:
            val_indices.append(i)
        else:
            train_indices.append(i)

    return train_indices, val_indices

def train_model_ddp(dataset, feature_cols: list, val_months: list = None,
                   model_save_dir: str = None, use_wandb: bool = False):
    """
    Train using DDP across multiple GPUs.
    """
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    is_main = (local_rank == 0)
    
    seed_everything(SEED + local_rank) # Unique seed per rank
    ensure_dirs()
    device = torch.device(f'cuda:{local_rank}')

    if model_save_dir is None:
        model_save_dir = PATHS['experiments_dir']
    if is_main:
        os.makedirs(model_save_dir, exist_ok=True)

    # WandB init (Only on main rank)
    if is_main and use_wandb and wandb is not None:
        if wandb.run is None:
            wandb.init(
                project=WANDB_CONFIG['project'],
                entity=WANDB_CONFIG['entity'],
                config=HPARAMS,
                reinit='allow'
            )
        # Strict HPARAMS update to prevent silent field creation
        for k, v in wandb.config.items():
            if k in HPARAMS:
                HPARAMS[k] = v

    # Split into train/val
    train_indices, val_indices = get_train_val_indices(dataset, val_months)
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # Distributed Samplers
    train_sampler = DistributedSampler(train_subset, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_subset, shuffle=False)

    train_loader = DataLoader(
        train_subset,
        batch_size=HPARAMS['batch_size'],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=HPARAMS['batch_size'] * 2,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
    )

    # Model definition
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

    # Wrap model in DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main:
        print(f"\n[DDP] Initialized with {world_size} GPUs")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HPARAMS['lr'] * world_size, # Scale learning rate by world size
        weight_decay=HPARAMS['weight_decay'],
    )

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=HPARAMS['lr'] * world_size,
        epochs=HPARAMS['epochs'],
        steps_per_epoch=steps_per_epoch,
        pct_start=0.12,
    )

    criterion = ZindiSolarLoss(
        dkt_weight=HPARAMS.get('dkt_weight', 0.4),
        zindi_weight=HPARAMS.get('zindi_weight', 0.6),
    )

    scaler = torch.amp.GradScaler('cuda')
    best_val_score = float('inf')
    history = {'train_loss': [], 'val_zindi': []}

    for epoch in range(HPARAMS['epochs']):
        train_sampler.set_epoch(epoch)
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

            with torch.amp.autocast('cuda'):
                delta_kt_pred, ghi_pred = model(x, station_idx, diag_vector, clear_sky, is_night, center_kt_landsaf)
                loss, loss_dict = criterion(delta_kt_pred, ghi_pred, target_delta_kt, target_ghi, is_night, clear_sky)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), HPARAMS['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_losses.append(loss_dict['loss'])

        # Synchronize train loss
        avg_train_loss = torch.tensor(np.mean(train_losses)).to(device)
        avg_train_loss = reduce_tensor(avg_train_loss).item()

        # Validation
        model.eval()
        val_mbe_sum = torch.tensor(0.0).to(device)
        val_mse_sum = torch.tensor(0.0).to(device)
        val_count = torch.tensor(0.0).to(device)

        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                station_idx = batch['station_idx'].to(device)
                clear_sky = batch['clear_sky_ghi'].to(device)
                is_night = batch['is_night'].to(device)
                target_ghi = batch['target_ghi'].to(device)
                center_kt_landsaf = batch['center_kt_landsaf'].to(device)
                diag_vector = batch['diag_vector'].to(device)

                with torch.amp.autocast('cuda'):
                    _, ghi_pred = model(x, station_idx, diag_vector, clear_sky, is_night, center_kt_landsaf)

                valid = ~torch.isnan(target_ghi)
                if valid.any():
                    err = ghi_pred[valid] - target_ghi[valid]
                    val_mbe_sum += err.sum()
                    val_mse_sum += (err**2).sum()
                    val_count += valid.sum()

        # Reduce validation metrics
        dist.all_reduce(val_mbe_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_mse_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_count, op=dist.ReduceOp.SUM)
        if val_count > 0:
            val_mbe = torch.abs(val_mbe_sum / val_count).item()
            val_rmse = torch.sqrt(val_mse_sum / val_count).item()
            val_zindi = 0.5 * val_mbe + 0.5 * val_rmse
        else:
            val_mbe = val_rmse = val_zindi = float('nan')

        if is_main:
            print(f"Epoch {epoch+1:3d} | Train: {avg_train_loss:.4f} | Val Zindi: {val_zindi:.2f} (MBE: {val_mbe:.2f}, RMSE: {val_rmse:.2f})")
            if use_wandb and wandb is not None:
                wandb.log({
                    'epoch': epoch+1, 
                    'train/loss': avg_train_loss, 
                    'val/mbe': val_mbe,
                    'val/rmse': val_rmse,
                    'val/zindi_score': val_zindi
                })

            if not np.isnan(val_zindi) and val_zindi < best_val_score:
                best_val_score = val_zindi
                torch.save(model.module.state_dict(), os.path.join(model_save_dir, 'best_model.pt'))
                print(f"  [DDP] New best model saved (Score: {best_val_score:.4f})")

    cleanup_ddp()
    return None

if __name__ == "__main__":
    # This would typically be called from a wrapper that loads the dataset
    # but we'll leave it for the pipeline to invoke.
    pass
