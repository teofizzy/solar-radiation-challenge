"""
Inference and submission generation.
Loads trained BiLSTM model, runs prediction on test data, and outputs Zindi submission CSV.

Updated for V1 reverted architecture:
- Model forward: (x, station_idx, clear_sky_ghi, is_night) -> (kt_pred, ghi_pred)
- No diag_vector, no atmos_feats, no center_kt_landsaf
- FP32 inference
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import HPARAMS, PATHS, ensure_dirs, get_n_stations
from src.model_lstm import PhysicsInformedBiLSTM
from src.utils import get_device, clean_memory


def predict(dataset, models=None, model_paths: list = None,
            feature_cols: list = None, batch_size: int = None):
    """
    Run inference on the dataset and return predictions.

    Parameters
    ----------
    dataset : SolarDataset
        Dataset to predict on.
    models : list of PhysicsInformedBiLSTM or None
        Trained models for ensembling.
    model_paths : list of str or None
        Paths to model checkpoints.
    feature_cols : list or None
        Feature column names (needed if loading model from checkpoint).
    batch_size : int or None
        Batch size for inference.

    Returns
    -------
    predictions : dict
        {sample_id: predicted_ghi} for all test samples.
    """
    device = get_device()
    batch_size = batch_size or HPARAMS['batch_size'] * 4

    # Load models if not provided
    if models is None:
        models = []
        if model_paths is None:
            model_paths = [
                os.path.join(PATHS['experiments_dir'], 'fold_1', 'best_model.pt'),
                os.path.join(PATHS['experiments_dir'], 'fold_2', 'best_model.pt')
            ]

        for path in model_paths:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            n_features = len(ckpt.get('feature_cols', feature_cols or []))
            n_stations = get_n_stations()

            m = PhysicsInformedBiLSTM(
                n_features=n_features,
                n_stations=n_stations,
                hidden_dim=HPARAMS['hidden_dim'],
                n_layers=HPARAMS['n_layers'],
                embed_dim=HPARAMS.get('station_embed_dim', 16),
                dropout=HPARAMS['dropout'],
            ).to(device)
            m.load_state_dict(ckpt['model_state_dict'])
            m.eval()
            models.append(m)
            print(f"[PREDICT] Loaded BiLSTM from {path}")
    else:
        for m in models:
            m.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    predictions = {}

    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            station_idx = batch['station_idx'].to(device)
            clear_sky = batch['clear_sky_ghi'].to(device)
            is_night = batch['is_night'].to(device)
            is_test = batch['is_test'].numpy()
            sample_ids = batch['sample_id']

            ensemble_ghi = []
            
            for m in models:
                _, ghi_pred = m(x, station_idx, clear_sky, is_night)
                ensemble_ghi.append(ghi_pred.cpu().numpy())
                
            # Average across the ensemble
            ghi_np = np.mean(ensemble_ghi, axis=0)

            for i in range(len(ghi_np)):
                if is_test[i] == 1:
                    sid = sample_ids[i]
                    pred_val = float(ghi_np[i])
                    # Physical constraint: non-negative
                    pred_val = max(0.0, pred_val)
                    predictions[sid] = pred_val

    print(f"[PREDICT] Generated {len(predictions)} test predictions "
          f"using {len(models)}-model ensemble")
    clean_memory()
    return predictions


def generate_submission(predictions: dict, output_path: str = None):
    """
    Generate Zindi-format submission CSV.

    Format:
        ID, TargetMBE, TargetRMSE
        (TargetMBE and TargetRMSE are identical per Zindi rules)
    """
    ensure_dirs()

    if output_path is None:
        output_path = os.path.join(PATHS['submissions_dir'], 'submission.csv')

    # Load sample submission for format reference
    sample_sub_path = PATHS.get('sample_submission')
    if sample_sub_path and os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)
        all_ids = sample_sub['ID'].tolist()
        print(f"  Sample submission: {len(all_ids)} rows")
    else:
        all_ids = list(predictions.keys())
        print(f"  Using prediction IDs: {len(all_ids)} rows")

    # Build submission
    rows = []
    missing_count = 0
    for sid in all_ids:
        if sid in predictions:
            val = predictions[sid]
        else:
            val = 0.0
            missing_count += 1
        rows.append({'ID': sid, 'TargetMBE': val, 'TargetRMSE': val})

    if missing_count > 0:
        print(f"  WARNING: {missing_count} test IDs had no prediction (filled with 0)")

    submission = pd.DataFrame(rows)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"[PREDICT] Submission saved to {output_path}")
    print(f"  Shape: {submission.shape}")
    print(f"  TargetMBE range: [{submission['TargetMBE'].min():.2f}, "
          f"{submission['TargetMBE'].max():.2f}]")

    return submission
