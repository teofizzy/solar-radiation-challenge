"""
Inference and submission generation.
Loads trained model, runs prediction on test data, and outputs Zindi submission CSV.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import HPARAMS, PATHS, ensure_dirs, get_n_stations
from src.model_lstm import PhysicsInformedBiLSTM
from src.utils import get_device, clean_memory


def predict(dataset, model=None, model_path: str = None,
            feature_cols: list = None, batch_size: int = None):
    """
    Run inference on the dataset and return predictions.

    Parameters
    ----------
    dataset : SolarDataset
        Dataset to predict on.
    model : PhysicsInformedBiLSTM or None
        Trained model. If None, loads from model_path.
    model_path : str or None
        Path to model checkpoint.
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

    # Load model if not provided
    if model is None:
        if model_path is None:
            model_path = os.path.join(PATHS['experiments_dir'], 'best_model.pt')

        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        n_features = len(ckpt.get('feature_cols', feature_cols or []))
        n_stations = get_n_stations()

        model = PhysicsInformedBiLSTM(
            n_features=n_features,
            n_stations=n_stations,
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[PREDICT] Loaded model from {model_path}")

    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
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

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                kt_pred, ghi_pred = model(x, station_idx, clear_sky, is_night)

            ghi_np = ghi_pred.cpu().numpy()

            for i in range(len(ghi_np)):
                if is_test[i] == 1:
                    sid = sample_ids[i]
                    pred_val = float(ghi_np[i])
                    # Post-processing: ensure non-negative
                    pred_val = max(0.0, pred_val)
                    predictions[sid] = pred_val

    print(f"[PREDICT] Generated {len(predictions)} test predictions")
    clean_memory()
    return predictions


def generate_submission(predictions: dict, output_path: str = None):
    """
    Generate Zindi-format submission CSV.

    Format:
        ID, TargetMBE, TargetRMSE
        (TargetMBE and TargetRMSE are identical per Zindi rules)

    Parameters
    ----------
    predictions : dict
        {sample_id: predicted_ghi}
    output_path : str or None
        Path to save submission CSV.

    Returns
    -------
    pd.DataFrame : submission DataFrame
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
            # Fallback: use a safe default (0.0 for missing predictions)
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
