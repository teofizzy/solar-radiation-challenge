"""
Inference and submission generation for the Physics-Informed BiLSTM pipeline.

Architecture (sequential residual + boundary routing):
  - BiLSTM predicts kt (clearness index, sigmoid-bounded [0, kt_max])
  - GHI = kt * clear_sky_ghi, night-gated to 0
  - Interior samples: ghi_final = bilstm + lgbm_residual (sequential)
  - Boundary samples: ghi_final = lgbm_standalone (direct GHI prediction)
  - NO MDSSF satellite fallback (evidence: MDSSF RMSE=466 contaminates score)
  - FP32 inference (no AMP)
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
            feature_cols: list = None, batch_size: int = None,
            return_details: bool = False):
    """
    Run inference on the dataset and return predictions.

    Parameters
    ----------
    dataset : SolarDataset
    models : list of trained models (optional)
    model_paths : list of checkpoint paths (optional)
    feature_cols : list of feature names (optional, used if loading from checkpoints)
    batch_size : int (optional)
    return_details : bool
        If True, return additional data needed for LightGBM Stage 2.

    Returns
    -------
    predictions : dict
        {sample_id: predicted_ghi} for all test samples.
    details : list of dict (only if return_details=True)
        Per-sample metadata for Stage 2 integration.
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
            ckpt_hparams = ckpt.get('hparams', {})
            n_features = len(ckpt.get('feature_cols', feature_cols or []))
            n_stations = get_n_stations()

            m = PhysicsInformedBiLSTM(
                n_features=n_features,
                n_stations=n_stations,
                hidden_dim=ckpt_hparams.get('hidden_dim', HPARAMS['hidden_dim']),
                n_layers=ckpt_hparams.get('n_layers', HPARAMS['n_layers']),
                embed_dim=ckpt_hparams.get('station_embed_dim', HPARAMS.get('station_embed_dim', 16)),
                dropout=ckpt_hparams.get('dropout', HPARAMS['dropout']),
                use_attention=ckpt_hparams.get('use_attention', HPARAMS.get('use_attention', False)),
                attn_dropout=ckpt_hparams.get('attn_dropout', HPARAMS.get('attn_dropout', 0.1)),
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
    details_list = []

    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            station_idx = batch['station_idx'].to(device)
            clear_sky = batch['clear_sky_ghi'].to(device)
            is_night = batch['is_night'].to(device)
            is_test = batch['is_test'].numpy()
            sample_ids = batch['sample_id']
            mdssf_ghi = batch['mdssf_ghi'].numpy()

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

                    if return_details:
                        details_list.append({
                            'sample_id': sid,
                            'ghi_pred': pred_val,
                            'mdssf_ghi': float(mdssf_ghi[i]),
                            'station_idx': int(batch['station_idx'][i].item()),
                        })

    print(f"[PREDICT] Generated {len(predictions)} test predictions "
          f"using {len(models)}-model ensemble")
    clean_memory()

    if return_details:
        return predictions, details_list
    return predictions


def generate_submission(predictions: dict, output_path: str = None,
                        boundary_predictions: dict = None,
                        fallback_df: pd.DataFrame = None):
    """
    Generate Zindi-format submission CSV.

    Routing logic (evidence-backed):
      1. Interior samples: use BiLSTM predictions (+ optional LGBM residual correction)
      2. Boundary samples: use LightGBM standalone predictions (boundary_predictions dict)
      3. Last resort: 0.0 (should never happen if pipeline is correct)

    MDSSF satellite fallback is REMOVED. Evidence: 4% of samples at RMSE=466
    inflates overall RMSE from 86 to 126, costing ~20 Zindi points.

    Parameters
    ----------
    predictions : dict
        {sample_id: predicted_ghi} from BiLSTM (+LGBM residual) inference.
    output_path : str
        Path to save submission CSV.
    boundary_predictions : dict or None
        {sample_id: predicted_ghi} from LightGBM standalone for boundary samples.
    fallback_df : pd.DataFrame or None
        DEPRECATED. Kept for backward compatibility but not used.
        MDSSF satellite fallback is evidence-backed as harmful.
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

    if boundary_predictions is None:
        boundary_predictions = {}

    # Build submission with routing
    rows = []
    interior_count = 0
    boundary_count = 0
    missing_count = 0
    
    for sid in all_ids:
        if sid in predictions:
            val = predictions[sid]
            interior_count += 1
        elif sid in boundary_predictions:
            val = boundary_predictions[sid]
            boundary_count += 1
        else:
            # Last resort -- should not happen in correct pipeline
            val = 0.0
            missing_count += 1
        rows.append({'ID': sid, 'TargetMBE': val, 'TargetRMSE': val})

    print(f"  Routing: {interior_count} interior (BiLSTM), "
          f"{boundary_count} boundary (LightGBM), "
          f"{missing_count} missing (0.0)")
    
    if missing_count > 0:
        print(f"  WARNING: {missing_count} test IDs had no prediction! "
              f"Check boundary detection logic.")

    submission = pd.DataFrame(rows)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"[PREDICT] Submission saved to {output_path}")
    print(f"  Shape: {submission.shape}")
    print(f"  TargetMBE range: [{submission['TargetMBE'].min():.2f}, "
          f"{submission['TargetMBE'].max():.2f}]")

    return submission


def stack_predictions(bilstm_preds, lgbm_residual_preds):
    """
    Sequential residual stacking: final = bilstm + lgbm_residual.
    
    NOT weighted blending. Evidence-backed: parallel blending degrades
    score when BiLSTM is the dominant predictor (Zindi 43->53).

    Parameters
    ----------
    bilstm_preds : dict or np.ndarray
        BiLSTM GHI predictions.
    lgbm_residual_preds : dict or np.ndarray
        LightGBM residual predictions (small corrections, ~±10-20 W/m2).

    Returns
    -------
    stacked : dict or np.ndarray (same type as input)
    """
    if isinstance(bilstm_preds, dict):
        stacked = {}
        for key in bilstm_preds:
            bi_val = bilstm_preds[key]
            # If LGBM has a residual for this key, add it
            lgb_res = lgbm_residual_preds.get(key, 0.0) if isinstance(lgbm_residual_preds, dict) else 0.0
            stacked[key] = max(0.0, bi_val + lgb_res)
        return stacked
    else:
        return np.maximum(0.0, bilstm_preds + lgbm_residual_preds)
