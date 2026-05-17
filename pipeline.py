"""
TAHMO Solar Radiation Challenge -- End-to-End Pipeline
Physics-Informed BiLSTM + LightGBM Parallel Sqrt-Residual Stacking

This script orchestrates the full pipeline:
Phase A (MVP): Config -> Data Loading -> Astro -> Dataset -> Model -> Train -> Predict
Phase B (Enrichment): ERA5 -> Physics -> Temporal features
Phase C (Satellite): LandSAF, TROPOMI, Static priors
Phase D (Stacking): LightGBM sqrt-residual -> Ensemble -> Calibration

Architecture:
    Stage 1: BiLSTM (direct kt prediction, ZindiLoss) -> Zindi ~45
    Stage 2: LightGBM (parallel sqrt-residual from MDSSF) -> Zindi 35-40
    Stage 3: Weighted ensemble (0.4*BiLSTM + 0.6*LightGBM) + per-station calibration -> Zindi 30-35

Usage (Colab):
    %run pipeline.py
    %run pipeline.py --with-lgbm    # Full pipeline with LightGBM stacking

Usage (local):
    python pipeline.py
    python pipeline.py --with-lgbm
"""

import os
import sys
import gc
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# 0. Configuration and seeding
# ------------------------------------------------------------------
from src.config import (
    SEED, seed_everything, PATHS, HPARAMS, STAGE2_HPARAMS, FEATURES,
    ensure_dirs, get_station_meta
)

seed_everything(SEED)
ensure_dirs()

print("=" * 70)
print("TAHMO Solar Radiation Challenge -- Physics-Informed BiLSTM Pipeline")
print("=" * 70)
print(f"  Seed: {SEED}")
print(f"  Feature toggles: {FEATURES}")
print(f"  Seq len: {HPARAMS['seq_len']} ({HPARAMS['half_window']} past + "
      f"{HPARAMS['half_window']} future)")
print(f"  Architecture: BiLSTM (hidden={HPARAMS['hidden_dim']}, "
      f"layers={HPARAMS['n_layers']})")
print(f"  Loss: ZindiLoss (lambda_smooth={HPARAMS.get('lambda_smooth', 0.001)})")
print(f"  Precision: FP32 (AMP disabled)")

from src.data_loader import load_raw_data
from src.feature_astro import compute_astro_features

def build_pipeline_data():
    """Run all feature engineering phases and return the final dataframe and feature columns."""
    print("\n" + "=" * 70)
    print("PHASE A.2: Data Loading")
    print("=" * 70)

    df = load_raw_data(force_recompute=False)
    stations = sorted(df['station'].unique())
    print(f"  Loaded {len(df):,} rows, {len(stations)} stations")

    # ------------------------------------------------------------------
    # 2. Astronomical Features (Phase A.3 -- MOST CRITICAL)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE A.3: Astronomical Features (pvlib Ineichen)")
    print("=" * 70)

    df = compute_astro_features(df, force_recompute=False)
    gc.collect()

    # ------------------------------------------------------------------
    # 3. ERA5 Features (Phase B.1)
    # ------------------------------------------------------------------
    if FEATURES['use_era5']:
        from src.feature_era5 import compute_era5_features

        print("\n" + "=" * 70)
        print("PHASE B.1: ERA5 Reanalysis Features (PCHIP interpolation)")
        print("=" * 70)

        df = compute_era5_features(df, force_recompute=False)
        gc.collect()

    # ------------------------------------------------------------------
    # 3.5 Static Topographic Features (Phase C)
    # ------------------------------------------------------------------
    if FEATURES.get('use_static', False):
        from src.feature_static import compute_static_features

        print("\n" + "=" * 70)
        print("PHASE C: Static Topographic Prior Features")
        print("=" * 70)

        df = compute_static_features(df, force_recompute=False)
        gc.collect()

    # ------------------------------------------------------------------
    # 4. Physics-Derived Features (Phase B.2)
    # ------------------------------------------------------------------
    if FEATURES['use_physics']:
        from src.feature_physics import compute_physics_features

        print("\n" + "=" * 70)
        print("PHASE B.2: Physics-Derived Features")
        print("=" * 70)

        df = compute_physics_features(df, force_recompute=False)
        gc.collect()

    # ------------------------------------------------------------------
    # 4.5. LandSAF Features (Phase C)
    # ------------------------------------------------------------------
    if FEATURES['use_landsaf']:
        from src.feature_landsaf import compute_landsaf_features

        print("\n" + "=" * 70)
        print("PHASE C: LandSAF Satellite Features")
        print("=" * 70)

        df = compute_landsaf_features(df, force_recompute=False)
        gc.collect()

    # ------------------------------------------------------------------
    # 4.6. TROPOMI Features (Phase D)
    # ------------------------------------------------------------------
    if FEATURES['use_tropomi']:
        from src.feature_tropomi import compute_tropomi_features

        print("\n" + "=" * 70)
        print("PHASE D: TROPOMI Satellite Features (Cloud & Aerosol)")
        print("=" * 70)

        df = compute_tropomi_features(df, force_recompute=False)
        gc.collect()

    # ------------------------------------------------------------------
    # 5. Temporal Features (Phase B.3)
    # ------------------------------------------------------------------
    if FEATURES['use_temporal']:
        from src.feature_temporal import compute_temporal_features

        print("\n" + "=" * 70)
        print("PHASE B.3: Temporal Features (covariates only)")
        print("=" * 70)

        df = compute_temporal_features(df, force_recompute=False)
        gc.collect()

        # ------------------------------------------------------------------
        # 5b. Interaction Features (Phase B.4)
        # ------------------------------------------------------------------
        from src.feature_interaction import compute_interaction_features
        print("\n" + "=" * 70)
        print("PHASE B.4: Multiplicative Physics Interactions")
        print("=" * 70)
        df = compute_interaction_features(df)
        gc.collect()

    # ------------------------------------------------------------------
    # 6. Dataset Construction (Phase A.4)
    # ------------------------------------------------------------------
    from src.dataset import get_feature_columns
    from src.utils import enforce_schema

    print("\n" + "=" * 70)
    print("PHASE A.4: Dataset Construction")
    print("=" * 70)

    feature_cols = get_feature_columns(df)
    
    import hashlib
    feat_hash = hashlib.md5(','.join(sorted(feature_cols)).encode()).hexdigest()[:8]
    
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"  Feature Hash: {feat_hash}")
    
    return df, feature_cols

def run_standard_pipeline(with_lgbm=False):
    """Run the full pipeline: BiLSTM training + optional LightGBM stacking."""
    df, feature_cols = build_pipeline_data()
    
    from src.dataset import SolarDataset
    dataset = SolarDataset(df, feature_cols, is_train=True)
    scaler_stats = dataset.get_scaler_stats()
    
    # ------------------------------------------------------------------
    # 7. Training (Phase A.7): BiLSTM 2-Fold CV
    # ------------------------------------------------------------------
    from src.train import train_model

    print("\n" + "=" * 70)
    print("PHASE A.7: Training BiLSTM (2-Fold CV)")
    print("=" * 70)

    folds = [
        {'name': 'fold_1', 'val_months': [3, 7, 11]},
        {'name': 'fold_2', 'val_months': [1, 5, 9]}
    ]

    trained_models = []
    all_histories = []
    all_oof_data = []  # Collect OOF for LightGBM

    for fold in folds:
        print(f"\n--- Training {fold['name'].upper()} ---")
        model_save_dir = os.path.join(PATHS['experiments_dir'], fold['name'])
        
        result = train_model(
            dataset=dataset,
            feature_cols=feature_cols,
            val_months=fold['val_months'],
            model_save_dir=model_save_dir,
            collect_oof=with_lgbm,  # Only collect OOF if LightGBM is enabled
        )
        
        if with_lgbm:
            model, history, _, oof_data = result
            all_oof_data.extend(oof_data)
        else:
            model, history, _ = result
        
        trained_models.append(model)
        all_histories.append(history)

    # ------------------------------------------------------------------
    # 8. LightGBM Stage 2 (Phase A.8) -- if enabled
    # ------------------------------------------------------------------
    lgbm_model = None
    if with_lgbm and all_oof_data:
        from src.stage2_lgbm import Stage2LightGBM

        print("\n" + "=" * 70)
        print("PHASE A.8: LightGBM Parallel Sqrt-Residual Stacking")
        print("=" * 70)

        # Convert OOF to arrays
        oof_ghi_true = np.array([r['ghi_true'] for r in all_oof_data])
        oof_mdssf = np.array([r['mdssf_ghi'] for r in all_oof_data])
        oof_bilstm = np.array([r['ghi_pred'] for r in all_oof_data])
        oof_stations = np.array([r['station_idx'] for r in all_oof_data])
        oof_sample_ids = [r['sample_id'] for r in all_oof_data]

        # Build tabular features for OOF samples
        # Map sample_ids back to rows in df to extract tabular features
        oof_features_df = _extract_features_for_samples(
            df, dataset, oof_sample_ids, feature_cols
        )

        # Train LightGBM
        lgbm_model = Stage2LightGBM()
        lgbm_model.fit(
            ghi_true=oof_ghi_true,
            mdssf=oof_mdssf,
            features_df=oof_features_df,
            bilstm_preds=oof_bilstm,
        )
        
        # Save LightGBM model
        lgbm_save_path = os.path.join(PATHS['experiments_dir'], 'stage2_lgbm.pkl')
        lgbm_model.save(lgbm_save_path)

        # Compute calibration ratios on OOF validation data
        from src.calibrate import compute_station_ratios

        print("\n" + "=" * 70)
        print("PHASE A.8b: Per-Station Calibration (OOF)")
        print("=" * 70)

        # Get LGBM predictions on OOF for calibration
        oof_lgbm_preds = lgbm_model.predict(
            mdssf=oof_mdssf,
            features_df=oof_features_df,
            bilstm_preds=oof_bilstm,
        )

        # Ensemble OOF predictions
        from src.predict import stack_predictions
        oof_stacked = stack_predictions(oof_bilstm, oof_lgbm_preds)

        # Compute per-station calibration ratios
        valid_oof = ~np.isnan(oof_ghi_true) & (oof_ghi_true > 0)
        station_ratios = compute_station_ratios(
            y_true=oof_ghi_true[valid_oof],
            y_pred=oof_stacked[valid_oof],
            station_ids=oof_stations[valid_oof],
        )

        gc.collect()

    # ------------------------------------------------------------------
    # 9. Prediction (Phase A.9)
    # ------------------------------------------------------------------
    from src.predict import predict, generate_submission, stack_predictions

    print("\n" + "=" * 70)
    print("PHASE A.9: Prediction and Submission")
    print("=" * 70)

    if with_lgbm and lgbm_model is not None:
        # Full pipeline: BiLSTM + LightGBM + Calibration
        bilstm_preds, test_details = predict(
            dataset=dataset, models=trained_models, return_details=True
        )

        # Build test features for LGBM
        test_sample_ids = [d['sample_id'] for d in test_details]
        test_mdssf = np.array([d['mdssf_ghi'] for d in test_details])
        test_bilstm = np.array([d['ghi_pred'] for d in test_details])
        test_stations = np.array([d['station_idx'] for d in test_details])

        test_features_df = _extract_features_for_samples(
            df, dataset, test_sample_ids, feature_cols
        )

        # LightGBM predictions on test
        lgbm_test_preds = lgbm_model.predict(
            mdssf=test_mdssf,
            features_df=test_features_df,
            bilstm_preds=test_bilstm,
        )

        # Ensemble
        stacked_test = stack_predictions(test_bilstm, lgbm_test_preds)

        # Apply per-station calibration
        from src.calibrate import apply_calibration
        calibrated_test = apply_calibration(stacked_test, test_stations, station_ratios)

        # Build final predictions dict
        predictions = {}
        for i, sid in enumerate(test_sample_ids):
            predictions[sid] = float(calibrated_test[i])

        # Log quality metrics
        bilstm_mean = np.mean(test_bilstm)
        lgbm_mean = np.mean(lgbm_test_preds)
        stacked_mean = np.mean(stacked_test)
        cal_mean = np.mean(calibrated_test)
        print(f"\n[PIPELINE] Prediction means:")
        print(f"  BiLSTM:      {bilstm_mean:.2f} W/m2")
        print(f"  LightGBM:    {lgbm_mean:.2f} W/m2")
        print(f"  Stacked:     {stacked_mean:.2f} W/m2")
        print(f"  Calibrated:  {cal_mean:.2f} W/m2")
    else:
        # BiLSTM-only pipeline
        predictions = predict(dataset=dataset, models=trained_models)

    submission = generate_submission(predictions)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    for i, fold in enumerate(folds):
        history = all_histories[i]
        best_idx = np.argmin(history['val_zindi'])
        print(f"  {fold['name'].upper()} Best Zindi: {history['val_zindi'][best_idx]:.4f} "
              f"(MBE: {history['val_mbe'][best_idx]:.2f}, RMSE: {history['val_rmse'][best_idx]:.2f})")
    if with_lgbm:
        print(f"  Stage 2: LightGBM + Calibration ACTIVE")
        print(f"  Ensemble weights: BiLSTM={STAGE2_HPARAMS['w_bilstm']}, "
              f"LightGBM={STAGE2_HPARAMS['w_lgbm']}")
    print(f"  Submission rows:      {len(submission):,}")
    print(f"  Submission saved to:  {PATHS['submissions_dir']}")


def _extract_features_for_samples(df, dataset, sample_ids, feature_cols):
    """Extract tabular features for a list of sample IDs from the dataset.
    
    Extracts the center-timestep raw feature values from the dataset's
    feat_matrix for LightGBM input. These are UN-normalized values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full pipeline dataframe.
    dataset : SolarDataset
        The windowed dataset.
    sample_ids : list
        Sample IDs to extract features for.
    feature_cols : list
        Feature column names.
    
    Returns
    -------
    features_df : pd.DataFrame
        Tabular features indexed to match sample_ids.
    """
    # Build a lookup: sample_id -> index in dataset.samples
    id_to_idx = {}
    for i, sample in enumerate(dataset.samples):
        sid = sample.get('sample_id', '')
        if sid:
            id_to_idx[sid] = i
    
    # Extract center-timestep features for each sample
    rows = []
    for sid in sample_ids:
        idx = id_to_idx.get(sid)
        if idx is not None:
            sample = dataset.samples[idx]
            center = sample['center']
            feat_matrix = sample['feat_matrix']  # (n_rows, n_features) raw values
            
            # Extract center timestep feature vector
            center_features = feat_matrix[center]  # (n_features,)
            
            row = {}
            for j, col in enumerate(feature_cols):
                row[col] = float(center_features[j]) if j < len(center_features) else 0.0
            # Add station_idx for categorical feature
            row['station_idx'] = sample.get('station_idx', 0)
            rows.append(row)
        else:
            # Fallback: zeros
            row = {col: 0.0 for col in feature_cols}
            row['station_idx'] = 0
            rows.append(row)
    
    result = pd.DataFrame(rows)
    # Replace NaN with 0 for LightGBM
    result = result.fillna(0.0)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TAHMO Solar Radiation Pipeline")
    parser.add_argument('--sweep', action='store_true', help="Run in Sweep Mode (HPO)")
    parser.add_argument('--with-lgbm', action='store_true', 
                        help="Enable Stage 2 LightGBM stacking + calibration")
    parser.add_argument('--wandb_test', action='store_true', help="Dry run for W&B integration")
    args = parser.parse_args()

    if args.sweep:
        print("[PIPELINE] Launching Sweep Mode...")
        from sweep import start_sweep
        start_sweep()
    else:
        run_standard_pipeline(with_lgbm=args.with_lgbm)
