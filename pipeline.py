"""
TAHMO Solar Radiation Challenge -- End-to-End Pipeline
Physics-Informed BiLSTM for Solar Radiation Reconstruction (V1 Reverted)

This script orchestrates the full pipeline:
Phase A (MVP): Config -> Data Loading -> Astro -> Dataset -> Model -> Train -> Predict
Phase B (Enrichment): ERA5 -> Physics -> Temporal features
Phase C (Satellite): LandSAF, TROPOMI, Static priors

Architecture: BiLSTM (direct kt prediction) + LightGBM residual stacking

Usage (Colab):
    %run pipeline.py

Usage (local):
    python pipeline.py
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
    SEED, seed_everything, PATHS, HPARAMS, FEATURES,
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

def run_standard_pipeline():
    df, feature_cols = build_pipeline_data()
    
    from src.dataset import SolarDataset
    dataset = SolarDataset(df, feature_cols, is_train=True)
    scaler_stats = dataset.get_scaler_stats()
    
    # ------------------------------------------------------------------
    # 7. Training (Phase A.7)
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

    for fold in folds:
        print(f"\n--- Training {fold['name'].upper()} ---")
        model_save_dir = os.path.join(PATHS['experiments_dir'], fold['name'])
        
        model, history, _ = train_model(
            dataset=dataset,
            feature_cols=feature_cols,
            val_months=fold['val_months'],
            model_save_dir=model_save_dir
        )
        trained_models.append(model)
        all_histories.append(history)

    # ------------------------------------------------------------------
    # 8. Prediction (Phase A.8)
    # ------------------------------------------------------------------
    from src.predict import predict, generate_submission

    print("\n" + "=" * 70)
    print("PHASE A.8: Prediction and Submission (Ensemble)")
    print("=" * 70)

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
    print(f"  Submission rows:      {len(submission):,}")
    print(f"  Submission saved to:  {PATHS['submissions_dir']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TAHMO Solar Radiation Pipeline")
    parser.add_argument('--sweep', action='store_true', help="Run in Sweep Mode (HPO)")
    parser.add_argument('--wandb_test', action='store_true', help="Dry run for W&B integration")
    args = parser.parse_args()

    if args.sweep:
        print("[PIPELINE] Launching Sweep Mode...")
        from sweep import start_sweep
        start_sweep()
    else:
        run_standard_pipeline()
