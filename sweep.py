"""
W&B Bayesian Hyperparameter Sweep with Sequential Residual LightGBM.

Supports three modes:
  1. python3 sweep.py                  -- Launch a new sweep (exploration)
  2. python3 sweep.py --resume SWEEP_ID -- Resume an existing sweep
  3. python3 sweep.py --refine          -- Narrow ranges from top-K prior runs (exploitation)

Architecture (evidence-backed, 4-source consensus):
  - BiLSTM predicts kt -> GHI (primary model)
  - LightGBM trains on OOF residuals: r = ghi_true - ghi_bilstm_OOF (sequential)
  - Boundary samples (symmetric window edges): LightGBM standalone predictions
  - NO MDSSF satellite fallback (evidence: costs ~20 Zindi points)

Global best model persistence:
  - A single 'global_best_model.pt' is maintained across ALL trials and sweeps.
  - Only overwritten when a new trial achieves a strictly better Zindi score.
  - Provenance (sweep_id, run_id, hparams) stored in JSON manifest.
"""

import os
import json
import time
import argparse
import shutil
import wandb
import gc
import numpy as np
import torch
import subprocess
import hashlib

def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()[:8]
    except:
        return 'unknown'

def get_feature_hash(feature_cols):
    return hashlib.md5(','.join(sorted(feature_cols)).encode()).hexdigest()[:8]

# W&B Login will be handled in the __main__ block via --api_key or environment variables.

from src.config import WANDB_CONFIG, HPARAMS, PATHS
from src.dataset import SolarDataset
from pipeline import build_pipeline_data, _extract_features_for_samples
from src.train import train_model
from src.predict import predict, generate_submission, stack_predictions
from src.dataset import create_test_dataset

# ------------------------------------------------------------------
# Ablation Control Flags (set by CLI -- do NOT edit directly)
# ------------------------------------------------------------------
# Ablation 1: Route symmetric window boundary samples to LGBM standalone
#             instead of MDSSF satellite fallback (evidence: ~20 Zindi points)
USE_BOUNDARY_LGBM = False  # requires USE_LGBM=True

# Ablation 2: Sequential residual LightGBM correction (BiLSTM + LGBM residual)
#             Replaces parallel sqrt-residual stacking
USE_LGBM = False

# Ablation 3: Bahdanau additive attention in BiLSTM
#             evidence: -7.9% RMSE (p<0.01)
USE_ATTENTION = False

# Ablation 4: Lean ~32-feature set (remove lu_* OHE, add 6 beyond-window features)
#             evidence: 8x hidden:feature ratio is optimal
USE_LEAN_FEATURES = False


# ------------------------------------------------------------------
# Global Best Model Manager
# ------------------------------------------------------------------
GLOBAL_BEST_DIR = os.path.join(PATHS['experiments_dir'], 'global_best')
GLOBAL_BEST_MODEL_PATH = os.path.join(GLOBAL_BEST_DIR, 'global_best_model.pt')
GLOBAL_BEST_MANIFEST_PATH = os.path.join(GLOBAL_BEST_DIR, 'global_best_manifest.json')


def _read_global_manifest():
    """Read the global best manifest, or return None if it does not exist."""
    if not os.path.exists(GLOBAL_BEST_MANIFEST_PATH):
        return None
    with open(GLOBAL_BEST_MANIFEST_PATH, 'r') as f:
        return json.load(f)


def _write_global_manifest(data: dict):
    """Atomically write the global best manifest (POSIX tmp+rename)."""
    tmp_path = GLOBAL_BEST_MANIFEST_PATH + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, GLOBAL_BEST_MANIFEST_PATH)


def try_update_global_best(checkpoint_path: str, val_zindi: float,
                           provenance: dict):
    """
    Attempt to update the global best model.
    Only overwrites if val_zindi is strictly better (lower).
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the trial's best_model.pt checkpoint.
    val_zindi : float
        The Zindi composite score (lower is better).
    provenance : dict
        Metadata: sweep_id, run_id, run_name, hparams, etc.
    
    Returns
    -------
    dict : {'updated': bool, 'prev_score': float or None}
    """
    os.makedirs(GLOBAL_BEST_DIR, exist_ok=True)
    
    manifest = _read_global_manifest()
    prev_score = manifest.get('val_zindi') if manifest else None
    
    if prev_score is None or val_zindi < prev_score:
        # Copy checkpoint to global best location
        shutil.copy2(checkpoint_path, GLOBAL_BEST_MODEL_PATH)
        
        # Also copy LightGBM and Calibration Ratios if present in the same folder!
        trial_dir = os.path.dirname(checkpoint_path)
        lgbm_src = os.path.join(trial_dir, 'stage2_lgbm.pkl')
        ratios_src = os.path.join(trial_dir, 'station_ratios.json')
        
        lgbm_dest = os.path.join(GLOBAL_BEST_DIR, 'stage2_lgbm.pkl')
        ratios_dest = os.path.join(GLOBAL_BEST_DIR, 'station_ratios.json')
        
        # Remove old ones if they exist
        if os.path.exists(lgbm_dest):
            os.remove(lgbm_dest)
        if os.path.exists(ratios_dest):
            os.remove(ratios_dest)
            
        if os.path.exists(lgbm_src):
            shutil.copy2(lgbm_src, lgbm_dest)
            print(f"[GLOBAL BEST] Copied Stage 2 LightGBM model to global best directory.")
        if os.path.exists(ratios_src):
            shutil.copy2(ratios_src, ratios_dest)
            print(f"[GLOBAL BEST] Copied Station Calibration Ratios to global best directory.")
        
        # Write manifest with provenance
        new_manifest = {
            'val_zindi': float(val_zindi),
            'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'provenance': provenance,
        }
        _write_global_manifest(new_manifest)
        
        # Log to W&B as artifact
        if wandb.run is not None:
            artifact = wandb.Artifact(
                name='global-best-model',
                type='model',
                metadata=new_manifest
            )
            artifact.add_file(GLOBAL_BEST_MODEL_PATH)
            
            # Add LightGBM and Station Ratios to the artifact too!
            if os.path.exists(lgbm_dest):
                artifact.add_file(lgbm_dest)
            if os.path.exists(ratios_dest):
                artifact.add_file(ratios_dest)
                
            wandb.log_artifact(artifact)
        
        print(f"[GLOBAL BEST] NEW BEST: {val_zindi:.4f} (prev: {prev_score})")
        return {'updated': True, 'prev_score': prev_score}
    else:
        print(f"[GLOBAL BEST] Not improved: {val_zindi:.4f} >= {prev_score:.4f}")
        return {'updated': False, 'prev_score': prev_score}


# ------------------------------------------------------------------
# Sweep Agent
# ------------------------------------------------------------------
def run_sweep_agent(config=None):
    """
    The function that W&B will call for each trial in the sweep.
    It reads the hyperparams from wandb.config and trains the model.
    
    If USE_LGBM is True, each trial also runs:
      1. BiLSTM training with OOF collection
      2. LightGBM sequential residual training on OOF residuals
      3. Additive residual correction + per-station calibration
      4. Final stacked Zindi score logged as the sweep metric
    """
    # Initialize wandb run
    with wandb.init(project=WANDB_CONFIG['project'], entity=WANDB_CONFIG['entity']):
        config = wandb.config
        config_dict = dict(config)
        
        # 1. IMMEDIATE UPDATE: HPARAMS must be updated before any other imports/objects
        HPARAMS.update(config_dict)
        
        # 2. Pass lambda_smooth to ZindiLoss via HPARAMS (if swept)
        # The train.py will read HPARAMS['lambda_smooth'] when constructing the loss
        
        print(f"\n[SWEEP] Starting trial with config: {config}")
        print(f"[SWEEP] LightGBM stacking: {'ENABLED' if USE_LGBM else 'DISABLED'}")
        
        # Load Data AFTER HPARAMS update to ensure correct seq_len
        df, feature_cols = build_pipeline_data()
        dataset = SolarDataset(df, feature_cols, is_train=True, hparams=HPARAMS)
        
        # Log metadata for reproducibility
        from src.config import SEED
        wandb.log({
            'meta/git_hash': get_git_hash(),
            'meta/feature_hash': get_feature_hash(feature_cols),
            'meta/seed': SEED,
            'meta/use_lgbm': USE_LGBM,
            'meta/use_boundary_lgbm': USE_BOUNDARY_LGBM,
            'meta/use_attention': USE_ATTENTION,
            'meta/use_lean_features': USE_LEAN_FEATURES,
        })
        
        # Pull experiments_dir from PATHS (not HPARAMS)
        run_name = wandb.run.name
        sweep_save_dir = os.path.join(PATHS['experiments_dir'], f"sweep_{run_name}")
        
        try:
            # Apply USE_ATTENTION config to HPARAMS before model construction
            if USE_ATTENTION:
                HPARAMS['use_attention'] = True
                HPARAMS['attn_dropout'] = config.get('attn_dropout', 0.1)
            else:
                HPARAMS['use_attention'] = False
            
            # ---- Stage 1: BiLSTM Training ----
            result = train_model(
                dataset, 
                feature_cols, 
                val_months=[3, 7, 11], # Standard competition temporal validation
                model_save_dir=sweep_save_dir,
                use_wandb=True,
                collect_oof=USE_LGBM,  # Collect OOF only when stacking
            )
            
            if USE_LGBM:
                model, history, best_model_path, oof_data = result
            else:
                model, history, best_model_path = result
                oof_data = None
            
            bilstm_zindi = min(history['val_zindi'])
            best_idx = np.argmin(history['val_zindi'])
            print(f"[SWEEP] BiLSTM best Zindi: {bilstm_zindi:.4f}")
            
            # ---- Stage 2: Sequential Residual LightGBM (if enabled) ----
            final_zindi = bilstm_zindi
            lgbm_model = None
            station_ratios = None
            
            if USE_LGBM and oof_data and len(oof_data) > 0:
                from src.stage2_lgbm import Stage2LightGBM
                from src.calibrate import compute_station_ratios, apply_calibration
                
                print(f"\n[SWEEP] Stage 2: Sequential Residual LightGBM on {len(oof_data)} OOF samples...")
                
                # Convert OOF to arrays
                oof_ghi_true = np.array([r['ghi_true'] for r in oof_data])
                oof_bilstm = np.array([r['ghi_pred'] for r in oof_data])
                oof_stations = np.array([r['station_idx'] for r in oof_data])
                oof_sample_ids = [r['sample_id'] for r in oof_data]
                
                # Build tabular features for OOF samples
                oof_features_df = _extract_features_for_samples(
                    df, dataset, oof_sample_ids, feature_cols
                )
                
                # Train LightGBM on SEQUENTIAL RESIDUALS (ghi_true - bilstm_OOF)
                lgbm_model = Stage2LightGBM()
                lgbm_model.fit(
                    ghi_true=oof_ghi_true,
                    features_df=oof_features_df,
                    bilstm_oof_preds=oof_bilstm,
                )
                
                # Save LightGBM model
                lgbm_save_path = os.path.join(sweep_save_dir, 'stage2_lgbm.pkl')
                lgbm_model.save(lgbm_save_path)
                
                # Get sequential residual-corrected OOF predictions
                oof_corrected = lgbm_model.predict_residual(
                    features_df=oof_features_df,
                    bilstm_preds=oof_bilstm,
                )
                
                # Per-station calibration ratios
                valid_oof = ~np.isnan(oof_ghi_true) & (oof_ghi_true > 0)
                station_ratios = compute_station_ratios(
                    y_true=oof_ghi_true[valid_oof],
                    y_pred=oof_corrected[valid_oof],
                    station_ids=oof_stations[valid_oof],
                )
                
                # Save station calibration ratios
                import json
                serializable_ratios = {str(k): float(v) for k, v in station_ratios.items()}
                ratios_save_path = os.path.join(sweep_save_dir, 'station_ratios.json')
                with open(ratios_save_path, 'w') as f:
                    json.dump(serializable_ratios, f, indent=2)
                
                # Calibrated OOF score (this is what the leaderboard will see)
                oof_calibrated = apply_calibration(
                    oof_corrected[valid_oof],
                    oof_stations[valid_oof],
                    station_ratios,
                )
                residuals = oof_calibrated - oof_ghi_true[valid_oof]
                stacked_mbe = float(np.abs(np.mean(residuals)))
                stacked_rmse = float(np.sqrt(np.mean(residuals ** 2)))
                final_zindi = 0.5 * stacked_mbe + 0.5 * stacked_rmse
                
                # Log residual statistics for monitoring
                oof_residuals = oof_ghi_true[valid_oof] - oof_bilstm[valid_oof]
                print(f"[SWEEP] BiLSTM OOF residual stats: "
                      f"mean={oof_residuals.mean():.2f}, std={oof_residuals.std():.2f}")
                print(f"[SWEEP] Sequential Zindi: {final_zindi:.4f} "
                      f"(BiLSTM-only: {bilstm_zindi:.4f}, "
                      f"improvement: {bilstm_zindi - final_zindi:.4f})")
                
                wandb.log({
                    'val/bilstm_zindi': bilstm_zindi,
                    'val/stacked_mbe': stacked_mbe,
                    'val/stacked_rmse': stacked_rmse,
                    'val/stacked_zindi': final_zindi,
                    'val/lgbm_improvement': bilstm_zindi - final_zindi,
                    'val/residual_mean': float(oof_residuals.mean()),
                    'val/residual_std': float(oof_residuals.std()),
                })
            
            # Log the FINAL score (stacked if LGBM, else BiLSTM-only)
            # This is what the Bayesian optimizer sees
            wandb.log({'val/zindi_score': final_zindi})
            print(f"[SWEEP] Trial finished. Final Zindi Score: {final_zindi:.4f}")

            # ---- Global Best Model Update ----
            provenance = {
                'sweep_id': getattr(wandb.run, 'sweep_id', None),
                'run_id': wandb.run.id,
                'run_name': run_name,
                'hparams': {k: v for k, v in config_dict.items()},
                'best_epoch': int(best_idx),
                'val_mbe': float(history['val_mbe'][best_idx]),
                'val_rmse': float(history['val_rmse'][best_idx]),
                'val_zindi': float(final_zindi),
                'use_lgbm': USE_LGBM,
            }
            try_update_global_best(best_model_path, final_zindi, provenance)

            # ---- Inference & Submission Logging ----
            print(f"[SWEEP] Generating submission for run {run_name}...")
            scaler_stats = dataset.get_scaler_stats()
            test_dataset = create_test_dataset(df, feature_cols, scaler_stats, hparams=HPARAMS)
            
            if USE_LGBM and lgbm_model is not None:
                # Stage 1: BiLSTM inference (interior samples only)
                bilstm_preds, test_details = predict(
                    test_dataset, models=[model], feature_cols=feature_cols,
                    return_details=True,
                )
                
                test_sample_ids = [d['sample_id'] for d in test_details]
                test_bilstm = np.array([d['ghi_pred'] for d in test_details])
                test_stations = np.array([d['station_idx'] for d in test_details])
                
                # Stage 2: Sequential residual correction for interior samples
                test_features_df = _extract_features_for_samples(
                    df, dataset, test_sample_ids, feature_cols
                )
                
                interior_corrected = lgbm_model.predict_residual(
                    features_df=test_features_df,
                    bilstm_preds=test_bilstm,
                )
                
                # Apply per-station calibration to interior
                calibrated_interior = apply_calibration(
                    interior_corrected, test_stations, station_ratios
                )
                
                interior_predictions = {}
                for i, sid in enumerate(test_sample_ids):
                    interior_predictions[sid] = float(calibrated_interior[i])
                
                # Stage 3: Boundary detection and LightGBM standalone prediction
                # Find test IDs not covered by BiLSTM (symmetric window boundaries)
                sample_sub_path = PATHS.get('sample_submission')
                if sample_sub_path and os.path.exists(sample_sub_path):
                    all_test_ids = set(pd.read_csv(sample_sub_path)['ID'].tolist())
                else:
                    all_test_ids = set()
                
                boundary_ids = all_test_ids - set(interior_predictions.keys())
                boundary_predictions = {}
                
                if boundary_ids and len(boundary_ids) > 0:
                    print(f"[SWEEP] Boundary routing: {len(boundary_ids)} samples "
                          f"to LightGBM standalone ({len(interior_predictions)} interior)")
                    
                    # Extract features for boundary samples from the main dataframe
                    test_mask = df['is_test'] == 1
                    boundary_mask = test_mask & df['ID'].isin(boundary_ids)
                    boundary_df = df.loc[boundary_mask].copy()
                    
                    if len(boundary_df) > 0:
                        # Get all columns that match the LGBM feature set
                        # Use all available feature columns (109-feature set)
                        all_feature_cols = [c for c in df.columns if c in (
                            lgbm_model.feature_cols or []) and c != 'bilstm_ghi_pred']
                        if not all_feature_cols:
                            # Fallback: use the same feature columns
                            all_feature_cols = [c for c in feature_cols if c in df.columns]
                        
                        boundary_features = boundary_df[all_feature_cols].copy()
                        boundary_features = boundary_features.fillna(0.0)
                        
                        # Predict direct GHI for boundary samples
                        boundary_ghi = lgbm_model.predict_boundary(
                            features_df=boundary_features,
                        )
                        
                        # Apply per-station calibration to boundary predictions
                        if 'station_idx' in boundary_df.columns:
                            boundary_stations = boundary_df['station_idx'].values
                            boundary_ghi = apply_calibration(
                                boundary_ghi, boundary_stations, station_ratios
                            )
                        
                        for i, sid in enumerate(boundary_df['ID'].values):
                            boundary_predictions[sid] = float(max(0.0, boundary_ghi[i]))
                        
                        print(f"[SWEEP] Boundary predictions generated: {len(boundary_predictions)}")
                else:
                    print(f"[SWEEP] No boundary samples detected (all covered by BiLSTM)")
                
                predictions = interior_predictions
            else:
                # BiLSTM-only inference (no LGBM)
                predictions = predict(test_dataset, models=[model], feature_cols=feature_cols)
                boundary_predictions = {}
            
            sub_filename = f"submission_{run_name}.csv"
            sub_path = os.path.join(PATHS['submissions_dir'], sub_filename)
            submission_df = generate_submission(
                predictions, 
                output_path=sub_path, 
                boundary_predictions=boundary_predictions,
            )
            
            # Log submission to W&B as an artifact
            artifact = wandb.Artifact(name=f"submission_{run_name}", type="submission")
            artifact.add_file(sub_path)
            wandb.log_artifact(artifact)
            
            # Log routing statistics
            wandb.log({
                'submission/interior_count': len(predictions),
                'submission/boundary_count': len(boundary_predictions),
            })
            print(f"[SWEEP] Submission logged to W&B: {sub_filename}")

        except Exception as e:
            print(f"[SWEEP] Trial failed with error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            del dataset, df
            gc.collect()


# ------------------------------------------------------------------
# Cross-Sweep Learning: Refine Ranges from Prior Runs
# ------------------------------------------------------------------
def compute_refined_ranges(project_name: str, entity: str, top_k: int = 10, api_key: str = None):
    """
    Query W&B API for top-K runs and compute narrowed hyperparameter ranges.
    """
    # Explicitly pass api_key to the API interface
    api = wandb.Api(api_key=api_key) if api_key else wandb.Api()
    path = f"{entity}/{project_name}" if entity else project_name
    
    try:
        # Use server-side filtering to avoid fetching thousands of failed runs
        filters = {"summary_metrics.val/zindi_score": {"$exists": True}}
        runs = api.runs(path, filters=filters, order="+summary_metrics.val/zindi_score")
    except Exception as e:
        print(f"[REFINE] Failed to query W&B API: {e}")
        return None
    
    # Filter runs that have the metric
    valid_runs = []
    for run in runs:
        if 'val/zindi_score' in run.summary:
            valid_runs.append(run)
    
    if len(valid_runs) < 3:
        print(f"[REFINE] Only {len(valid_runs)} valid runs found. Need at least 3.")
        return None
    
    # Sort by score (lower is better)
    # Sort by score (ensure numeric comparison)
    def get_score(run):
        try:
            val = run.summary.get('val/zindi_score')
            if val is None: return float('inf')
            return float(val)
        except (ValueError, TypeError):
            return float('inf')

    valid_runs.sort(key=get_score)
    top_runs = valid_runs[:top_k]
    
    print(f"[REFINE] Analyzing top {len(top_runs)} runs...")
    print(f"  Best score: {float(top_runs[0].summary['val/zindi_score']):.4f}")
    print(f"  Worst in top-K: {float(top_runs[-1].summary['val/zindi_score']):.4f}")
    
    refined = {}
    
    # Continuous parameters: narrow using IQR with 10% buffer
    continuous_params = ['lr', 'dropout', 'weight_decay']
    for p in continuous_params:
        values = [run.config.get(p) for run in top_runs if p in run.config]
        if len(values) >= 3:
            q10 = float(np.percentile(values, 10))
            q90 = float(np.percentile(values, 90))
            buffer = (q90 - q10) * 0.1 if q90 > q10 else abs(q10) * 0.1
            refined[p] = {
                'distribution': 'log_uniform_values' if p in ('lr', 'weight_decay') else 'uniform',
                'min': max(1e-7, q10 - buffer),
                'max': q90 + buffer,
            }
            print(f"  {p}: [{refined[p]['min']:.2e}, {refined[p]['max']:.2e}]")
    
    # Categorical parameters: keep only values that appeared in top-K
    categorical_params = ['hidden_dim', 'n_layers', 'station_embed_dim']
    for p in categorical_params:
        values = [run.config.get(p) for run in top_runs if p in run.config]
        if values:
            unique = sorted(set(values))
            refined[p] = {'values': unique}
            print(f"  {p}: {unique}")
    
    return refined


def normalize_sweep_id(sweep_id: str):
    """
    Ensure sweep_id is fully qualified (entity/project/id).
    W&B agents on remote clusters often fail to resolve short IDs.
    """
    if "/" in sweep_id:
        # Already qualified (or at least partially)
        parts = sweep_id.split("/")
        if len(parts) == 3:
            return sweep_id
        if len(parts) == 1:
            pass # fall through
            
    entity = WANDB_CONFIG['entity']
    project = WANDB_CONFIG['project']
    
    if not entity or not project:
        return sweep_id # Cannot normalize
        
    return f"{entity}/{project}/{sweep_id}"


# ------------------------------------------------------------------
# Sweep Launcher
# ------------------------------------------------------------------
def get_default_sweep_config():
    """Return the default (exploration) sweep configuration for BiLSTM.
    
    Search space designed for the PhysicsInformedBiLSTM:
      - hidden_dim: BiLSTM hidden size (output dim = 2x this, bidirectional)
      - n_layers: BiLSTM layers (2 is locked -- 3L causes gradient explosion)
      - dropout: regularization (BiLSTM is more sensitive than Transformers)
      - lr: learning rate for AdamW
      - weight_decay: L2 regularization
      - station_embed_dim: station embedding dimension
      - lambda_smooth: kt regularization weight in ZindiLoss
      - use_attention: whether to use Bahdanau additive attention
      - attn_dropout: dropout on attention weights (overfitting prevention)
    """
    return {
        'method': 'bayes',
        'metric': {
            'name': 'val/zindi_score',
            'goal': 'minimize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 8,
            's': 2
        },
        'parameters': {
            # BiLSTM Architecture
            'hidden_dim':        {'values': [128, 160, 192, 256]},
            'n_layers':          {'value': 2},  # LOCKED: 3L causes gradient explosion
            'dropout':           {'distribution': 'uniform', 'min': 0.05, 'max': 0.30},
            'station_embed_dim': {'values': [8, 16, 32]},

            # Attention (Bahdanau additive, evidence-backed)
            'use_attention':     {'value': True},  # Enable for attention ablation
            'attn_dropout':      {'distribution': 'uniform', 'min': 0.05, 'max': 0.20},

            # Optimization
            'lr':                {'distribution': 'log_uniform_values', 'min': 3e-4, 'max': 3e-3},
            'weight_decay':      {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-3},

            # ZindiLoss: kt smoothness penalty (solar-sweep-1 used ~0.008)
            'lambda_smooth':     {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-2},
        }
    }


def start_sweep(resume_id: str = None, refine: bool = False, count: int = 40, api_key: str = None):
    """
    Launch a sweep with one of three modes:
    
    1. New sweep (default): Broad Bayesian exploration.
    2. Resume (--resume SWEEP_ID): Continue an existing sweep's Bayesian state.
    3. Refine (--refine): Query top-K runs and launch a narrowed exploitation sweep.
    """
    os.makedirs(GLOBAL_BEST_DIR, exist_ok=True)
    
    if resume_id:
        # Mode 2: Resume existing sweep
        print(f"[SWEEP] Resuming sweep: {resume_id}")
        sweep_id = resume_id
    elif refine:
        # Mode 3: Refine from prior runs
        print("[SWEEP] Refining sweep ranges from prior runs...")
        refined = compute_refined_ranges(
            WANDB_CONFIG['project'],
            WANDB_CONFIG.get('entity'),
            top_k=10,
            api_key=api_key
        )
        
        if refined is None:
            print("[SWEEP] Falling back to default sweep config.")
            sweep_config = get_default_sweep_config()
        else:
            sweep_config = get_default_sweep_config()
            sweep_config['parameters'].update(refined)
            print("[SWEEP] Refined sweep config applied.")
    else:
        # Mode 1: New exploration sweep
        sweep_config = get_default_sweep_config()
    
    # 4. START AGENT
    # ------------------------------------------------------------------
    # Detect GPU device for logging
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    print(f"\n[SWEEP] Agent starting on Device: {device}:{gpu_id}")

    # Initialize or join existing sweep
    # Priority: 1. CLI --resume, 2. Env WANDB_SWEEP_ID, 3. Create New
    final_sweep_id = resume_id or os.environ.get("WANDB_SWEEP_ID")
    
    if not final_sweep_id:
        print(f"[SWEEP] Creating NEW sweep (Refine={refine})...")
        final_sweep_id = wandb.sweep(
            sweep_config,
            project=WANDB_CONFIG['project'],
            entity=WANDB_CONFIG['entity']
        )
    else:
        print(f"[SWEEP] Joining EXISTING sweep ID: {final_sweep_id}")

    # Start the agent with fully qualified sweep ID to avoid 404 on CSCS
    full_sweep_id = normalize_sweep_id(final_sweep_id)
    print(f"[SWEEP] Agent joining: {full_sweep_id}")
    
    wandb.agent(full_sweep_id, function=run_sweep_agent, count=count)
    print("[SWEEP] Sweep completed.")
    
    # Print final global best
    manifest = _read_global_manifest()
    if manifest:
        print(f"\n[SWEEP] FINAL GLOBAL BEST: Zindi={manifest['val_zindi']:.4f}")
        print(f"  Run: {manifest['provenance'].get('run_name', 'unknown')}")
        print(f"  MBE: {manifest['provenance'].get('val_mbe', 'N/A')}")
        print(f"  RMSE: {manifest['provenance'].get('val_rmse', 'N/A')}")
        print(f"  Model: {GLOBAL_BEST_MODEL_PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="W&B Bayesian Sweep with Cross-Sweep Learning"
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Resume an existing sweep by ID (keeps Bayesian optimizer state).'
    )
    parser.add_argument(
        '--refine', action='store_true',
        help='Narrow ranges from top-K prior runs (exploitation mode).'
    )
    parser.add_argument(
        '--count', type=int, default=40,
        help='Number of trials to run (default: 40).'
    )
    parser.add_argument(
        '--api_key', type=str, default=None,
        help='W&B API Key for authentication (useful for Colab).'
    )
    parser.add_argument(
        '--with-lgbm', action='store_true',
        help='Enable LightGBM sequential residual stacking + boundary routing (Ablation 1+2).'
    )
    parser.add_argument(
        '--with-attention', action='store_true',
        help='Enable Bahdanau additive attention in BiLSTM (Ablation 3).'
    )
    parser.add_argument(
        '--with-lean-features', action='store_true',
        help='Enable lean ~32 feature set (remove lu_* OHE, add beyond-window features) (Ablation 4).'
    )
    parser.add_argument(
        '--ablation', type=str, default=None,
        choices=['1', '2', '3', '4', '12', '123', '1234'],
        help='Shorthand: 1=boundary(requires lgbm), 2=lgbm+boundary, 3=+attention, 4=+lean features.'
    )
    args = parser.parse_args()

    # Set module-level ablation flags from CLI args
    USE_LGBM = args.with_lgbm
    USE_BOUNDARY_LGBM = args.with_lgbm  # Boundary requires LGBM
    USE_ATTENTION = args.with_attention
    USE_LEAN_FEATURES = args.with_lean_features

    # --ablation shorthand OVERRIDES individual args (cumulative: each level adds to previous)
    if args.ablation:
        abl = args.ablation
        if '1' in abl or '2' in abl:
            USE_LGBM = True
            USE_BOUNDARY_LGBM = True
        if '3' in abl:
            USE_ATTENTION = True
        if '4' in abl:
            USE_LEAN_FEATURES = True

    # Apply USE_LEAN_FEATURES: set HPARAMS flag so dataset.py can read it
    HPARAMS['use_lean_features'] = USE_LEAN_FEATURES

    # Print active ablation config
    print("\n" + "=" * 60)
    print("ABLATION CONFIGURATION")
    print("=" * 60)
    print(f"  Ablation 1 (Boundary LGBM):    {'ENABLED' if USE_BOUNDARY_LGBM else 'DISABLED'}")
    print(f"  Ablation 2 (Sequential LGBM):  {'ENABLED' if USE_LGBM else 'DISABLED'}")
    print(f"  Ablation 3 (Attention):        {'ENABLED' if USE_ATTENTION else 'DISABLED'}")
    print(f"  Ablation 4 (Lean Features):    {'ENABLED' if USE_LEAN_FEATURES else 'DISABLED'}")
    print("=" * 60 + "\n")

    # Handle Login
    api_key = args.api_key or os.environ.get('WANDB_API_KEY')
    try:
        if api_key:
            wandb.login(key=api_key)
            print("[SWEEP] Successfully logged into W&B via provided key.")
        else:
            # Native discovery: checks ~/.netrc, then ~/.config/wandb/
            wandb.login()
            print("[SWEEP] Successfully logged into W&B via native credentials (.netrc).")
    except Exception as e:
        print(f"[SWEEP] W&B Login failed: {e}")
        if os.environ.get('WANDB_MODE') != 'disabled':
            print("[SWEEP] Check your ~/.netrc or WANDB_API_KEY environment variable.")

    start_sweep(resume_id=args.resume, refine=args.refine, count=args.count, api_key=api_key)
