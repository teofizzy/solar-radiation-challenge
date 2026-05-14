"""
W&B Bayesian Hyperparameter Sweep with Adaptive Cross-Sweep Learning.

Supports three modes:
  1. python3 sweep.py                  -- Launch a new sweep (exploration)
  2. python3 sweep.py --resume SWEEP_ID -- Resume an existing sweep
  3. python3 sweep.py --refine          -- Narrow ranges from top-K prior runs (exploitation)

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

# W&B Login will be handled in the __main__ block via --api_key or environment variables.

from src.config import WANDB_CONFIG, HPARAMS, PATHS
from src.dataset import SolarDataset
from pipeline import build_pipeline_data
from src.train import train_model
from src.predict import predict, generate_submission
from src.dataset import create_test_dataset


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
    """
    # Initialize wandb run
    with wandb.init(project=WANDB_CONFIG['project'], entity=WANDB_CONFIG['entity']):
        config = wandb.config
        config_dict = dict(config)
        
        # 1. IMMEDIATE UPDATE: HPARAMS must be updated before any other imports/objects
        HPARAMS.update(config_dict)
        
        print(f"\n[SWEEP] Starting trial with config: {config}")
        
        # Compatibility Correction: hidden_dim must be divisible by transformer_heads
        heads = HPARAMS.get('transformer_heads', 6)
        if HPARAMS['hidden_dim'] % heads != 0:
            old_dim = HPARAMS['hidden_dim']
            HPARAMS['hidden_dim'] = ((old_dim // heads) + 1) * heads
            print(f"[SWEEP] Adjusted hidden_dim for compatibility: {old_dim} -> {HPARAMS['hidden_dim']} (divisible by {heads} heads)")
            
        # Load Data AFTER HPARAMS update to ensure correct seq_len
        df, feature_cols = build_pipeline_data()
        dataset = SolarDataset(df, feature_cols, is_train=True, hparams=HPARAMS)
        
        # Pull experiments_dir from PATHS (not HPARAMS)
        run_name = wandb.run.name
        sweep_save_dir = os.path.join(PATHS['experiments_dir'], f"sweep_{run_name}")
        
        try:
            model, history, best_model_path = train_model(
                dataset, 
                feature_cols, 
                val_months=[3, 7, 11], # Standard competition temporal validation
                model_save_dir=sweep_save_dir,
                use_wandb=True
            )
            
            best_zindi = min(history['val_zindi'])
            print(f"[SWEEP] Trial finished. Best Zindi Score: {best_zindi:.4f}")

            # ---- Global Best Model Update ----
            provenance = {
                'sweep_id': getattr(wandb.run, 'sweep_id', None),
                'run_id': wandb.run.id,
                'run_name': run_name,
                'hparams': {k: v for k, v in config_dict.items()},
                'best_epoch': int(np.argmin(history['val_zindi'])),
                'val_mbe': float(history['val_mbe'][np.argmin(history['val_zindi'])]),
                'val_rmse': float(history['val_rmse'][np.argmin(history['val_zindi'])]),
            }
            try_update_global_best(best_model_path, best_zindi, provenance)

            # ---- Inference & Submission Logging ----
            print(f"[SWEEP] Generating submission for run {run_name}...")
            scaler_stats = dataset.get_scaler_stats()
            test_dataset = create_test_dataset(df, feature_cols, scaler_stats, hparams=HPARAMS)
            
            predictions = predict(test_dataset, models=[model], feature_cols=feature_cols)
            
            sub_filename = f"submission_{run_name}.csv"
            sub_path = os.path.join(PATHS['submissions_dir'], sub_filename)
            submission_df = generate_submission(predictions, output_path=sub_path)
            
            # Log submission to W&B as an artifact
            artifact = wandb.Artifact(name=f"submission_{run_name}", type="submission")
            artifact.add_file(sub_path)
            wandb.log_artifact(artifact)
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
def compute_refined_ranges(project_name: str, entity: str, top_k: int = 10):
    """
    Query W&B API for top-K runs and compute narrowed hyperparameter ranges.
    
    Returns a dict of parameter -> {'min': float, 'max': float} for continuous
    params, or {'values': list} for categorical params.
    """
    api = wandb.Api()
    path = f"{entity}/{project_name}" if entity else project_name
    
    try:
        runs = api.runs(path, order="+summary_metrics.val/zindi_score")
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
    valid_runs.sort(key=lambda x: x.summary['val/zindi_score'])
    top_runs = valid_runs[:top_k]
    
    print(f"[REFINE] Analyzing top {len(top_runs)} runs...")
    print(f"  Best score: {top_runs[0].summary['val/zindi_score']:.4f}")
    print(f"  Worst in top-K: {top_runs[-1].summary['val/zindi_score']:.4f}")
    
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
    categorical_params = ['hidden_dim', 'n_layers', 'transformer_heads']
    for p in categorical_params:
        values = [run.config.get(p) for run in top_runs if p in run.config]
        if values:
            unique = sorted(set(values))
            refined[p] = {'values': unique}
            print(f"  {p}: {unique}")
    
    return refined


# ------------------------------------------------------------------
# Sweep Launcher
# ------------------------------------------------------------------
def get_default_sweep_config():
    """Return the default (exploration) sweep configuration."""
    return {
        'method': 'bayes',
        'metric': {
            'name': 'val/zindi_score_ema',
            'goal': 'minimize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 2,
            'eta': 3,
            'max_iter': 40
        },
        'parameters': {
            'lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 5e-3
            },
            'hidden_dim': {
                'values': [128, 192, 256]
            },
            'n_layers': {
                'values': [3, 4, 6]
            },
            'transformer_heads': {
                'values': [4, 6, 8]
            },
            'dropout': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.3
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-6,
                'max': 1e-4
            }
        }
    }


def start_sweep(resume_id: str = None, refine: bool = False, count: int = 40):
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
            top_k=10
        )
        
        if refined is None:
            print("[SWEEP] Falling back to default sweep config.")
            sweep_config = get_default_sweep_config()
        else:
            sweep_config = get_default_sweep_config()
            sweep_config['parameters'].update(refined)
            print("[SWEEP] Refined sweep config applied.")
        
        sweep_id = wandb.sweep(
            sweep_config,
            project=WANDB_CONFIG['project'],
            entity=WANDB_CONFIG['entity']
        )
    else:
        # Mode 1: New exploration sweep
        sweep_config = get_default_sweep_config()
        
        print("[SWEEP] Initializing NEW exploration sweep...")
        sweep_id = wandb.sweep(
            sweep_config,
            project=WANDB_CONFIG['project'],
            entity=WANDB_CONFIG['entity']
        )
    
    print(f"[SWEEP] Sweep ID: {sweep_id}")
    print(f"[SWEEP] Global best model dir: {GLOBAL_BEST_DIR}")
    
    # Check existing global best
    manifest = _read_global_manifest()
    if manifest:
        print(f"[SWEEP] Existing global best: Zindi={manifest['val_zindi']:.4f} "
              f"(from {manifest['provenance'].get('run_name', 'unknown')})")
    else:
        print("[SWEEP] No existing global best. First trial will establish baseline.")
    
    print(f"[SWEEP] Starting agent ({count} trials)...")
    wandb.agent(sweep_id, function=run_sweep_agent, count=count)
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
    args = parser.parse_args()
    
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
    
    start_sweep(resume_id=args.resume, refine=args.refine, count=args.count)
