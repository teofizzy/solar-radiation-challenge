import os
import argparse
import wandb
import gc

# Attempt Colab authentication early
try:
    from google.colab import userdata
    wandb_api_key = userdata.get('WANDB_API_KEY')
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        print("[SWEEP] Successfully logged into W&B via Colab Secrets.")
except ImportError:
    pass
except Exception as e:
    print(f"[SWEEP] Colab secrets not available: {e}")

from src.config import WANDB_CONFIG, HPARAMS, PATHS
from src.dataset import SolarDataset
from pipeline import build_pipeline_data
from src.train import train_model

def run_sweep_agent():
    """
    The function that W&B will call for each trial in the sweep.
    It reads the hyperparams from wandb.config and trains the model.
    """
    # Initialize wandb run
    with wandb.init(project=WANDB_CONFIG['project'], entity=WANDB_CONFIG['entity']):
        config = wandb.config
        
        # We need to load data first
        print(f"\n[SWEEP] Starting trial with config: {config}")
        df, feature_cols = build_pipeline_data()
        # SolarDataset reads seq_len from HPARAMS internally; no kwarg needed.
        dataset = SolarDataset(df, feature_cols, is_train=True)
        
        # Train model with current config
        # wandb.config is a special object; convert to plain dict before updating HPARAMS.
        HPARAMS.update(dict(config))
        
        # Pull experiments_dir from PATHS (not HPARAMS)
        run_name = wandb.run.name
        sweep_save_dir = os.path.join(PATHS['experiments_dir'], f"sweep_{run_name}")
        
        try:
            model, history = train_model(
                dataset, 
                feature_cols, 
                val_months=[3, 7, 11], # Standard competition temporal validation
                model_save_dir=sweep_save_dir,
                use_wandb=True
            )
            print(f"[SWEEP] Trial finished. Best Zindi Score: {min(history['val_zindi']):.4f}")
        except Exception as e:
            print(f"[SWEEP] Trial failed with error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            del dataset, df
            gc.collect()

def start_sweep():
    # Define Sweep Config as recommended by Proxima for Colab + PyTorch BiLSTM
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val/zindi_score',
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
                'values': [32, 64, 128]
            },
            'dropout': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.4
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-6,
                'max': 1e-4
            }
        }
    }
    
    print("[SWEEP] Initializing Sweep...")
    sweep_id = wandb.sweep(sweep_config, project=WANDB_CONFIG['project'], entity=WANDB_CONFIG['entity'])
    print(f"[SWEEP] Sweep ID: {sweep_id}")
    
    # Run the agent (budget: max 40 trials)
    # The max_iter in hyperband is epochs, but we can limit total runs using count
    print("[SWEEP] Starting agent...")
    wandb.agent(sweep_id, function=run_sweep_agent, count=40)
    print("[SWEEP] Sweep completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run W&B Bayesian Sweep for PyTorch BiLSTM")
    args = parser.parse_args()
    start_sweep()
