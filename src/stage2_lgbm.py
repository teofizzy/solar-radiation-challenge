"""
Stage 2: LightGBM Residual Calibration.

Trains a gradient boosted decision tree on the residuals from Stage 1
(PatchTransformer physics predictions). This corrects for structured 
biases that the Transformer cannot efficiently model:
  - Per-station sensor drift
  - Aerosol/cloud regime-specific offsets
  - Hour-of-day systematic errors
  - Categorical station effects

Inspired by:
  - XGBoost + 60-day rolling window (17.01 W/m2 RMSE, Oak Ridge paper)
  - WTX-TPE-LGBM (DWT + Transformer + LightGBM residual correction)

Usage:
    # After Stage 1 training, collect OOF predictions
    stage2 = Stage2LightGBM()
    stage2.fit(oof_ghi_physics, oof_ghi_true, oof_features)
    
    # At inference time
    ghi_final = stage2.predict(test_ghi_physics, test_features)
"""

import os
import numpy as np
import pandas as pd
import joblib

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[STAGE2] WARNING: lightgbm not installed. Stage 2 disabled.")

from src.config import SEED


# Features used by LightGBM Stage 2
STAGE2_FEATURE_COLS = [
    'ghi_physics',         # Stage 1 prediction (primary feature)
    'station_idx',         # Categorical: per-station calibration
    'hour_sin',            # Diurnal cycle
    'hour_cos',
    'cos_zenith',          # Solar geometry
    'kt_landsaf',          # Satellite clearness index
    'clear_sky_ghi',       # Upper bound
    'elevation',           # Static topographic
]

# Optional features (added if available)
STAGE2_OPTIONAL_COLS = [
    'tropomi_aerosol',     # Aerosol loading
    'tropomi_cloud',       # Cloud fraction
    'drift_proxy',         # Sensor drift estimate
    'tcwv',                # Total column water vapour
    'wind_speed',          # Atmospheric dynamics
    'clearness_mismatch',  # kt_obs - kt_landsaf (train only)
]


class Stage2LightGBM:
    """LightGBM residual corrector for Stage 2 of the Two-Stage pipeline.
    
    Trains on: residual = ghi_true - ghi_physics (Stage 1 output)
    Predicts:  correction to add to Stage 1 physics prediction
    
    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Step size shrinkage.
    """
    
    def __init__(self, n_estimators=500, max_depth=8, learning_rate=0.01):
        if not HAS_LGBM:
            raise ImportError("lightgbm is required for Stage 2. Install with: pip install lightgbm")
        
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_samples=20,
            random_state=SEED,
            verbose=-1,
            n_jobs=-1,
        )
        self.feature_cols = None
        self.is_fitted = False
    
    def _prepare_features(self, ghi_physics, features_df):
        """Build feature matrix from Stage 1 predictions and tabular features.
        
        Parameters
        ----------
        ghi_physics : np.ndarray (N,)
            Stage 1 GHI predictions.
        features_df : pd.DataFrame
            Tabular features indexed to match ghi_physics.
        
        Returns
        -------
        X : pd.DataFrame
            Feature matrix for LightGBM.
        """
        X = pd.DataFrame({'ghi_physics': ghi_physics})
        
        # Add available features
        for col in STAGE2_FEATURE_COLS[1:] + STAGE2_OPTIONAL_COLS:
            if col in features_df.columns:
                X[col] = features_df[col].values
        
        # Derived features
        if 'clear_sky_ghi' in X.columns:
            csg = X['clear_sky_ghi'].values
            X['clearness_pred'] = np.where(csg > 1.0, ghi_physics / csg, 0.0)
        
        self.feature_cols = list(X.columns)
        
        # Handle NaN
        X = X.fillna(0.0)
        
        return X
    
    def fit(self, ghi_physics, ghi_true, features_df, 
            categorical_features=None):
        """Train LightGBM on Stage 1 residuals.
        
        Parameters
        ----------
        ghi_physics : np.ndarray (N,)
            Stage 1 GHI predictions (OOF recommended).
        ghi_true : np.ndarray (N,)
            True GHI values.
        features_df : pd.DataFrame
            Tabular features.
        categorical_features : list or None
            Categorical feature names for LightGBM.
        """
        X = self._prepare_features(ghi_physics, features_df)
        
        # Target: residual = true - physics
        residuals = ghi_true - ghi_physics
        
        # Remove NaN targets
        valid = ~np.isnan(residuals) & ~np.isnan(ghi_true)
        X_valid = X.loc[valid]
        y_valid = residuals[valid]
        
        # Identify categorical features
        cat_features = []
        if 'station_idx' in X_valid.columns:
            cat_features.append('station_idx')
        if categorical_features:
            cat_features.extend(categorical_features)
        
        print(f"[STAGE2] Training LightGBM on {len(X_valid)} samples, "
              f"{len(self.feature_cols)} features")
        print(f"  Residual stats: mean={y_valid.mean():.2f}, "
              f"std={y_valid.std():.2f}, "
              f"median={np.median(y_valid):.2f}")
        
        # Convert categorical columns to int for LightGBM
        for col in cat_features:
            if col in X_valid.columns:
                X_valid[col] = X_valid[col].astype(int)
        
        self.model.fit(
            X_valid, y_valid,
            categorical_feature=cat_features if cat_features else 'auto',
        )
        self.is_fitted = True
        
        # Log feature importance
        importance = self.model.feature_importances_
        feat_imp = sorted(zip(self.feature_cols, importance), 
                         key=lambda x: x[1], reverse=True)
        print(f"[STAGE2] Top-5 feature importances:")
        for name, imp in feat_imp[:5]:
            print(f"  {name}: {imp}")
        
        # In-sample residual correction check
        pred_residuals = self.model.predict(X_valid)
        corrected = ghi_physics[valid] + pred_residuals
        corrected_rmse = np.sqrt(np.mean((corrected - ghi_true[valid])**2))
        uncorrected_rmse = np.sqrt(np.mean((ghi_physics[valid] - ghi_true[valid])**2))
        print(f"[STAGE2] In-sample RMSE: {uncorrected_rmse:.2f} -> {corrected_rmse:.2f} "
              f"({uncorrected_rmse - corrected_rmse:.2f} reduction)")
    
    def predict(self, ghi_physics, features_df):
        """Apply Stage 2 correction to Stage 1 predictions.
        
        Parameters
        ----------
        ghi_physics : np.ndarray (N,)
            Stage 1 GHI predictions.
        features_df : pd.DataFrame
            Tabular features.
        
        Returns
        -------
        ghi_final : np.ndarray (N,)
            Corrected GHI predictions.
        """
        if not self.is_fitted:
            print("[STAGE2] WARNING: Model not fitted. Returning Stage 1 predictions.")
            return ghi_physics
        
        X = self._prepare_features(ghi_physics, features_df)
        
        # Convert categorical columns to int
        if 'station_idx' in X.columns:
            X['station_idx'] = X['station_idx'].astype(int)
        
        residual_pred = self.model.predict(X)
        ghi_final = ghi_physics + residual_pred
        
        # Physical constraints: non-negative
        ghi_final = np.clip(ghi_final, 0, None)
        
        return ghi_final
    
    def save(self, path):
        """Save fitted model to disk."""
        joblib.dump({
            'model': self.model,
            'feature_cols': self.feature_cols,
            'is_fitted': self.is_fitted,
        }, path)
        print(f"[STAGE2] Model saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load a fitted model from disk."""
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.feature_cols = data['feature_cols']
        instance.is_fitted = data['is_fitted']
        print(f"[STAGE2] Model loaded from {path}")
        return instance
