"""
Stage 2: LightGBM Sequential Residual Correction.

Architecture (4-source evidence-backed):
    LGBM trains on BiLSTM OOF RESIDUALS (sequential boosting):
        target = ghi_true - ghi_bilstm_OOF  (raw GHI residual space)
    
    At inference (interior samples with BiLSTM predictions):
        ghi_final = ghi_bilstm + lgbm_residual
    
    At inference (boundary samples without BiLSTM predictions):
        ghi_final = lgbm_standalone  (direct GHI prediction)

Key design decisions (evidence-based, 4-source consensus):
    1. SEQUENTIAL residual architecture (not parallel): LGBM corrects BiLSTM
       errors, not satellite errors. Parallel stacking DEGRADES score (43->53).
    2. Raw GHI residuals (not sqrt): simpler, residuals are small (~10-20 W/m2)
       when BiLSTM is strong, no variance stabilization needed.
    3. Uses ALL 109 features + bilstm_ghi_pred as extra feature.
    4. Boundary mode: direct GHI prediction when no BiLSTM pred available.
       bilstm_ghi_pred = NaN for boundary samples (LightGBM handles missing).

References:
    - FXL3 paper: 3-stage pipeline with sequential residual boosting
    - WTX-TPE-LGBM: Transformer + LightGBM sequential residual refinement
    - 4/4 AI sources: sequential residual correct when NN is dominant
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

from src.config import SEED, STAGE2_HPARAMS


class Stage2LightGBM:
    """LightGBM sequential residual corrector.
    
    Trains on: ghi_true - ghi_bilstm_OOF (sequential residuals)
    Predicts:  residual correction (additive) for interior samples
               direct GHI for boundary samples (standalone mode)
    
    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds (default: from STAGE2_HPARAMS).
    max_depth : int
        Maximum tree depth (default: from STAGE2_HPARAMS).
    learning_rate : float
        Step size shrinkage (default: from STAGE2_HPARAMS).
    """
    
    def __init__(self, n_estimators=None, max_depth=None, learning_rate=None):
        if not HAS_LGBM:
            raise ImportError("lightgbm is required for Stage 2. Install with: pip install lightgbm")
        
        n_estimators = n_estimators or STAGE2_HPARAMS['n_estimators']
        max_depth = max_depth or STAGE2_HPARAMS['max_depth']
        learning_rate = learning_rate or STAGE2_HPARAMS['learning_rate']
        
        # Residual corrector (sequential)
        self.residual_model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=STAGE2_HPARAMS.get('num_leaves', 31),
            subsample=STAGE2_HPARAMS.get('subsample', 0.8),
            colsample_bytree=STAGE2_HPARAMS.get('colsample_bytree', 0.8),
            reg_alpha=STAGE2_HPARAMS.get('reg_alpha', 0.1),
            reg_lambda=STAGE2_HPARAMS.get('reg_lambda', 0.1),
            min_child_samples=STAGE2_HPARAMS.get('min_child_samples', 20),
            random_state=SEED,
            verbose=-1,
            n_jobs=-1,
        )
        
        # Standalone model for boundary samples (direct GHI prediction)
        self.boundary_model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=STAGE2_HPARAMS.get('num_leaves', 31),
            subsample=STAGE2_HPARAMS.get('subsample', 0.8),
            colsample_bytree=STAGE2_HPARAMS.get('colsample_bytree', 0.8),
            reg_alpha=STAGE2_HPARAMS.get('reg_alpha', 0.1),
            reg_lambda=STAGE2_HPARAMS.get('reg_lambda', 0.1),
            min_child_samples=STAGE2_HPARAMS.get('min_child_samples', 20),
            random_state=SEED,
            verbose=-1,
            n_jobs=-1,
        )
        
        self.feature_cols = None
        self.is_fitted = False
        self.boundary_fitted = False
    
    def _prepare_features(self, features_df, bilstm_preds=None):
        """Build feature matrix from tabular features and optional BiLSTM predictions.
        
        Parameters
        ----------
        features_df : pd.DataFrame
            Full 109-feature tabular data.
        bilstm_preds : np.ndarray or None
            BiLSTM GHI predictions (added as extra feature if provided).
            For boundary samples, this should be NaN (LightGBM handles missing).
        
        Returns
        -------
        X : pd.DataFrame
            Feature matrix for LightGBM.
        """
        X = features_df.copy()
        
        # Add BiLSTM prediction as extra feature
        # For boundary samples, this will be NaN -- LightGBM handles missing natively
        if bilstm_preds is not None:
            X['bilstm_ghi_pred'] = bilstm_preds
        elif 'bilstm_ghi_pred' not in X.columns:
            X['bilstm_ghi_pred'] = np.nan
        
        self.feature_cols = list(X.columns)
        
        return X
    
    def fit(self, ghi_true, features_df, bilstm_oof_preds,
            categorical_features=None):
        """Train LightGBM on sequential residuals from BiLSTM OOF predictions.
        
        Parameters
        ----------
        ghi_true : np.ndarray (N,)
            True GHI values.
        features_df : pd.DataFrame
            All 109 tabular features.
        bilstm_oof_preds : np.ndarray (N,)
            BiLSTM OUT-OF-FOLD predictions. MUST be OOF to avoid leakage.
        categorical_features : list or None
            Categorical feature names.
        """
        X = self._prepare_features(features_df, bilstm_oof_preds)
        
        # Target: sequential residual (ghi_true - bilstm_OOF)
        residuals = ghi_true - bilstm_oof_preds
        
        # Remove invalid samples
        valid = (~np.isnan(residuals) & ~np.isnan(ghi_true) & 
                 ~np.isnan(bilstm_oof_preds) & (ghi_true > 0))
        X_valid = X.loc[valid].copy()
        y_residuals = residuals[valid]
        
        # Identify categorical features
        cat_features = []
        if 'station_idx' in X_valid.columns:
            cat_features.append('station_idx')
        if categorical_features:
            cat_features.extend(categorical_features)
        
        print(f"[STAGE2] Training RESIDUAL LightGBM on {len(X_valid)} daytime samples, "
              f"{len(self.feature_cols)} features")
        print(f"  Residual stats: mean={y_residuals.mean():.4f}, "
              f"std={y_residuals.std():.4f}, "
              f"median={np.median(y_residuals):.4f}")
        
        # Convert categorical columns to int for LightGBM
        for col in cat_features:
            if col in X_valid.columns:
                X_valid[col] = X_valid[col].astype(int)
        
        # Train residual model
        self.residual_model.fit(
            X_valid, y_residuals,
            categorical_feature=cat_features if cat_features else 'auto',
        )
        self.is_fitted = True
        
        # Also train boundary model on direct GHI (same data, different target)
        print(f"[STAGE2] Training BOUNDARY LightGBM (direct GHI) on {len(X_valid)} samples...")
        y_direct = ghi_true[valid]
        
        # For boundary model, bilstm_ghi_pred = NaN to simulate boundary conditions
        X_boundary = X_valid.copy()
        X_boundary['bilstm_ghi_pred'] = np.nan
        
        self.boundary_model.fit(
            X_boundary, y_direct,
            categorical_feature=cat_features if cat_features else 'auto',
        )
        self.boundary_fitted = True
        
        # Log feature importance (residual model)
        importance = self.residual_model.feature_importances_
        feat_imp = sorted(zip(self.feature_cols, importance), 
                         key=lambda x: x[1], reverse=True)
        print(f"[STAGE2] Residual model top-10 feature importances:")
        for name, imp in feat_imp[:10]:
            print(f"  {name}: {imp}")
        
        # Quality check: residual correction vs uncorrected
        pred_residuals = self.residual_model.predict(X_valid)
        ghi_corrected = bilstm_oof_preds[valid] + pred_residuals
        ghi_corrected = np.maximum(ghi_corrected, 0.0)
        
        corrected_rmse = np.sqrt(np.mean((ghi_corrected - ghi_true[valid])**2))
        uncorrected_rmse = np.sqrt(np.mean((bilstm_oof_preds[valid] - ghi_true[valid])**2))
        print(f"[STAGE2] In-sample RMSE: BiLSTM-only={uncorrected_rmse:.2f} -> "
              f"BiLSTM+LGBM={corrected_rmse:.2f} "
              f"({uncorrected_rmse - corrected_rmse:.2f} reduction)")
        
        # Quality check: boundary model
        pred_boundary = self.boundary_model.predict(X_boundary)
        pred_boundary = np.maximum(pred_boundary, 0.0)
        boundary_rmse = np.sqrt(np.mean((pred_boundary - ghi_true[valid])**2))
        print(f"[STAGE2] Boundary model in-sample RMSE: {boundary_rmse:.2f}")
    
    def predict_residual(self, features_df, bilstm_preds, clear_sky_ghi=None):
        """Predict GHI correction for INTERIOR samples (have BiLSTM predictions).
        
        Returns ghi_final = bilstm_pred + lgbm_residual.
        
        Parameters
        ----------
        features_df : pd.DataFrame
            All 109 tabular features.
        bilstm_preds : np.ndarray (N,)
            BiLSTM GHI predictions for these samples.
        clear_sky_ghi : np.ndarray (N,) or None
            Clear-sky GHI for upper bound clamping.
        
        Returns
        -------
        ghi_final : np.ndarray (N,)
            Corrected GHI predictions (bilstm + residual).
        """
        if not self.is_fitted:
            print("[STAGE2] WARNING: Residual model not fitted. Returning BiLSTM as-is.")
            return np.maximum(bilstm_preds, 0.0)
        
        X = self._prepare_features(features_df, bilstm_preds)
        
        # Convert categorical columns to int
        if 'station_idx' in X.columns:
            X['station_idx'] = X['station_idx'].astype(int)
        
        # Predict residual and add to BiLSTM
        residual = self.residual_model.predict(X)
        ghi_final = bilstm_preds + residual
        
        # Physical constraints
        ghi_final = np.maximum(ghi_final, 0.0)
        if clear_sky_ghi is not None:
            upper_bound = 1.3 * np.maximum(clear_sky_ghi, 0.0)
            ghi_final = np.minimum(ghi_final, upper_bound)
        
        return ghi_final
    
    def predict_boundary(self, features_df, clear_sky_ghi=None):
        """Predict GHI for BOUNDARY samples (no BiLSTM prediction available).
        
        Uses standalone LightGBM trained on direct GHI with bilstm_pred=NaN.
        
        Parameters
        ----------
        features_df : pd.DataFrame
            All 109 tabular features for boundary samples.
        clear_sky_ghi : np.ndarray (N,) or None
            Clear-sky GHI for upper bound clamping.
        
        Returns
        -------
        ghi_pred : np.ndarray (N,)
            Direct GHI predictions for boundary samples.
        """
        if not self.boundary_fitted:
            print("[STAGE2] WARNING: Boundary model not fitted. Returning 0.")
            return np.zeros(len(features_df))
        
        X = self._prepare_features(features_df)
        # bilstm_ghi_pred = NaN for boundary samples (set in _prepare_features)
        
        # Convert categorical columns to int
        if 'station_idx' in X.columns:
            X['station_idx'] = X['station_idx'].astype(int)
        
        ghi_pred = self.boundary_model.predict(X)
        
        # Physical constraints
        ghi_pred = np.maximum(ghi_pred, 0.0)
        if clear_sky_ghi is not None:
            upper_bound = 1.3 * np.maximum(clear_sky_ghi, 0.0)
            ghi_pred = np.minimum(ghi_pred, upper_bound)
        
        return ghi_pred
    
    # Keep backward-compatible predict() method
    def predict(self, features_df, bilstm_preds=None, clear_sky_ghi=None,
                mdssf=None):
        """Backward-compatible predict. Routes to residual or boundary mode.
        
        If bilstm_preds is provided: sequential residual mode.
        If bilstm_preds is None: boundary standalone mode.
        
        The mdssf parameter is accepted but ignored (kept for backward compat).
        """
        if bilstm_preds is not None:
            return self.predict_residual(features_df, bilstm_preds, clear_sky_ghi)
        else:
            return self.predict_boundary(features_df, clear_sky_ghi)
    
    def save(self, path):
        """Save fitted models to disk."""
        joblib.dump({
            'residual_model': self.residual_model,
            'boundary_model': self.boundary_model,
            'feature_cols': self.feature_cols,
            'is_fitted': self.is_fitted,
            'boundary_fitted': self.boundary_fitted,
        }, path)
        print(f"[STAGE2] Model saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load fitted models from disk."""
        data = joblib.load(path)
        instance = cls()
        instance.residual_model = data['residual_model']
        instance.boundary_model = data.get('boundary_model', data.get('model'))
        instance.feature_cols = data['feature_cols']
        instance.is_fitted = data['is_fitted']
        instance.boundary_fitted = data.get('boundary_fitted', False)
        print(f"[STAGE2] Model loaded from {path}")
        return instance
