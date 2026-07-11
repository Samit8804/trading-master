"""Model training with hyperparameter tuning via grid search.

The primary model is XGBoost (regressor) because:

  1. **Handles mixed data types** — categorical (one-hot), continuous, and
     missing values without explicit imputation.
  2. **Robust to collinearity** — tree-based models are not affected by
     correlated features, unlike linear models.
  3. **Feature importance** — built-in gain / weight / cover importance
     for interpretability.
  4. **Regularisation** — L1 (``reg_alpha``) and L2 (``reg_lambda``)
     to prevent overfitting.
  5. **State of the art** — consistently top-performing on tabular data.

Hyperparameter rationale
------------------------
* ``n_estimators`` — more trees reduce bias but increase overfitting risk.
  Grid: 100, 200, 300.
* ``max_depth`` — deeper trees capture more complex interactions but
  overfit more easily. Grid: 4, 6, 8.
* ``learning_rate`` — lower rates require more trees but generalise better.
  Grid: 0.01, 0.05, 0.10.
* ``subsample`` — fraction of rows sampled per tree. < 1.0 prevents
  overfitting. Grid: 0.7, 0.8, 1.0.
* ``colsample_bytree`` — fraction of columns sampled per tree. Similar
  regularisation effect. Grid: 0.7, 0.8, 1.0.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from crowdwisdom_quant.config import Config

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and tune an XGBoost regressor on trading features."""

    def __init__(
        self,
        model_dir: Path | None = None,
        params: Dict[str, Any] | None = None,
        search_grid: Dict[str, list] | None = None,
    ) -> None:
        self.model_dir = (model_dir or Config.PROJECT_ROOT / "models").resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.params = params or Config.MODEL_DEFAULTS
        self.search_grid = search_grid or Config.HYPERPARAM_GRID
        self._best_model: Optional[XGBRegressor] = None
        self._best_params: Optional[Dict[str, Any]] = None
        self._cv_results: Optional[Dict[str, Any]] = None
        self._feature_importances: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        tune: bool = True,
        model_name: str = "xgboost_model.joblib",
    ) -> XGBRegressor:
        """Train (and optionally tune) the XGBoost model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series or np.ndarray
            Target (PnL or win-rate indicator).
        tune : bool
            If True, run grid-search cross-validation to select best
            hyperparameters before training the final model.
        model_name : str
            Filename for the saved Joblib dump.

        Returns
        -------
        XGBRegressor
            Trained model.
        """
        y_arr = np.asarray(y, dtype=np.float64)

        if tune:
            logger.info("Running hyperparameter grid search ...")
            self._run_grid_search(X, y_arr)
            best_params = self._best_params
            logger.info("Best hyperparameters found: %s", best_params)
        else:
            best_params = self.params

        # Train final model on full training set
        logger.info("Training final XGBoost model ...")
        train_params = {
            k: v for k, v in best_params.items() if k != "random_state"
        }
        model = XGBRegressor(**train_params, random_state=Config.RANDOM_SEED, importance_type="gain")
        model.fit(X, y_arr)

        self._best_model = model
        self._store_feature_importances(model, X.columns)
        self._save(model, model_name)

        return model

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------
    def _run_grid_search(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Perform grid-search with 3-fold cross-validation.

        We use ``neg_mean_squared_error`` as the scoring metric because
        it aligns with the primary regression objective (minimising
        prediction error).
        """
        base = XGBRegressor(random_state=Config.RANDOM_SEED, importance_type="gain", verbosity=0)
        gs = GridSearchCV(
            estimator=base,
            param_grid=self.search_grid,
            cv=TimeSeriesSplit(n_splits=3),
            scoring="neg_mean_squared_error",
            n_jobs=min(4, __import__("os").cpu_count() or 1),
            verbose=0,
        )
        gs.fit(X, y)

        self._best_params = gs.best_params_
        self._cv_results = gs.cv_results_

        logger.info(
            "Grid search complete. Best MSE: %.6f",
            -gs.best_score_,
        )

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    def _store_feature_importances(
        self, model: XGBRegressor, feature_names: pd.Index
    ) -> None:
        """Build a DataFrame of feature importances sorted by 'gain'."""
        importances = model.feature_importances_
        self._feature_importances = (
            pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def print_feature_importance(self, top_n: int = 20) -> None:
        """Print the top-N most important features."""
        if self._feature_importances is None:
            logger.warning("No feature importances available. Train the model first.")
            return
        print("\n=== Feature Importance (gain) ===")
        print(self._feature_importances.head(top_n).to_string(index=False))
        print()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save(self, model: XGBRegressor, name: str) -> None:
        path = self.model_dir / name
        joblib.dump(model, path)
        logger.info("Model saved to %s", path)

    def load(self, model_name: str = "xgboost_model.joblib") -> XGBRegressor:
        """Load a previously trained model from disk."""
        path = self.model_dir / model_name
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        model: XGBRegressor = joblib.load(path)
        self._best_model = model
        logger.info("Model loaded from %s", path)
        return model

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def best_model(self) -> Optional[XGBRegressor]:
        return self._best_model

    @property
    def best_params(self) -> Optional[Dict[str, Any]]:
        return self._best_params

    @property
    def feature_importances(self) -> Optional[pd.DataFrame]:
        return self._feature_importances
