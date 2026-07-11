"""Prediction module: apply a trained model to make out-of-sample forecasts.

Design
------
* The ``Predictor`` class wraps a trained ``XGBRegressor`` (or any
  scikit-learn-compatible estimator) and provides ``predict()``.
* It can also build a **prediction table** that aligns actual vs. predicted
  PnL with timestamps, strategy, and fold metadata for downstream analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from crowdwisdom_quant.models.trainer import ModelTrainer

logger = logging.getLogger(__name__)


class Predictor:
    """Apply a trained model to make predictions."""

    def __init__(self, trainer: ModelTrainer | None = None) -> None:
        self._trainer = trainer
        self._model: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def model(self) -> Any:
        if self._model is None and self._trainer is not None:
            self._model = self._trainer.best_model
        return self._model

    @model.setter
    def model(self, value: Any) -> None:
        self._model = value

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted PnL for feature matrix ``X``."""
        if self.model is None:
            raise RuntimeError(
                "No model available. Train or load a model first."
            )
        return self.model.predict(X)

    def build_prediction_table(
        self,
        X: pd.DataFrame,
        y_actual: pd.Series | np.ndarray,
        timestamps: pd.Series,
        strategies: pd.Series,
        fold: int,
    ) -> pd.DataFrame:
        """Build a detailed prediction table for analysis / visualisation.

        Columns: Timestamp, Actual_PnL, Predicted_PnL, Strategy, Hour, Day, Fold
        """
        y_pred = self.predict(X)
        y_act = np.asarray(y_actual)

        table = pd.DataFrame(
            {
                "Timestamp": timestamps.values,
                "Actual_PnL": y_act,
                "Predicted_PnL": y_pred,
                "Strategy": strategies.values.astype(str),
                "Hour": pd.to_datetime(timestamps).dt.hour.values,
                "Day": pd.to_datetime(timestamps).dt.day_name().values,
                "Fold": fold,
            }
        )
        return table
