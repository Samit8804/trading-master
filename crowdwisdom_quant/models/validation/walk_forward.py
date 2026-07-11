"""Walk-forward validation module.

This is the **most critical component** for avoiding time-series leakage.

Methodology
-----------
1. The dataset is sorted chronologically.
2. For each fold:
   - **Train**: first ``train_days`` days of data.
   - **Test**: next ``test_days`` days of data.
3. The window slides forward by ``test_days`` (non-overlapping test folds).
4. Within each fold, features are re-scaled and rolling statistics are
   recomputed **on the training fold only** to prevent any leakage.
5. Metrics are recorded for every fold; the final output includes averaged
   metrics and a complete prediction table.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from crowdwisdom_quant.config.settings import Config
from crowdwisdom_quant.features.engineering import FeatureEngineer
from crowdwisdom_quant.models.metrics import compute_metrics, metrics_dataframe
from crowdwisdom_quant.models.predictor import Predictor
from crowdwisdom_quant.models.trainer import ModelTrainer

logger = logging.getLogger(__name__)
console = Console(stderr=True, highlight=False)


class WalkForwardValidator:
    """Execute walk-forward validation and collect per-fold results."""

    def __init__(
        self,
        trainer: ModelTrainer | None = None,
        predictor: Predictor | None = None,
        train_days: int | None = None,
        test_days: int | None = None,
        expanding_window: bool = False,
    ) -> None:
        self.trainer = trainer or ModelTrainer()
        self.predictor = predictor or Predictor()
        self.train_days = train_days or Config.TRAIN_DAYS
        self.test_days = test_days or Config.TEST_DAYS
        self.expanding_window = expanding_window

        self._metrics_list: List[Dict[str, float]] = []
        self._prediction_tables: List[pd.DataFrame] = []
        self._fold_results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        data: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run walk-forward validation on the provided merged dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Merged trade+macro dataset sorted chronologically (must contain
            ``timestamp`` and ``pnl`` columns).

        Returns
        -------
        metrics_df : pd.DataFrame
            Per-fold metrics with a final ``MEAN`` row.
        prediction_table : pd.DataFrame
            Concatenated prediction table across all folds.
        """
        data = data.sort_values("timestamp").reset_index(drop=True)
        folds = self._build_folds(data)

        for fold_idx, (train_df, test_df) in enumerate(folds):
            logger.info(
                "Fold %d: train %d rows (%s to %s), test %d rows (%s to %s)",
                fold_idx,
                len(train_df),
                train_df["timestamp"].min(),
                train_df["timestamp"].max(),
                len(test_df),
                test_df["timestamp"].min(),
                test_df["timestamp"].max(),
            )

            # Feature engineering within the fold
            fe = FeatureEngineer()  # fresh scaler per fold
            X_train, y_train = fe.fit_transform(train_df, fit_scaler=True)
            X_test, y_test = fe.transform(test_df)

            if X_test.empty:
                logger.warning("Fold %d: empty test set, skipping.", fold_idx)
                continue

            # Train on this fold's training data
            model = self.trainer.train(
                X_train, y_train, tune=(fold_idx == 0), model_name=f"fold_{fold_idx}.joblib"
            )
            self.predictor.model = model

            # Predict on this fold's test data
            y_pred = self.predictor.predict(X_test)

            # Metrics (pass timestamps for correct Sharpe annualisation)
            metrics = compute_metrics(
                y_test.values, y_pred, timestamps=test_df["timestamp"].values
            )
            metrics["fold"] = fold_idx
            self._metrics_list.append(metrics)

            # Prediction table
            table = pd.DataFrame(
                {
                    "Timestamp": test_df["timestamp"].values,
                    "Actual_PnL": y_test.values,
                    "Predicted_PnL": y_pred,
                    "Strategy": test_df["strategy_permutation"].values.astype(str),
                    "Hour": pd.to_datetime(test_df["timestamp"]).dt.hour.values,
                    "Day": pd.to_datetime(test_df["timestamp"]).dt.day_name().values,
                    "Fold": fold_idx,
                }
            )
            self._prediction_tables.append(table)

            self._fold_results.append(
                {
                    "fold": fold_idx,
                    "train_start": train_df["timestamp"].min(),
                    "train_end": train_df["timestamp"].max(),
                    "test_start": test_df["timestamp"].min(),
                    "test_end": test_df["timestamp"].max(),
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                }
            )

        metrics_df = metrics_dataframe(self._metrics_list)
        prediction_table = (
            pd.concat(self._prediction_tables, ignore_index=True)
            if self._prediction_tables
            else pd.DataFrame()
        )

        self._log_summary(metrics_df)
        return metrics_df, prediction_table

    # ------------------------------------------------------------------
    # Fold construction
    # ------------------------------------------------------------------
    def _build_folds(
        self, data: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Split the data into sequential train/test windows.

        Each fold slides forward by ``test_days`` calendar days.
        The training window is always the most recent ``train_days``
        calendar days *preceding* the test window.

        When ``expanding_window=True``, the training set grows from the
        start of the dataset to the test window (instead of being a fixed
        ``train_days`` window).  This is closer to a classical walk-forward
        methodology and lets the model learn from long-term patterns.
        """
        min_ts = data["timestamp"].min()
        max_ts = data["timestamp"].max()

        folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        train_start = min_ts
        global_start = min_ts  # fixed origin for expanding window

        while True:
            train_end = train_start + pd.Timedelta(days=self.train_days)
            test_start = train_end
            test_end = test_start + pd.Timedelta(days=self.test_days)

            if test_start > max_ts:
                break

            if self.expanding_window:
                train_df = data[
                    (data["timestamp"] >= global_start)
                    & (data["timestamp"] < test_start)
                ].copy()
            else:
                train_df = data[
                    (data["timestamp"] >= train_start)
                    & (data["timestamp"] < train_end)
                ].copy()
            test_df = data[
                (data["timestamp"] >= test_start) & (data["timestamp"] < test_end)
            ].copy()

            if train_df.empty or test_df.empty:
                # Slide and continue
                train_start += pd.Timedelta(days=self.test_days)
                continue

            folds.append((train_df, test_df))
            train_start += pd.Timedelta(days=self.test_days)

        if not folds:
            raise ValueError(
                "No folds could be constructed. The dataset may be too small "
                f"for {self.train_days}d train / {self.test_days}d test windows."
            )

        logger.info("Built %d walk-forward folds.", len(folds))
        return folds

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def _log_summary(self, metrics_df: pd.DataFrame) -> None:
        """Print a rich summary table of the walk-forward results."""
        mean_row = metrics_df[metrics_df["fold"] == "MEAN"]
        if mean_row.empty:
            return

        table = Table(
            title="Walk-Forward Validation Summary",
            title_style="bold cyan",
            border_style="blue",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta", justify="right")

        for col in mean_row.columns:
            if col == "fold":
                continue
            val = mean_row[col].values[0]
            formatted = f"{val:>10.6f}"
            # Color-code Sharpe and R²
            if col in ("sharpe_ratio", "r2") and val > 0:
                formatted = f"[green]{formatted}[/green]"
            elif col in ("sharpe_ratio", "r2") and val < 0:
                formatted = f"[red]{formatted}[/red]"
            table.add_row(col, formatted)

        console.print()
        console.print(table)
        console.print()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def metrics_list(self) -> List[Dict[str, float]]:
        return self._metrics_list

    @property
    def metrics_summary(self) -> Dict[str, float]:
        if not self._metrics_list:
            return {}
        return {
            k: np.mean([m[k] for m in self._metrics_list])
            for k in self._metrics_list[0]
        }
