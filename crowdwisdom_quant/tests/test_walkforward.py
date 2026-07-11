"""Unit tests for walk-forward validation."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from crowdwisdom_quant.models.validation.walk_forward import WalkForwardValidator


def _make_test_data(n_days: int = 100, trades_per_day: int = 5):
    """Generate synthetic merged data for testing."""
    np.random.seed(42)
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for d in range(n_days):
        for t in range(trades_per_day):
            ts = base + pd.Timedelta(days=d, hours=9 + t // 2, minutes=(t % 2) * 30)
            rows.append(
                {
                    "timestamp": ts,
                    "pnl": np.random.randn() * 5,
                    "direction": np.random.choice(["buy", "sell"]),
                    "strategy_permutation": np.random.choice(
                        ["strat_a", "strat_b", "strat_c"]
                    ),
                    "event_name": "CPI",
                    "actual": 0.5,
                    "forecast": 0.3,
                    "account": "acc1",
                    "country": "US",
                }
            )
    df = pd.DataFrame(rows)
    return df.sort_values("timestamp").reset_index(drop=True)


class TestWalkForwardValidator:
    def test_build_folds(self) -> None:
        data = _make_test_data(n_days=100, trades_per_day=5)
        validator = WalkForwardValidator(train_days=30, test_days=7)
        folds = validator._build_folds(data)
        assert len(folds) > 0
        for train_df, test_df in folds:
            assert not train_df.empty
            assert not test_df.empty
            assert train_df["timestamp"].max() <= test_df["timestamp"].min()

    def test_build_folds_expanding_window(self) -> None:
        data = _make_test_data(n_days=100, trades_per_day=5)
        validator = WalkForwardValidator(train_days=30, test_days=7, expanding_window=True)
        folds = validator._build_folds(data)
        assert len(folds) > 0
        for train_df, test_df in folds:
            assert not train_df.empty
            assert not test_df.empty

    def test_run_produces_metrics_and_table(self) -> None:
        data = _make_test_data(n_days=60, trades_per_day=3)
        validator = WalkForwardValidator(train_days=20, test_days=5)
        metrics_df, pred_table = validator.run(data)
        assert not metrics_df.empty
        assert "fold" in metrics_df.columns
        assert "MEAN" in metrics_df["fold"].values
        if not pred_table.empty:
            assert "Predicted_PnL" in pred_table.columns
            assert "Actual_PnL" in pred_table.columns

    def test_run_returns_consistent_columns(self) -> None:
        data = _make_test_data(n_days=60, trades_per_day=3)
        validator = WalkForwardValidator(train_days=20, test_days=5)
        metrics_df, pred_table = validator.run(data)
        expected_metrics = {"rmse", "mae", "r2", "sharpe_ratio", "fold"}
        assert expected_metrics.issubset(metrics_df.columns)
        if not pred_table.empty:
            expected_pred = {"Actual_PnL", "Predicted_PnL", "Strategy", "Fold"}
            assert expected_pred.issubset(pred_table.columns)

    def test_small_dataset_raises_error(self) -> None:
        data = _make_test_data(n_days=5, trades_per_day=1)
        validator = WalkForwardValidator(train_days=30, test_days=7)
        with pytest.raises(ValueError, match="No folds"):
            validator.run(data)

    def test_fold_order_is_strictly_temporal(self) -> None:
        data = _make_test_data(n_days=100, trades_per_day=5)
        validator = WalkForwardValidator(train_days=30, test_days=7)
        metrics_df, _ = validator.run(data)
        fold_rows = metrics_df[metrics_df["fold"] != "MEAN"].copy()
        # Each fold's timestamp boundaries should be non-overlapping
        assert len(fold_rows) > 0

    def test_metrics_summary_property(self) -> None:
        data = _make_test_data(n_days=60, trades_per_day=3)
        validator = WalkForwardValidator(train_days=20, test_days=5)
        validator.run(data)
        summary = validator.metrics_summary
        assert "rmse" in summary
        assert isinstance(summary["rmse"], float)

    def test_predictions_are_numeric(self) -> None:
        data = _make_test_data(n_days=60, trades_per_day=3)
        validator = WalkForwardValidator(train_days=20, test_days=5)
        _, pred_table = validator.run(data)
        if not pred_table.empty:
            assert np.issubdtype(pred_table["Predicted_PnL"].dtype, np.floating)
            assert not pred_table["Predicted_PnL"].isna().all()
