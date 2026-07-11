"""Unit tests for the metrics module."""

import numpy as np
import pandas as pd
import pytest

from crowdwisdom_quant.models.metrics import (
    compute_metrics,
    metrics_dataframe,
    _sharpe_ratio,
    _sortino_ratio,
    _max_drawdown,
    _profit_factor,
)


class TestMetrics:
    def test_perfect_prediction(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = compute_metrics(y_true, y_pred)
        assert m["rmse"] == 0.0
        assert m["mae"] == 0.0
        assert m["r2"] == 1.0

    def test_constant_prediction(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        m = compute_metrics(y_true, y_pred)
        assert m["rmse"] > 0
        assert m["mae"] > 0

    def test_sharpe_ratio(self) -> None:
        pnl = np.array([1.0, -0.5, 0.5, 0.0, 1.0])
        sr = _sharpe_ratio(pnl, annual_factor=1)
        assert isinstance(sr, float)

    def test_sharpe_with_timestamps(self) -> None:
        pnl = np.array([1.0, -0.5, 0.5, 0.0, 1.0])
        timestamps = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
        sr = _sharpe_ratio(pnl, timestamps=timestamps.values)
        assert isinstance(sr, float)

    def test_sortino_ratio(self) -> None:
        pnl = np.array([1.0, -0.5, 0.5, 0.0, 1.0])
        sr = _sortino_ratio(pnl, annual_factor=1)
        assert isinstance(sr, float)

    def test_max_drawdown(self) -> None:
        pnl = np.array([1.0, 2.0, -3.0, 4.0])
        dd = _max_drawdown(pnl)
        assert dd < 0

    def test_max_drawdown_all_positive(self) -> None:
        pnl = np.array([1.0, 2.0, 3.0])
        dd = _max_drawdown(pnl)
        assert dd == 0.0

    def test_max_drawdown_all_negative(self) -> None:
        pnl = np.array([-1.0, -2.0, -3.0])
        dd = _max_drawdown(pnl)
        # All-negative PnL: peak is 0, drawdown = (cum - 0)/1 = cum
        assert dd <= 0.0

    def test_profit_factor(self) -> None:
        pnl = np.array([10.0, -5.0, 3.0, -2.0])
        pf = _profit_factor(pnl)
        assert pf > 1.0

    def test_profit_factor_all_losses(self) -> None:
        pnl = np.array([-5.0, -3.0])
        pf = _profit_factor(pnl)
        assert pf == 0.0

    def test_profit_factor_no_losses(self) -> None:
        pnl = np.array([5.0, 3.0])
        pf = _profit_factor(pnl)
        assert pf == 999.0

    def test_empty_input(self) -> None:
        y_true = np.array([])
        y_pred = np.array([])
        m = compute_metrics(y_true, y_pred)
        assert all(np.isnan(v) for v in m.values())

    def test_single_element(self) -> None:
        y_true = np.array([5.0])
        y_pred = np.array([3.0])
        m = compute_metrics(y_true, y_pred)
        assert m["rmse"] == 2.0
        assert m["mae"] == 2.0

    def test_metrics_dataframe(self) -> None:
        metrics_list = [
            {"rmse": 10.0, "mae": 8.0, "r2": -0.1, "sharpe_ratio": 0.5, "fold": 0},
            {"rmse": 11.0, "mae": 9.0, "r2": -0.2, "sharpe_ratio": -0.3, "fold": 1},
        ]
        df = metrics_dataframe(metrics_list)
        assert "MEAN" in df["fold"].values
        assert df.shape[0] == 3  # 2 folds + 1 mean
        assert df.loc[df["fold"] == "MEAN", "rmse"].values[0] == 10.5

    def test_metrics_with_timestamps(self) -> None:
        y_true = np.array([1.0, -1.0, 0.5])
        y_pred = np.array([0.8, -0.8, 0.3])
        timestamps = np.array([
            "2025-01-01", "2025-01-02", "2025-01-03"
        ], dtype="datetime64")
        m = compute_metrics(y_true, y_pred, timestamps=timestamps)
        assert "sharpe_ratio" in m
        assert isinstance(m["sharpe_ratio"], float)

    def test_zero_std_pnl(self) -> None:
        pnl = np.array([5.0, 5.0, 5.0])
        sr = _sharpe_ratio(pnl)
        assert sr == 0.0
