"""Tests for visualization modules (heatmap, equity curve)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest

from crowdwisdom_quant.visualization.heatmap import generate_heatmap
from crowdwisdom_quant.visualization.equity_curve import generate_equity_curve


class TestHeatmap:
    """Verify heatmap generation with various inputs."""

    def test_heatmap_with_valid_data(self, tmp_path: Path) -> None:
        dates = pd.date_range("2025-01-01", periods=50, freq="h", tz="UTC")
        df = pd.DataFrame({
            "Hour": dates.hour,
            "Day": dates.day_name(),
            "Strategy": ["strat_a"] * 25 + ["strat_b"] * 25,
            "Predicted_PnL": [float(i % 2) for i in range(50)],
        })
        fig = generate_heatmap(df, output_path=tmp_path / "heatmap.png")
        assert fig is not None
        assert (tmp_path / "heatmap.png").exists()

    def test_heatmap_empty_data(self, tmp_path: Path) -> None:
        df = pd.DataFrame()
        fig = generate_heatmap(df, output_path=tmp_path / "empty_heatmap.png")
        assert fig is not None

    def test_heatmap_single_strategy(self, tmp_path: Path) -> None:
        dates = pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC")
        df = pd.DataFrame({
            "Hour": dates.hour,
            "Day": dates.day_name(),
            "Strategy": ["strat_a"] * 10,
            "Predicted_PnL": [1.0] * 10,
        })
        fig = generate_heatmap(df, output_path=tmp_path / "single.png")
        assert fig is not None


class TestEquityCurve:
    """Verify equity curve generation with various inputs."""

    def test_equity_curve_with_valid_data(self, tmp_path: Path) -> None:
        dates = pd.date_range("2025-01-01", periods=50, freq="h", tz="UTC")
        df = pd.DataFrame({
            "Timestamp": dates,
            "Actual_PnL": [1.0] * 25 + [-0.5] * 25,
            "Predicted_PnL": [0.5] * 50,
        })
        fig = generate_equity_curve(df, output_path=tmp_path / "equity.png")
        assert fig is not None
        assert (tmp_path / "equity.png").exists()

    def test_equity_curve_empty_data(self, tmp_path: Path) -> None:
        df = pd.DataFrame()
        fig = generate_equity_curve(df, output_path=tmp_path / "empty.png")
        assert fig is not None

    def test_equity_curve_all_positive(self, tmp_path: Path) -> None:
        dates = pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC")
        df = pd.DataFrame({
            "Timestamp": dates,
            "Actual_PnL": [1.0] * 10,
            "Predicted_PnL": [0.5] * 10,
        })
        fig = generate_equity_curve(df, output_path=tmp_path / "all_pos.png")
        assert fig is not None
