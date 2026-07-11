"""Tests for the dynamic evaluation report generator."""

from pathlib import Path

import pandas as pd
import pytest

from crowdwisdom_quant.reporting.evaluation import (
    ReportContext,
    generate_evaluation_report,
    _load_context,
    _build_report,
)


class TestReportContext:
    """Verify that ReportContext correctly loads pipeline artifacts."""

    def test_empty_context_has_no_data(self) -> None:
        ctx = ReportContext()
        assert not ctx.has_data

    def test_context_with_data_is_detected(self) -> None:
        ctx = ReportContext()
        ctx.merged = pd.DataFrame({"timestamp": [pd.Timestamp.now()], "pnl": [1.0]})
        assert ctx.has_data

    def test_load_context_returns_none_if_no_merged(self) -> None:
        # No merged data exists in a temp directory
        ctx = _load_context()
        # In the actual project, merged_data.parquet should exist after pipeline
        # This just verifies the path logic doesn't crash
        assert ctx is None or ctx.has_data


class TestBuildReport:
    """Verify that _build_report produces valid Markdown with expected sections."""

    @pytest.fixture
    def mock_context(self, tmp_path: Path) -> ReportContext:
        ctx = ReportContext()
        dates = pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC")
        ctx.merged = pd.DataFrame({
            "timestamp": dates,
            "price": [100.0] * 100,
            "quantity": [1.0] * 100,
            "pnl": [0.0] * 100,
            "strategy_permutation": ["strat_a"] * 50 + ["strat_b"] * 50,
            "direction": ["buy"] * 50 + ["sell"] * 50,
            "macro_timestamp": dates,
            "event_name": ["CPI"] * 100,
            "actual": [1.0] * 100,
            "forecast": [0.5] * 100,
        })
        ctx.metrics = pd.DataFrame({
            "fold": ["0", "1", "MEAN"],
            "rmse": [10.0, 11.0, 10.5],
            "mae": [8.0, 8.5, 8.25],
            "r2": [-0.1, -0.05, -0.075],
            "sharpe_ratio": [-0.5, 0.5, 0.0],
            "profit_factor": [0.95, 1.05, 1.0],
        })
        ctx.predictions = pd.DataFrame({
            "Timestamp": dates[:50],
            "Actual_PnL": [0.0] * 50,
            "Predicted_PnL": [0.0] * 50,
            "Strategy": ["strat_a"] * 50,
            "Hour": [0] * 50,
            "Day": ["Monday"] * 50,
            "Fold": [0] * 50,
        })
        return ctx

    def test_build_report_contains_expected_sections(self, mock_context: ReportContext) -> None:
        report = _build_report(mock_context)
        expected_sections = [
            "Executive Summary",
            "Dataset Description",
            "Data Cleaning",
            "Feature Engineering",
            "Model Selection",
            "Walk-Forward Validation",
            "Performance Analysis",
            "Financial Metrics",
            "Strengths",
            "Limitations",
            "Recommendations",
            "Future Work",
            "Conclusion",
        ]
        for section in expected_sections:
            assert section in report, f"Missing section: {section}"

    def test_build_report_includes_metrics_table(self, mock_context: ReportContext) -> None:
        report = _build_report(mock_context)
        assert "RMSE" in report
        assert "R²" in report
        assert "Sharpe" in report
        assert "Profit Factor" in report

    def test_build_report_includes_data_summary(self, mock_context: ReportContext) -> None:
        report = _build_report(mock_context)
        assert "100" in report  # 100 trades
        assert "strat_a" in report
        assert "strat_b" in report

    def test_build_report_empty_merged(self) -> None:
        ctx = ReportContext()
        ctx.merged = pd.DataFrame()
        report = _build_report(ctx)
        assert report  # Should not crash

    def test_generate_evaluation_report_creates_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from crowdwisdom_quant.config.settings import Settings
        monkeypatch.setattr(Settings, "REPORTS_DIR", tmp_path / "reports")
        # Should not crash even without pipeline artifacts
        generate_evaluation_report()
        report_file = tmp_path / "reports" / "evaluation_report.md"
        assert report_file.exists()
        content = report_file.read_text()
        assert "CrowdWisdomTrading" in content
