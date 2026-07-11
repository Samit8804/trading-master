"""Tests for CLI utilities (Stopwatch, timed, progress bars, summary)."""

import time
from typing import Any, Dict, List

import pytest

from crowdwisdom_quant.utils.cli_utils import (
    Stopwatch,
    print_summary,
    run_step,
    timed,
)


class TestStopwatch:
    """Verify the Stopwatch context manager."""

    def test_stopwatch_measures_time(self) -> None:
        with Stopwatch() as sw:
            time.sleep(0.05)
        assert sw.elapsed >= 0.05

    def test_stopwatch_zero_before_context(self) -> None:
        sw = Stopwatch()
        assert sw.elapsed == 0.0

    def test_stopwatch_string_representation(self) -> None:
        with Stopwatch() as sw:
            time.sleep(0.01)
        s = str(sw)
        assert "s" in s or "m" in s


class TestTimedDecorator:
    """Verify the @timed decorator wraps and executes."""

    def test_timed_decorator_runs_function(self) -> None:
        results: List[str] = []

        @timed
        def sample() -> None:
            results.append("done")

        sample()
        assert results == ["done"]

    def test_timed_decorator_passes_args(self) -> None:
        @timed
        def add(a: int, b: int) -> int:
            return a + b

        result = add(3, 4)
        assert result == 7

    def test_timed_decorator_propagates_exception(self) -> None:
        @timed
        def crash() -> None:
            raise ValueError("oops")

        with pytest.raises(ValueError, match="oops"):
            crash()


class TestRunStep:
    """Verify the run_step function captures success and failure."""

    def test_run_step_success(self) -> None:
        def good() -> None:
            pass

        result = run_step("test_step", good)
        assert result["status"] == "✓"
        assert result["step"] == "Test Step"

    def test_run_step_failure(self) -> None:
        def bad() -> None:
            raise RuntimeError("step failed")

        result = run_step("bad_step", bad)
        assert result["status"] == "✗"
        assert "step failed" in result["detail"]

    def test_run_step_timing(self) -> None:
        def slow() -> None:
            time.sleep(0.05)

        result = run_step("slow", slow)
        dur = result["duration"].rstrip("sm")
        assert float(dur) >= 0


class TestPrintSummary:
    """Verify print_summary does not crash with various inputs."""

    def test_summary_empty(self) -> None:
        print_summary([])

    def test_summary_single_result(self) -> None:
        print_summary([
            {"step": "Test", "status": "✓", "duration": "0.5s", "detail": ""}
        ])

    def test_summary_multiple_results(self) -> None:
        print_summary([
            {"step": "Step 1", "status": "✓", "duration": "1.0s", "detail": ""},
            {"step": "Step 2", "status": "✗", "duration": "0.3s", "detail": "Error"},
            {"step": "Step 3", "status": "✓", "duration": "2.5s", "detail": "OK"},
        ])

    def test_summary_missing_keys(self) -> None:
        print_summary([
            {"step": "Partial"}  # Missing status, duration, detail
        ])
