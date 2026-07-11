"""CLI utilities: progress bars, timing, colored output, execution summaries.

Thin wrappers around ``rich`` that standardise the look and feel
of the CrowdWisdomTrading command line.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

F = TypeVar("F", bound=Callable[..., Any])

# Single shared console instance (writes to stderr so stdout can be piped)
console = Console(stderr=True, highlight=False)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class Stopwatch:
    """Context manager that records elapsed wall-clock time.

    Usage::

        with Stopwatch() as sw:
            do_something()
        print(f"Took {sw.elapsed:.2f}s")
    """

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._start is not None:
            self.elapsed = time.perf_counter() - self._start

    def __str__(self) -> str:
        if self.elapsed < 60:
            return f"{self.elapsed:.1f}s"
        return f"{self.elapsed / 60:.1f}m"


def timed(func: F) -> F:
    """Decorator that logs a step's duration via ``rich``."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        name = func.__name__.replace("cmd_", "").replace("_", " ").title()
        with Stopwatch() as sw:
            result = func(*args, **kwargs)
        console.log(f"[green]✓[/green] {name} completed in [cyan]{sw}[/cyan]")
        return result
    return wrapper  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Progress bars
# ---------------------------------------------------------------------------

@contextmanager
def progress_spinner(description: str = "Working ...") -> Iterator[Progress]:
    """Show an indeterminate spinner while a task runs.

    Usage::

        with progress_spinner("Scraping macro events") as p:
            task = p.add_task("...")
            do_work()
            p.update(task, completed=True)
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as p:
        yield p


@contextmanager
def progress_bar(
    total: int,
    description: str = "Processing ...",
    transient: bool = True,
) -> Iterator[Progress]:
    """Show a determinate progress bar.

    Usage::

        with progress_bar(total=10, description="Folds") as p:
            task = p.add_task("...", total=10)
            for i in range(10):
                do_work()
                p.advance(task)
    """
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
    ) as p:
        yield p


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print an execution-summary table.

    Parameters
    ----------
    results : list[dict]
        Each dict must have keys ``step`` (str), ``status`` (✓/✗),
        ``duration`` (str), and optionally ``detail`` (str).
    """
    table = Table(title="Pipeline Execution Summary", title_style="bold")
    table.add_column("Step", style="cyan")
    table.add_column("Status", no_wrap=True)
    table.add_column("Duration", style="magenta")
    table.add_column("Detail", style="white")

    total_duration = 0.0
    for r in results:
        status = Text(r.get("status", "?"), style="green" if r.get("status") == "✓" else "red")
        dur = r.get("duration", "—")
        detail = r.get("detail", "")
        table.add_row(r.get("step", ""), status, dur, detail)
        # Parse duration for total
        if isinstance(dur, str) and dur.endswith("s"):
            try:
                total_duration += float(dur.rstrip("s"))
            except ValueError:
                pass

    console.print()
    console.print(Panel(table, border_style="blue"))
    console.print(f"[bold]Total:[/bold] [cyan]{total_duration:.1f}s[/cyan]")
    console.print()


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

def run_step(name: str, func: Callable[[], Any]) -> Dict[str, Any]:
    """Execute a single pipeline step with timing and error capture.

    Returns a result dict consumable by ``print_summary``.
    """
    step_label = name.replace("_", " ").title()
    console.rule(f"[bold]{step_label}[/bold]")
    with Stopwatch() as sw:
        try:
            func()
            return {
                "step": step_label,
                "status": "✓",
                "duration": str(sw),
                "detail": "",
            }
        except Exception as e:
            console.print(f"[red]✗ {step_label} failed: {e}[/red]")
            return {
                "step": step_label,
                "status": "✗",
                "duration": str(sw),
                "detail": str(e),
            }
