#!/usr/bin/env python3
"""Command-line entry point for the CrowdWisdomTrading pipeline.

Usage::

    python main.py scrape          — Download macro events via Apify
    python main.py preprocess      — Clean trades, clean events, merge
    python main.py feature         — Engineer features from merged data
    python main.py train           — Train (tuned) XGBoost model
    python main.py validate        — Run walk-forward validation
    python main.py visualize       — Generate heatmap + equity curve
    python main.py report          — Generate evaluation report (Markdown)
    python main.py run_all         — Execute the entire pipeline sequentially
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.text import Text

# Ensure the project root is on sys.path so internal imports resolve
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path and __name__ == "__main__":
    sys.path.insert(0, str(_HERE.parent))

from crowdwisdom_quant.config.settings import Config
from crowdwisdom_quant.data.scraping.macro_scraper import MacroEconomicScraper
from crowdwisdom_quant.data.preprocessing.trades import TradingLogCleaner
from crowdwisdom_quant.data.preprocessing.events import MacroEventCleaner
from crowdwisdom_quant.data.preprocessing.merger import TradeEventMerger
from crowdwisdom_quant.features.engineering import FeatureEngineer
from crowdwisdom_quant.models.trainer import ModelTrainer
from crowdwisdom_quant.models.validation.walk_forward import WalkForwardValidator
from crowdwisdom_quant.reporting.evaluation import generate_evaluation_report
from crowdwisdom_quant.utils.cli_utils import (
    console,
    progress_bar,
    print_summary,
    run_step,
    timed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Pipeline commands
# ---------------------------------------------------------------------------


@timed
def cmd_scrape() -> None:
    """Download macro-economic events from Apify (or CSV fallback)."""
    Config.ensure_dirs()
    scraper = MacroEconomicScraper()
    n = scraper.run()
    console.log(f"[green]✓[/green] Scraping complete. [bold]{n}[/bold] events inserted.")


@timed
def cmd_preprocess() -> None:
    """Clean trades, clean events, and merge."""
    Config.ensure_dirs()

    console.log("[cyan]▸[/cyan] Cleaning trading logs ...")
    cleaner = TradingLogCleaner()
    trades = cleaner.run()
    n_trades = len(trades)

    console.log("[cyan]▸[/cyan] Cleaning macro events ...")
    event_cleaner = MacroEventCleaner()
    events = event_cleaner.run()
    n_events = len(events)

    console.log("[cyan]▸[/cyan] Merging trades with events ...")
    merger = TradeEventMerger()
    merged = merger.run()

    console.log(
        f"[green]✓[/green] Preprocessing: {n_trades} trades, "
        f"{n_events} events → {len(merged)} merged rows."
    )


@timed
def cmd_feature() -> None:
    """Engineer features from merged data."""
    Config.ensure_dirs()
    fe = FeatureEngineer()
    X, y = fe.run()
    console.log(
        f"[green]✓[/green] Feature engineering: {X.shape[0]} samples, "
        f"[bold]{X.shape[1]}[/bold] features."
    )


@timed
def cmd_train() -> None:
    """Train (and tune) the XGBoost model."""
    Config.ensure_dirs()
    processed = Config.PROCESSED_DATA_DIR
    features_path = processed / "features.parquet"
    target_path = processed / "target.parquet"

    if not features_path.exists():
        console.log("[red]✗[/red] Features not found. Run [bold]feature[/bold] first.")
        return

    X = pd.read_parquet(features_path)
    y = pd.read_parquet(target_path)["pnl"]

    trainer = ModelTrainer()
    trainer.train(X, y, tune=True)

    # Print feature importance as a rich table
    fi = trainer.feature_importances
    if fi is not None:
        table = Table(title="Feature Importance (Gain)", title_style="bold")
        table.add_column("Feature", style="cyan")
        table.add_column("Importance", style="magenta", justify="right")
        for _, row in fi.head(10).iterrows():
            table.add_row(str(row["feature"]), f"{row['importance']:.4f}")
        console.print(table)


@timed
def cmd_validate() -> None:
    """Run walk-forward validation."""
    Config.ensure_dirs()
    merged_path = Config.PROCESSED_DATA_DIR / "merged_data.parquet"

    if not merged_path.exists():
        console.log("[red]✗[/red] Merged data not found. Run [bold]preprocess[/bold] first.")
        return

    data = pd.read_parquet(merged_path)
    console.log(f"Loaded merged data: {len(data):,} rows.")

    validator = WalkForwardValidator()
    metrics_df, pred_table = validator.run(data)

    # Save results
    metrics_df.to_parquet(
        Config.PROCESSED_DATA_DIR / "walkforward_metrics.parquet", index=False
    )
    pred_table.to_parquet(
        Config.PROCESSED_DATA_DIR / "prediction_table.parquet", index=False
    )
    console.log("[green]✓[/green] Walk-forward results saved.")


@timed
def cmd_visualize() -> None:
    """Generate heatmap and equity curve."""
    Config.ensure_dirs()
    pred_path = Config.PROCESSED_DATA_DIR / "prediction_table.parquet"

    if not pred_path.exists():
        console.log("[red]✗[/red] Prediction table not found. Run [bold]validate[/bold] first.")
        return

    pred_table = pd.read_parquet(pred_path)
    console.log(f"Loaded prediction table: {len(pred_table):,} rows.")

    from crowdwisdom_quant.visualization.heatmap import generate_heatmap
    from crowdwisdom_quant.visualization.equity_curve import generate_equity_curve

    generate_heatmap(pred_table)
    generate_equity_curve(pred_table)
    console.log(f"[green]✓[/green] Visualizations saved to {Config.VISUALIZATION_DIR}.")


@timed
def cmd_report() -> None:
    """Generate evaluation report (Markdown)."""
    Config.ensure_dirs()
    generate_evaluation_report()
    report_path = Config.REPORTS_DIR / "evaluation_report.md"
    console.log(f"[green]✓[/green] Report written to {report_path}.")


@timed
def cmd_run_all() -> None:
    """Execute the entire pipeline in order."""
    steps: List[Dict[str, Any]] = [
        {"name": "scrape", "func": cmd_scrape, "detail": ""},
        {"name": "preprocess", "func": cmd_preprocess, "detail": ""},
        {"name": "feature", "func": cmd_feature, "detail": ""},
        {"name": "validate", "func": cmd_validate, "detail": ""},
        {"name": "visualize", "func": cmd_visualize, "detail": ""},
        {"name": "report", "func": cmd_report, "detail": ""},
    ]

    console.rule("[bold cyan]CrowdWisdomTrading — Full Pipeline[/bold cyan]")

    results: List[Dict[str, Any]] = []
    with progress_bar(total=len(steps), description="Pipeline progress") as p:
        task = p.add_task("Running ...", total=len(steps))
        for step in steps:
            result = run_step(step["name"], step["func"])
            result["detail"] = step["detail"]
            results.append(result)
            p.advance(task)

    print_summary(results)

    # Check for failures
    failures = [r for r in results if r["status"] == "✗"]
    if failures:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommand help."""
    parser = argparse.ArgumentParser(
        prog="crowdwisdom_quant",
        description="Quantitative trading research platform — macro-driven ML pipeline.",
        epilog=(
            "Example: python main.py run_all\n"
            "For details see docs/architecture.md"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=[
            "scrape",
            "preprocess",
            "feature",
            "train",
            "validate",
            "visualize",
            "report",
            "run_all",
        ],
        metavar="{" + ",".join([
            "scrape", "preprocess", "feature", "train",
            "validate", "visualize", "report", "run_all",
        ]) + "}",
        help="| ".join([
            "Download macro events",
            "Clean & merge data",
            "Engineer features",
            "Train XGBoost model",
            "Walk-forward validation",
            "Generate plots",
            "Write evaluation report",
            "Run entire pipeline",
        ]),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main() -> None:
    """CLI entry point — dispatches to the requested command."""
    args = parse_args()
    command_map = {
        "scrape": cmd_scrape,
        "preprocess": cmd_preprocess,
        "feature": cmd_feature,
        "train": cmd_train,
        "validate": cmd_validate,
        "visualize": cmd_visualize,
        "report": cmd_report,
        "run_all": cmd_run_all,
    }
    command_map[args.command]()


if __name__ == "__main__":
    main()
