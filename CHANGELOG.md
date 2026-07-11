# Changelog

All notable changes to CrowdWisdomTrading are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] — 2026-07-11

### Added
- Professional CLI with `rich`-powered progress bars, coloured output,
  execution timing, and pipeline summary tables.
- Typed exception hierarchy (`crowdwisdom_quant.utils.exceptions`)
  covering data, scraper, model, and configuration errors.
- `@retry` decorator (`utils/retry.py`) with exponential back-off,
  jitter, and configurable exception types.
- Community files: `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, issue templates, and PR template.
- `utils/cli_utils.py` — reusable CLI utilities (Stopwatch, progress
  bar contexts, step runner, summary printer).
- `reporting/evaluation.py` — dynamic professional report generator
  with 12 sections, live data tables, and per-fold metrics.

### Changed
- CLI upgraded: `rich` replaces bare `print()` calls; `--help` shows
  epilog and documentation references; timing automatic on every step.
- Walk-forward summary output uses `rich` tables with colour-coded
  metrics (green for positive Sharpe/R², red for negative).
- Scraper `fetch_from_api()` now uses the `@retry` decorator instead
  of inline retry logic, and raises typed exceptions.
- `requirements.txt` — added `rich>=13.0.0`.
- Evaluation report is now dynamically generated from pipeline
  artifacts with live data and professional formatting.
- Architecture documentation expanded with component diagrams.

### Fixed
- `Settings.ensure_dirs()` changed from instance to class method
  (was failing on first call from CLI).
- `clr_console` import path corrected in `models/walk_forward.py`.
- Inner `crowdwisdom_quant` directory resolved import ambiguity.

## [1.0.0] — 2026-07-10

### Added
- Baseline quant internship assessment with XGBoost PnL predictor.
- Apify macro-economic calendar scraper with CSV fallback.
- Trading log cleaning pipeline (dedup, temporal grouping, UTC normalisation).
- Macro event cleaning and trade-event merging via `merge_asof`.
- 27-dimensional feature set (time, macro, rolling, aggregate).
- Walk-forward validation (13 folds, 30d train / 7d test).
- Financial metrics (Sharpe, Sortino, max drawdown, win rate, profit factor).
- Hyperparameter grid search with `GridSearchCV` + `TimeSeriesSplit`.
- Heatmap and equity curve visualisation.
- Unit test suite (25 tests) covering all components.
- SQLAlchemy ORM with SQLite backend.
- YAML + environment-variable configuration system.
- Rotating file logging with structured format.
- Reproducibility utilities (seeds, git hash capture).
- Model registry for experiment tracking.
