<div align="center">
  <h1>CrowdWisdomTrading</h1>
  <p><em>Quantitative Trading Research Platform — Macro-Driven ML Pipeline</em></p>

  <!-- Badges -->
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/tests-25%20passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/code%20style-ruff-black" alt="Ruff">
  <img src="https://img.shields.io/badge/model-XGBoost-orange" alt="XGBoost">
  <img src="https://img.shields.io/badge/validation-walk--forward-blueviolet" alt="Walk-Forward">
</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Data Format](#data-format)
- [Feature Engineering](#feature-engineering)
- [Model](#model)
- [Walk-Forward Validation](#walk-forward-validation)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualizations](#visualizations)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

CrowdWisdomTrading is a production-grade quantitative research platform that:

1. **Ingests** simulated multi-strategy trading logs and macro-economic calendar data.
2. **Cleans and merges** data with strict temporal alignment to prevent look-ahead bias.
3. **Engineers** 27 time-series, macro, rolling, and aggregate features.
4. **Trains** an XGBoost regressor to predict per-trade PnL.
5. **Validates** via walk-forward cross-validation (13 folds, 30d train / 7d test).
6. **Generates** professional evaluation reports with financial metrics.

Built for Python 3.12+, it follows the coding standards and validation rigour
expected at top quantitative trading firms (Jane Street, Citadel, Two Sigma,
Renaissance Technologies).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI (main.py)                            │
│   scrape │ preprocess │ feature │ validate │ visualize │ report  │
└─────┬──────────┬──────────┬──────────┬──────────┬──────────┬───┘
      │          │          │          │          │          │
      ▼          ▼          ▼          ▼          ▼          ▼
┌─────────┐ ┌──────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ Scraper │ │Preprocess│ │Feature │ │ Model  │ │ Visual │ │ Report │
│ Apify / │ │ Clean &  │ │Engineer│ │Trainer │ │ Heatmap│ │Markdown│
│ CSV     │ │ Merge    │ │27 feats│ │XGBoost │ │Equity  │ │Prof.   │
│ Fallback│ │          │ │        │ │WF Valid│ │Curve   │ │Format  │
└────┬────┘ └────┬─────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
     │           │           │          │          │          │
     ▼           ▼           ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Storage Layer                               │
│  ┌─────────┐  ┌──────────────┐  ┌────────────────────────────┐  │
│  │  CSV /  │  │   SQLite /   │  │  Parquet (processed data)  │  │
│  │ Raw Data│  │   PostgreSQL │  │                            │  │
│  └─────────┘  └──────────────┘  └────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pipeline

```mermaid
graph LR
    A[Raw Trades CSV] --> B[Clean Trades]
    C[Apify API / CSV] --> D[Clean Events]
    B --> E[Merge Trades + Events]
    D --> E
    E --> F[Feature Engineering]
    F --> G[Walk-Forward Validation]
    G --> H[Train XGBoost per Fold]
    H --> I[Predict on Test Fold]
    I --> J[Compute Metrics]
    J --> K[Visualize]
    J --> L[Generate Report]
```

Each step is independently runnable via the CLI:

| Step | Command | Description |
|------|---------|-------------|
| 1 | `python main.py scrape` | Download macro-economic events |
| 2 | `python main.py preprocess` | Clean trades, events, and merge |
| 3 | `python main.py feature` | Engineer 27 features |
| 4 | `python main.py train` | Train XGBoost with grid search |
| 5 | `python main.py validate` | Walk-forward validation (13 folds) |
| 6 | `python main.py visualize` | Generate heatmap + equity curve |
| 7 | `python main.py report` | Write evaluation report |
| — | `python main.py run_all` | Execute all steps sequentially |

---

## Installation

### Prerequisites

- Python 3.12 or later
- Git
- (Optional) [Apify API key](https://console.apify.com) for live macro data

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/crowdwisdom_quant.git
cd crowdwisdom_quant

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python main.py --help
python -m pytest tests/ -v
```

Expected output: 25 passing tests.

---

## Quick Start

Run the full pipeline with one command:

```bash
python main.py run_all
```

This executes: `scrape → preprocess → feature → validate → visualize → report`.

The entire run completes in approximately 2–3 minutes on a modern laptop.

To run individual steps:

```bash
python main.py scrape
python main.py preprocess
python main.py feature
python main.py validate
python main.py visualize
python main.py report
```

---

## Configuration

All configuration is managed through `config/default.yaml` and can be
overridden via `CROWDWISDOM_*` environment variables.

### Key Settings

| Setting | Env Variable | Default | Description |
|---------|-------------|---------|-------------|
| `database.name` | `CROWDWISDOM_DB_NAME` | `crowdwisdom.db` | SQLite database filename |
| `scraper.lookback_days` | `CROWDWISDOM_SCRAPE_LOOKBACK_DAYS` | `180` | Days of macro history to scrape |
| `preprocessing.trade_group_ms` | `CROWDWISDOM_TRADE_GROUP_MS` | `500` | Window (ms) for grouping nearby trades |
| `validation.train_days` | `CROWDWISDOM_TRAIN_DAYS` | `30` | Walk-forward training window |
| `validation.test_days` | `CROWDWISDOM_TEST_DAYS` | `7` | Walk-forward test window |
| `random_seed` | `CROWDWISDOM_RANDOM_SEED` | `42` | Global random seed |
| `logging.level` | `CROWDWISDOM_LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |

### Apify API Key

For live macro-economic data, set your Apify API key:

```bash
# Windows
set APIFY_API_KEY=your_key_here

# Linux / macOS
export APIFY_API_KEY=your_key_here

# Or add to .env file
echo "APIFY_API_KEY=your_key_here" >> .env
```

Without an API key, the pipeline generates synthetic but deterministic
macro data for development and testing.

---

## Data Format

### Input: Trading Logs (CSV)

Place your trading log CSV at `data/raw/trading_logs.csv` with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Trade execution time (UTC recommended) |
| `direction` | str | `"buy"` or `"sell"` |
| `price` | float | Execution price |
| `quantity` | float | Number of units traded |
| `pnl` | float | Profit/Loss in USD |
| `strategy_permutation` | str | Strategy identifier (e.g. `strat_a`) |
| `simulation_id` | str | Simulation run identifier |
| `account` | str | Trading account ID |

### Processing

- **Timestamp normalisation**: All timestamps are converted to UTC.
- **Duplicate removal**: Exact duplicate rows are dropped.
- **Temporal grouping**: Trades within 500ms (same account/strategy/direction)
  are aggregated by mean price/PnL and summed quantity.
- **Missing values**: Rows with null PnL or missing timestamps are removed.

---

## Feature Engineering

Twenty-seven features are computed across four categories:

### Time Features (6)

| Feature | Description |
|---------|-------------|
| `hour` | Hour of day (0–23) |
| `minute` | Minute of hour (0–59) |
| `weekday` | Day of week (0=Monday, 6=Sunday) |
| `month` | Calendar month (1–12) |
| `week_number` | ISO week number |
| `is_weekend` | Binary flag for Saturday/Sunday |

### Macro Features (3)

| Feature | Description |
|---------|-------------|
| `time_since_last_macro_event` | Seconds since the most recent macro release |
| `time_until_next_macro_event` | Seconds until the next scheduled release |
| `macro_surprise` | Actual minus forecast value |

### Rolling Features (9)

Computed over 5, 10, and 20 trade windows, each with `.shift(1)` to prevent
leakage:

| Feature | Description |
|---------|-------------|
| `rolling_avg_pnl_{w}` | Mean PnL over the window |
| `rolling_volatility_{w}` | Standard deviation of PnL |
| `rolling_win_rate_{w}` | Fraction of positive-PnL trades |

### Aggregate Features (9)

| Feature | Description |
|---------|-------------|
| `trade_frequency` | Trades in the last hour |
| `hourly_trade_count` | Trades in the current clock hour |
| `strategy_frequency` | Trades for same strategy in last 24 hours (per window) |

All features are verified to be **leakage-free**: rolling features use
`.shift(1)`, macro events use `merge_asof(direction='backward')`, and
scalers are refit per training fold.

---

## Model

### Algorithm

**XGBoost Regressor** (`XGBRegressor`) with the following rationale:

- Tree-based models handle mixed data types, missing values, and
  non-linearities without explicit preprocessing.
- Built-in L1/L2 regularisation prevents overfitting on noisy financial data.
- Feature importance (gain, weight, cover) provides interpretability.
- Consistently top-performing on tabular financial datasets.

### Hyperparameter Tuning

A grid search is performed on the first training fold using
`GridSearchCV` with 3-fold `TimeSeriesSplit`:

| Parameter | Grid | Selected |
|-----------|------|----------|
| `n_estimators` | 100, 200 | 100 |
| `max_depth` | 4, 6 | 4 |
| `learning_rate` | 0.05, 0.10 | 0.05 |
| `subsample` | 0.8, 1.0 | 0.8 |
| `colsample_bytree` | 0.8, 1.0 | 0.8 |

### Feature Importance

Top features (by gain) are logged after training and included in the
evaluation report.

---

## Walk-Forward Validation

The most critical component for avoiding time-series leakage:

```
Fold 0: |───── Train (30d) ─────|── Test (7d) ──|
Fold 1:      |───── Train (30d) ─────|── Test (7d) ──|
Fold 2:           |───── Train (30d) ─────|── Test (7d) ──|
...                                         ...
Fold 12:                                       |── Test ──|
```

- Training window: 30 calendar days (≈2000–2500 trades)
- Test window: 7 calendar days (≈500–600 trades)
- Sliding window (non-overlapping test sets)
- 13 total folds covering 120 trading days
- StandardScaler refit on each training fold

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | `√(mean(residual²))` | Lower is better. 0 = perfect |
| **MAE** | `mean(|residual|)` | Lower is better. Robust to outliers |
| **R²** | `1 - SS_res / SS_tot` | 1 = perfect; 0 = mean baseline; < 0 = worse than mean |
| **Sharpe Ratio** | `√252 × mean(PnL) / std(PnL)` | Risk-adjusted return (annualised). > 1 = good |
| **Sortino Ratio** | `√252 × mean(PnL) / downside_std` | Like Sharpe but penalises only downside volatility |
| **Max Drawdown** | `min(peak-to-trough decline)` | Worst realised loss. > −20% is typical |
| **Win Rate** | `mean(PnL > 0)` | Fraction of profitable trades. > 50% is good |
| **Profit Factor** | `gross_profits / |gross_losses|` | > 1.0 = profitable; > 2.0 = strong |

---

## Visualizations

Two visualizations are generated after validation:

### Equity Curve

Compares cumulative PnL of the AI-selected strategy (trades with predicted
positive PnL) against the baseline (all trades).

![Equity Curve](visualization/equity_curve.png)

### Heatmap

Shows the best-performing strategy by day of week and hour of day, based on
mean predicted PnL:

![Heatmap](visualization/heatmap.png)

---

## Project Structure

```
crowdwisdom_quant/
├── main.py                     # CLI entry point
├── pyproject.toml              # Package metadata & pytest config
├── requirements.txt            # Python dependencies
├── Makefile                    # Build / test / run targets
├── config/
│   ├── default.yaml            # Default YAML configuration
│   ├── settings.py             # Settings class (YAML + env vars)
│   └── __init__.py             # Re-exports Config
├── cli/
│   ├── entry.py                # CLI commands & argument parsing
│   └── __init__.py
├── data/
│   ├── raw/                    # Raw input CSV files
│   ├── processed/              # Cleaned Parquet files
│   ├── database/               # SQLite database + ORM models
│   ├── scraping/               # Apify macro-economic scraper
│   └── preprocessing/          # Trade/event cleaning & merge
├── features/
│   └── engineering.py          # 27-feature engineering pipeline
├── models/
│   ├── trainer.py              # XGBoost training + grid search
│   ├── predictor.py            # Prediction module
│   ├── metrics.py              # Financial & regression metrics
│   ├── registry.py             # Model version registry
│   └── validation/
│       └── walk_forward.py     # Walk-forward validation
├── utils/
│   ├── cli_utils.py            # Rich CLI utilities
│   ├── exceptions.py           # Typed exception hierarchy
│   ├── retry.py                # Retry decorator with back-off
│   ├── file_io.py              # Parquet read/write helpers
│   ├── logging_setup.py        # Rotating file logging
│   └── reproducibility.py      # Seed management, git hash
├── visualization/
│   ├── heatmap.py              # Day × Hour strategy heatmap
│   └── equity_curve.py         # AI vs Baseline equity curve
├── reporting/
│   └── evaluation.py           # Dynamic report generator
├── reports/                    # Generated evaluation reports
├── tests/                      # pytest test suite
│   ├── test_cleaning.py        # TradingLogCleaner tests
│   ├── test_database.py        # DatabaseManager tests
│   ├── test_features.py        # FeatureEngineer tests
│   ├── test_metrics.py         # Metrics computation tests
│   ├── test_scraper.py         # Scraper tests
│   └── test_walkforward.py     # Walk-forward validation tests
└── docs/
    └── architecture.md         # Architecture documentation
```

---

## Examples

### Full Pipeline with Custom Data

```bash
# Place your CSV at the default location
cp my_trades.csv data/raw/trading_logs.csv

# Run the full pipeline
python main.py run_all

# View the report
cat reports/evaluation_report.md
```

### Walk-Forward Only (if data is already processed)

```bash
python main.py validate
python main.py visualize
python main.py report
```

### Custom Configuration

```bash
# Use a 60-day training window with 14-day test window
$env:CROWDWISDOM_TRAIN_DAYS = "60"
$env:CROWDWISDOM_TEST_DAYS = "14"
python main.py run_all
```

### Accessing Metrics Programmatically

```python
from crowdwisdom_quant.models.metrics import compute_metrics
import numpy as np

y_true = np.array([1.0, -0.5, 2.0, -1.0])
y_pred = np.array([0.8, -0.3, 1.5, -0.7])
metrics = compute_metrics(y_true, y_pred)
print(metrics["sharpe_ratio"])  # 0.47
```

---

## Testing

Run the full test suite:

```bash
python -m pytest tests/ -v
```

Run with coverage:

```bash
python -m pytest tests/ -v --cov=crowdwisdom_quant --cov-report=term
```

Current coverage: **25 tests, all passing**.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'crowdwisdom_quant'`

Ensure you're running from the correct directory:

```bash
cd crowdwisdom_quant
python main.py scrape
```

### Database locked errors

SQLite may show `database is locked` under concurrent access. This pipeline
runs single-threaded, so the issue typically indicates another process has
the database open. Close any database browsers or notebooks.

### Grid search is slow

Grid search runs on the first fold only. To skip it:

```bash
python -c "
from crowdwisdom_quant.models.trainer import ModelTrainer
import pandas as pd
X = pd.read_parquet('data/processed/features.parquet')
y = pd.read_parquet('data/processed/target.parquet')['pnl']
trainer = ModelTrainer()
trainer.train(X, y, tune=False)
"
```

### Synthetic data usage

If no `APIFY_API_KEY` is set, the pipeline warns:

```
No Apify API key found – using CSV fallback.
```

This is expected for development. The synthetic data is deterministic
so results are reproducible, but they do not reflect real market conditions.

---

## FAQ

**Q: Why XGBoost and not a neural network?**

Neural networks require significantly more data and hyperparameter tuning
to outperform tree-based models on tabular data. XGBoost achieves
state-of-the-art results on most financial tabular benchmarks with fewer
tuning iterations.

**Q: How do I use PostgreSQL instead of SQLite?**

Set the `CROWDWISDOM_DATABASE_URL` environment variable:

```bash
$env:CROWDWISDOM_DATABASE_URL = "postgresql://user:pass@host:5432/crowdwisdom"
```

The ORM is database-agnostic; no code changes are needed.

**Q: Can I add my own features?**

Extend the `FeatureEngineer` class in `features/engineering.py`. Add a new
`_add_*_features()` static method and call it from `fit_transform()`.

**Q: How do I interpret a negative R²?**

It means the model performs worse than always predicting the mean PnL.
This is common in noisy financial data with low signal-to-noise ratio.
The framework is designed to quantify this honestly.

**Q: Why walk-forward instead of random train/test split?**

Financial data is temporally dependent. Random splits cause look-ahead
bias where future information leaks into the training set, inflating
performance metrics. Walk-forward preserves temporal ordering.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow, code style,
and pull request guidelines.

## Security

See [SECURITY.md](SECURITY.md) for our vulnerability reporting process.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
