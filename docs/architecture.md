# CrowdWisdomQuant — Architecture Overview

## High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                       CLI (main.py)                         │
│              commands: scrape, preprocess, feature,         │
│              train, validate, visualize, report, run_all    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                      Pipeline Controller                     │
│              (cli/entry.py — orchestrates steps)             │
└──┬──────────┬──────────┬──────────┬──────────┬──────────────┘
   │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼
┌──────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────────┐
│Data  │ │Data    │ │Feature │ │Model   │ │ Visualization│
│Acq.  │ │Prep.   │ │Eng.    │ │Train   │ │ & Reporting  │
├──────┤ ├────────┤ ├────────┤ ├────────┤ ├──────────────┤
│• API │ │• Clean │ │• Time  │ │• XGB   │ │• Heatmap     │
│• CSV │ │  Trades│ │  feats │ │• Grid  │ │• EquityCurve │
│• DB  │ │• Clean │ │• Macro │ │  Search│ │• Report (MD) │
│      │ │  Events│ │  feats │ │• Walk- │ │              │
│      │ │• Merge │ │• Rolling│ │  Fwd   │ │              │
│      │ │        │ │• Agg.  │ │• Model │ │              │
│      │ │        │ │        │ │  Reg.  │ │              │
└──────┘ └────────┘ └────────┘ └────────┘ └──────────────┘
```

## Package Structure

```
crowdwisdom_quant/                    # Python package root
├── config/                           # Configuration management
│   ├── __init__.py                   # Re-exports Config from settings
│   ├── settings.py                   # Settings class (YAML + env vars)
│   └── default.yaml                  # Default YAML configuration
├── cli/                              # Command-line interface
│   ├── __init__.py
│   └── entry.py                      # CLI entry point & commands
├── data/                             # Data layer
│   ├── database/                     # ORM & persistence
│   │   ├── schema.py                 # SQLAlchemy ORM models
│   │   └── manager.py                # Database session management
│   ├── scraping/                     # Data acquisition
│   │   └── macro_scraper.py          # Apify-based macro event scraper
│   └── preprocessing/                # Data cleaning & merging
│       ├── trades.py                 # Trading log cleaner
│       ├── events.py                 # Macro event cleaner
│       └── merger.py                 # Trade-event merge (asof)
├── features/                         # Feature engineering
│   ├── __init__.py
│   └── engineering.py                # All feature builders
├── models/                           # ML model layer
│   ├── registry.py                   # Model version registry
│   ├── trainer.py                    # XGBoost training + grid search
│   ├── predictor.py                  # Prediction wrapper
│   ├── metrics.py                    # Regression & trading metrics
│   └── validation/                   # Walk-forward validation
│       └── walk_forward.py           # Time-series CV engine
├── visualization/                    # Plot generation
│   ├── heatmap.py                    # Day×Hour strategy heatmap
│   └── equity_curve.py               # AI vs Baseline equity curve
├── reporting/                        # Report generation
│   └── evaluation.py                 # Markdown evaluation report
├── utils/                            # Cross-cutting utilities
│   ├── logging_setup.py              # Structured logging (rotation)
│   ├── reproducibility.py            # Seed management, env capture
│   └── file_io.py                    # File read/write helpers
├── database/                         # [DEPRECATED] backward compat
├── scraper/                          # [DEPRECATED] backward compat
├── preprocessing/                    # [DEPRECATED] backward compat
├── features/                         # [DEPRECATED] backward compat
├── models/                           # [DEPRECATED] backward compat
└── main.py                           # [DEPRECATED] thin CLI wrapper
```

## Data Flow

1. **Scrape** → Raw macro events → SQLite DB (data/database/)
2. **Preprocess** → Clean trades (CSV) + clean events (DB) → Merge via `merge_asof`
3. **Feature Engineering** → Time features, macro features, rolling stats → Parquet
4. **Walk-Forward Validation** → For each (30d train / 7d test) fold:
   - Fit scaler on train fold → Transform both folds
   - Grid search on fold 0 only → Train final model per fold
   - Predict on test fold → Record metrics
5. **Visualize** → Heatmap (best strategy by day×hour) + Equity curves
6. **Report** → Markdown evaluation report with all metrics + plots

## Anti-Leakage Guarantees

- `merge_asof(direction="backward")` ensures trades never see future events
- Scalers fitted independently per training fold
- Rolling features use `.shift(1)` — current trade's PnL never leaks
- `hourly_trade_count` uses `cumcount().shift(1)` — no forward bias
- Grid search uses `TimeSeriesSplit` (not KFold) — no temporal overfit
- Hyperparameter tuning ONLY on fold 0 (subsequent folds re-use params)
- All test folds are strictly after their corresponding training fold

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `src/`-style layout | Industry standard for Python packages; clear separation |
| SQLite dev, PostgreSQL prod | `DATABASE_URL`-driven; one-line migration |
| YAML config + env vars | Twelve-factor app; CI-friendly |
| Walk-forward (not CV) | Only valid method for time-series financial data |
| XGBoost over neural nets | Superior on tabular data; built-in regularisation |
| Model registry | Version tracking for reproducibility |
| Structured logging | Machine-parseable logs for monitoring |
