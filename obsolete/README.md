# Obsolete / Superseded Scripts

Scripts moved here are no longer part of the main pipeline. Kept for reference.

| File | Reason |
|------|--------|
| `market_constrained_trainer.py` | Experimental trainer, never integrated. Main pipeline uses `GenerativeTrainer` via `train.py`. |
| `empirical_vs_grid_calibration.py` | Exploratory comparison. Superseded by `calibrate.py` + `options_calibration.py`. |
| `enhanced_calibration.py` | VVIX + futures extraction. Alternative to `advanced_calibration`. Main pipeline uses `calibrate.py` -> `advanced_calibration`. |
| `multi_maturity_calibration.py` | Multi-maturity IV surface. Overlaps with `options_calibration.py` and `backtesting.py`. |

**Main pipeline**: `train.py` | `calibrate.py` | `backtest.py` | `fetch_options.py` | `dashboard.py`
