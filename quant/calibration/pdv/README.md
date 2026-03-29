# PDV Calibration Notes

This folder documents implementation choices made for the project PDV calibration stack.

## Why Nelder-Mead

- The calibration objective is noisy (MC noise + implied-vol inversion), so gradient estimates are unstable.
- Nelder-Mead is robust in low dimension (3 PDV params) and works well with bounded budgets.
- We use a grid/coarse search first, then Nelder-Mead refinement from the best seed.

## Defaults and Data

- Default underlying is `SPX` (not `SPY`) for options surface calibration.
- Risk-free rate comes from NY Fed SOFR data (`data/market/rates/sofr_daily_nyfed.csv`).
- No silent fallback to fixed 5% should be used in calibration launchers.

## Loss Design

- Smile loss can run in `iv` mode (relative IV error) to stay aligned with paper-style targets.
- Filters remove low-quality quotes:
  - minimum option mid,
  - maximum spread percentage,
  - clipping of extreme relative errors.
- VIX term smoothness penalty can be enabled to avoid jagged term-structure fits.

## Hard Presets

- `--hard`: long run for quality (many paths + long NM refinement).
- `--hard-lite`: intermediate long run with lower runtime.
- Progress logs print evaluation count and running loss.

## Not Implemented (yet)

- Full Bayesian / CMA-ES global optimizer pipeline.
- Exact paper-identical PDV dynamics if additional state variables are required.
- Full GPU-only calibration loop benchmarking.

