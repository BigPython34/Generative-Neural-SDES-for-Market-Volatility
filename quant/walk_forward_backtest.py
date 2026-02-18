"""
Walk-Forward Backtest (Advanced Phase)
======================================
Strict temporal rolling: train on [t-w, t], test on [t, t+h].
No look-ahead: H, eta, xi0 recalibrated at each fold (future: per-fold calibration).

Usage:
    from quant.walk_forward_backtest import WalkForwardBacktester
    wf = WalkForwardBacktester(train_window_days=60, test_window_days=5)
    results = wf.run(max_folds=5)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from quant.options_cache import OptionsDataCache
from quant.backtesting import HistoricalBacktester
from utils.config import load_config


class WalkForwardBacktester:
    """
    Walk-forward backtest with strict temporal ordering.

    - train_window_days: Days of data for calibration (e.g. 60)
    - test_window_days: Days to hold out for evaluation (e.g. 5)
    - step_days: Rolling step (default = test_window_days)
    """

    def __init__(self, train_window_days: int = 60, test_window_days: int = 5,
                 step_days: int = None, use_snapshots_fallback: bool = True):
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days or test_window_days
        self.use_snapshots_fallback = use_snapshots_fallback
        self.cfg = load_config()
        self.cache = OptionsDataCache()
        self.results = []

    def run(self, max_folds: int = None, reindex: bool = True) -> list:
        """
        Run walk-forward backtest.
        For each fold: run HistoricalBacktester on test_window only.
        TODO: Recalibrate H, eta, xi0 from train_window before each fold.
        """
        if reindex:
            self.cache.reindex_from_disk()
        snapshots_df = self.cache.list_snapshots("SPY")
        if snapshots_df.empty:
            print("[WARN] No cached options. Run fetch_options.py first.")
            print(f"       Cache path: {os.path.abspath(self.cache.cache_dir)}")
            return []

        print(f"Cache: {self.cache.cache_dir} | {len(snapshots_df)} snapshots")
        snapshots_df['date'] = pd.to_datetime(snapshots_df['datetime']).dt.date
        dates = sorted(snapshots_df['date'].unique())
        n_dates = len(dates)
        required = self.train_window_days + self.test_window_days

        # Fallback: use snapshots as "periods" when not enough calendar days
        if n_dates < required and self.use_snapshots_fallback and len(snapshots_df) >= 5:
            train_n = max(2, len(snapshots_df) - 3)
            test_n = min(3, len(snapshots_df) - train_n)
            if train_n >= 2 and test_n >= 1:
                print(f"[INFO] Only {n_dates} calendar days; using snapshot-based fold "
                      f"(train={train_n} snapshots, test={test_n})")
                return self._run_snapshot_mode(snapshots_df, train_n, test_n, max_folds)

        if n_dates < required:
            print(f"[WARN] {n_dates} dates, {required} needed. Cache: {self.cache.cache_dir}")
            print(f"       Use smaller windows: WalkForwardBacktester(train_window_days=10, test_window_days=3)")
            return []

        results = []
        start_idx = self.train_window_days
        fold = 0

        while start_idx + self.test_window_days <= len(dates):
            if max_folds and fold >= max_folds:
                break

            test_start = dates[start_idx]
            test_end = dates[min(start_idx + self.test_window_days - 1, len(dates) - 1)]
            train_end = dates[start_idx - 1]

            print(f"\n--- Fold {fold}: test [{test_start}, {test_end}] ---")

            try:
                bt = HistoricalBacktester()
                df = bt.run_real_backtest(date_range=(test_start, test_end))

                if df is not None and not df.empty:
                    row = {
                        "fold": fold,
                        "test_start": str(test_start),
                        "test_end": str(test_end),
                        "n_scenarios": len(df),
                        "bs_rmse_avg": float(df["bs_rmse"].mean()),
                        "bergomi_rmse_avg": float(df["bergomi_rmse"].mean()),
                        "neural_rmse_avg": float(df["neural_sde_rmse"].mean())
                        if df["neural_sde_rmse"].notna().any() else None,
                    }
                    results.append(row)
                    print(f"   BS={row['bs_rmse_avg']:.2f}% Bergomi={row['bergomi_rmse_avg']:.2f}%")
            except Exception as e:
                print(f"   [FAIL] {e}")

            start_idx += self.step_days
            fold += 1

        self.results = results

        out_path = Path("outputs/walk_forward_results.json")
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nWalk-forward: {len(results)} folds -> {out_path}")

        return results

    def _run_snapshot_mode(self, snapshots_df: pd.DataFrame, train_n: int, test_n: int,
                           max_folds: int = None) -> list:
        """Run walk-forward using snapshot indices instead of calendar dates."""
        snapshots_df = snapshots_df.sort_values('datetime').reset_index(drop=True)
        n = len(snapshots_df)
        results = []
        start = 0
        fold = 0

        while start + train_n + test_n <= n:
            if max_folds and fold >= max_folds:
                break
            test_slice = snapshots_df.iloc[start + train_n:start + train_n + test_n]
            test_dates = (test_slice['date'].min(), test_slice['date'].max())
            print(f"\n--- Fold {fold}: test snapshots [{test_dates[0]}, {test_dates[1]}] ---")

            try:
                bt = HistoricalBacktester()
                df = bt.run_real_backtest(date_range=test_dates)

                if df is not None and not df.empty:
                    row = {
                        "fold": fold,
                        "test_start": str(test_dates[0]),
                        "test_end": str(test_dates[1]),
                        "n_scenarios": len(df),
                        "bs_rmse_avg": float(df["bs_rmse"].mean()),
                        "bergomi_rmse_avg": float(df["bergomi_rmse"].mean()),
                        "neural_rmse_avg": float(df["neural_sde_rmse"].mean())
                        if df["neural_sde_rmse"].notna().any() else None,
                    }
                    results.append(row)
                    print(f"   BS={row['bs_rmse_avg']:.2f}% Bergomi={row['bergomi_rmse_avg']:.2f}%")
            except Exception as e:
                print(f"   [FAIL] {e}")

            start += test_n
            fold += 1

        self.results = results
        out_path = Path("outputs/walk_forward_results.json")
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nWalk-forward: {len(results)} folds -> {out_path}")
        return results
