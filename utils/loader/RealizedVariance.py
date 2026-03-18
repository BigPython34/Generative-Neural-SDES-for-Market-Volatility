from __future__ import annotations

import pandas as pd
import numpy as np
import jax.numpy as jnp
import os

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


from utils.config import load_config

class RealizedVolatilityLoader:
    """
    Computes Realized Volatility from S&P 500 intraday price data.
    
    Theory: RV_t = sqrt(sum(r_i^2)) where r_i are intraday returns
    
    Key distinction:
    - Realized Vol from 5-min SPX returns → H ≈ 0.05-0.14 (rough!)
    - Gatheral, Jaisson, Rosenbaum (2018): "Volatility is Rough"
    """
    def __init__(self,  config_path: str = "config/params.yaml",
                 bar_interval_min: int = None):
        self.config = load_config(config_path)
        self.file_path = self.config['data']['rv_source']
        if not self.file_path:
            raise ValueError("Missing SPX intraday RV source. Set data.rv_source in config.")
        self.max_gap_hours = self.config['data'].get('max_gap_hours', 4)
        self.stride_ratio = self.config['data'].get('stride_ratio', 0.5)
        self.trading_hours = self.config['data'].get('trading_hours_per_day', 6.5)
        self.trading_days_per_year = float(self.config['data'].get('trading_days_per_year', 252.0))
        rv_filter = self.config['data'].get('rv_filter', {}) if isinstance(self.config.get('data', {}), dict) else {}
        self.rv_min = float(rv_filter.get('min', self.config['data'].get('rv_min', 0.001)))
        self.rv_max = float(rv_filter.get('max', self.config['data'].get('rv_max', 5.0)))
        
        # Auto-detect bar frequency from data or use explicit parameter
        if bar_interval_min is not None:
            self._bar_min = bar_interval_min
        else:
            self._bar_min = self._detect_bar_interval()
        self.bars_per_day = int(self.trading_hours * 60 / self._bar_min)
        target_bar_min = int(self.config.get('simulation', {}).get('bar_interval_min', self._bar_min))
        print(
            f"   RV Loader: base={self._bar_min}m ({self.bars_per_day} bars/day), target={target_bar_min}m"
        )
    
    def _detect_bar_interval(self) -> int:
        """Auto-detect bar interval from data timestamps."""
        
        df = pd.read_csv(self.file_path, nrows=200)
        dt = pd.to_datetime(df['time'], unit='s')
        deltas = dt.diff().dropna()
        # Filter out overnight gaps (keep only < 2h)
        intraday = deltas[deltas < pd.Timedelta(hours=2)]
        median_min = int(intraday.median().total_seconds() / 60)
        # Round to nearest standard frequency
        for standard in [1, 5, 10, 15, 30, 60]:
            if abs(median_min - standard) <= standard * 0.3:
                return standard
        return median_min
        
        
    def get_realized_vol_paths(
        self,
        segment_length: int | None = None,
        rv_window: int | None = None,
        *,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        """
        Computes Realized Variance paths for P-measure training.
        
          Strategy:
             1. Compute an intraday rolling RV series aligned to the simulator's
                 time step (`simulation.bar_interval_min`). By default the window is
                 ~1 trading day (scaled from `data.rv_window`).
            2. Segment the resulting RV series into overlapping training paths.
        """
        if segment_length is None:
            segment_length = self.config['data']['segment_length']
            
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Please place S&P 500 data at: {self.file_path}")

        target_bar_min = int(self.config.get('simulation', {}).get('bar_interval_min', self._bar_min))
        train_window_bars = int(self.config.get('data', {}).get('rv_training_window_bars', 1))
        train_window_bars = max(1, train_window_bars)
        window_str = f"window={train_window_bars} bar" + ("s" if train_window_bars != 1 else "")
        print(f"   Intraday RV config: bar={target_bar_min}m, {window_str}")

        daily_window = int(rv_window or self.config['data'].get('daily_rv_window', 5))
        daily_min_year = int(self.config['data'].get('daily_min_year', 2000))

        paths, daily_rv = build_realized_vol_training_paths(
            intraday_file=self.file_path,
            segment_length=int(segment_length),
            stride_ratio=float(self.stride_ratio),
            base_bar_min=int(self._bar_min),
            target_bar_min=int(target_bar_min),
            trading_hours_per_day=float(self.trading_hours),
            trading_days_per_year=float(self.trading_days_per_year),
            train_window_bars=int(train_window_bars),
            rv_min=float(self.rv_min),
            rv_max=float(self.rv_max),
            daily_window_days=int(daily_window),
            daily_min_year=int(daily_min_year),
            shuffle=bool(shuffle),
            seed=seed,
        )

        print(f"   Final RV series: {len(daily_rv)} points")
        print(f"   Mean RV: {np.mean(daily_rv):.4f} (Vol: {np.sqrt(np.mean(daily_rv))*100:.1f}%)")
        
        print(f"Dataset ready: {paths.shape[0]} paths of length {segment_length}")
        return jnp.array(paths)




@dataclass(frozen=True)
class RVSeriesSettings:
    """Settings for computing daily (log) RV/TSRV series from CSV files.

    Kept here so Hurst analysis can be tested by passing explicit settings.
    """

    max_gap_hours: float = 4.0
    tsrv_frequencies: set[str] = field(default_factory=lambda: {"5s"})
    daily_window_days: int = 5
    daily_min_year: int = 1990
    min_bars_per_day: dict[str, int] = field(
        default_factory=lambda: {
            "5s": 200,
            "5m": 20,
            "15m": 10,
            "30m": 6,
            "1h": 4,
            "daily": 1,
        }
    )





def _coerce_datetime(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    utc: bool = True,
) -> pd.Series:
    if time_col not in df.columns:
        raise KeyError(f"Missing column '{time_col}'")

    if np.issubdtype(df[time_col].dtype, np.number):
        return pd.to_datetime(df[time_col], unit="s", utc=utc, errors="coerce")

    return pd.to_datetime(df[time_col], utc=utc, errors="coerce")


def rv_series_settings_from_config(cfg: dict) -> RVSeriesSettings:
    """Build RVSeriesSettings from the global params.yaml dict."""
    if not isinstance(cfg, dict):
        return RVSeriesSettings()
    data_cfg = cfg.get("data", {}) or {}
    hurst_cfg = data_cfg.get("hurst_estimation", {}) or {}
    if not isinstance(hurst_cfg, dict):
        return RVSeriesSettings()

    tsrv = hurst_cfg.get("tsrv_frequencies")
    tsrv_set = {str(x) for x in tsrv} if isinstance(tsrv, (list, tuple, set)) else {"5s"}

    min_bars = hurst_cfg.get("min_bars_per_day")
    min_bars_map = min_bars if isinstance(min_bars, dict) else {}
    # Merge defaults with user overrides
    merged_min_bars = RVSeriesSettings().min_bars_per_day | {
        str(k): int(v) for k, v in min_bars_map.items() if v is not None
    }

    max_gap_hours = float(data_cfg.get("max_gap_hours", RVSeriesSettings().max_gap_hours))
    return RVSeriesSettings(
        max_gap_hours=max_gap_hours,
        tsrv_frequencies=tsrv_set,
        daily_window_days=int(hurst_cfg.get("daily_window_days", 5)),
        daily_min_year=int(hurst_cfg.get("daily_min_year", 1990)),
        min_bars_per_day=merged_min_bars,
    )




def segment_trading_days(
    df: pd.DataFrame,
    *,
    max_gap_hours: float = 4.0,
    min_bars_per_day: int = 10,
    time_col: str = "time",
    price_col: str = "close",
    utc: bool = True,
) -> list[pd.DataFrame]:
    """Split an intraday series into trading-day/session segments using gap detection.

    A new segment starts when the time delta between consecutive bars exceeds
    `max_gap_hours`.

    Returns a list of DataFrames each containing ['datetime', price_col].
    """
    if price_col not in df.columns:
        raise KeyError(f"Missing column '{price_col}'")

    out = df.copy()
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    out["datetime"] = _coerce_datetime(out, time_col=time_col, utc=utc)
    out = out.sort_values("datetime").dropna(subset=["datetime", price_col]).reset_index(drop=True)

    dt_sec = out["datetime"].diff().dt.total_seconds()
    gap_mask = dt_sec > (max_gap_hours * 3600.0)
    out["day_id"] = gap_mask.cumsum()

    segments: list[pd.DataFrame] = []
    for _, grp in out.groupby("day_id"):
        if len(grp) >= int(min_bars_per_day):
            segments.append(grp[["datetime", price_col]].reset_index(drop=True))

    return segments


def compute_rv_from_segments(
    segments: list[pd.DataFrame],
    *,
    price_col: str = "close",
    floor: float = 1e-16,
) -> tuple[np.ndarray, list[pd.Timestamp]]:
    """Compute daily realized variance RV_d = sum_i (Δ log P_i)^2 from segments."""
    rv: list[float] = []
    dates: list[pd.Timestamp] = []

    for seg in segments:
        prices = seg[price_col].values.astype(float)
        if len(prices) < 2:
            continue
        log_returns = np.diff(np.log(prices))
        rv_day = float(np.sum(log_returns**2))
        if rv_day > floor:
            rv.append(rv_day)
            dates.append(pd.Timestamp(seg["datetime"].iloc[-1]))

    return np.asarray(rv, dtype=float), dates


def compute_tsrv_from_segments(
    segments: list[pd.DataFrame],
    *,
    K: Optional[int] = None,
    price_col: str = "close",
    min_returns: int = 20,
    floor: float = 1e-16,
) -> tuple[np.ndarray, list[pd.Timestamp]]:
    """Two-Scale Realized Variance (TSRV) per Zhang-Mykland-Aït-Sahalia (2005)."""
    tsrv: list[float] = []
    dates: list[pd.Timestamp] = []

    for seg in segments:
        prices = seg[price_col].values.astype(float)
        if len(prices) < 3:
            continue
        log_prices = np.log(prices)
        n = len(log_prices) - 1  # number of returns
        if n < int(min_returns):
            continue

        K_use = int(K) if K is not None else max(2, int(np.round(n ** (2.0 / 3.0))))

        returns_all = np.diff(log_prices)
        rv_fast = float(np.sum(returns_all**2))
        n_fast = n

        rv_slow = 0.0
        n_slow_total = 0
        for k in range(K_use):
            sub_prices = log_prices[k::K_use]
            if len(sub_prices) < 2:
                continue
            sub_returns = np.diff(sub_prices)
            rv_slow += float(np.sum(sub_returns**2))
            n_slow_total += len(sub_returns)

        rv_slow /= float(K_use)
        n_bar_slow = n_slow_total / float(K_use)

        tsrv_day = rv_slow - (n_bar_slow / float(n_fast)) * rv_fast

        correction = 1.0 - n_bar_slow / float(n_fast)
        if correction > 0.1:
            tsrv_day /= correction

        tsrv_day = max(float(tsrv_day), floor)
        tsrv.append(tsrv_day)
        dates.append(pd.Timestamp(seg["datetime"].iloc[-1]))

    return np.asarray(tsrv, dtype=float), dates


def compute_daily_rv_multiscale(
    file_path: str | Path,
    *,
    max_gap_hours: float = 4.0,
    min_bars_per_day: int = 10,
    use_tsrv: bool = False,
    tsrv_K: Optional[int] = None,
    time_col: str = "time",
    price_col: str = "close",
    utc: bool = True,
) -> tuple[np.ndarray, list[pd.Timestamp]]:
    """Load a CSV and compute daily RV/TSRV from intraday prices."""
    file_path = Path(file_path)
    df = pd.read_csv(file_path)

    segments = segment_trading_days(
        df,
        max_gap_hours=max_gap_hours,
        min_bars_per_day=min_bars_per_day,
        time_col=time_col,
        price_col=price_col,
        utc=utc,
    )

    if use_tsrv:
        return compute_tsrv_from_segments(segments, K=tsrv_K, price_col=price_col)

    return compute_rv_from_segments(segments, price_col=price_col)


def compute_rv_from_daily_close_series(
    file_path: str | Path,
    *,
    window_days: int = 5,
    min_year: int = 2000,
    trading_days_per_year: float = 252.0,
    time_col: str = "time",
    close_col: str = "close",
    utc: bool = True,
) -> pd.Series:
    """Annualized close-to-close RV estimate (rolling) as a date-indexed Series."""
    file_path = Path(file_path)
    if not file_path.exists():
        return pd.Series(dtype=float)

    df = pd.read_csv(file_path)
    if close_col not in df.columns and "Close" in df.columns:
        close_col = "Close"
    if close_col not in df.columns:
        return pd.Series(dtype=float)

    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    dt = _coerce_datetime(df, time_col=time_col, utc=utc)
    df = df.assign(datetime=dt).dropna(subset=["datetime", close_col])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date
    df = df[df["datetime"].dt.year >= int(min_year)].reset_index(drop=True)

    # One close per date
    df = df.groupby("date", as_index=False).last()
    prices = df[close_col].values.astype(float)
    if len(prices) < (window_days + 1):
        return pd.Series(dtype=float)

    log_ret = np.diff(np.log(prices))
    window_days = int(window_days)
    annualize = float(trading_days_per_year) / float(window_days)
    rv = pd.Series(log_ret * log_ret).rolling(window_days).sum() * annualize
    rv = rv.dropna()

    out = pd.Series(rv.values.astype(float), index=df.loc[rv.index, "date"].values)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def compute_log_rv_series_from_file(
    file_path: str | Path,
    *,
    freq_label: str,
    settings: RVSeriesSettings,
    min_obs: int = 9,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Compute daily log(RV) or log(TSRV) from one CSV file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    if freq_label == "daily":
        return compute_log_rv_from_daily_close(
            file_path,
            window_days=settings.daily_window_days,
            min_year=settings.daily_min_year,
            non_overlapping=True,
            time_col="time",
            close_col="close",
            utc=True,
        )

    min_bars = int(settings.min_bars_per_day.get(freq_label, 10))
    use_tsrv = freq_label in settings.tsrv_frequencies
    rv, dates = compute_daily_rv_multiscale(
        file_path,
        max_gap_hours=float(settings.max_gap_hours),
        min_bars_per_day=min_bars,
        use_tsrv=use_tsrv,
    )

    if len(rv) < int(min_obs):
        raise ValueError(
            f"{file_path.name}: only {len(rv)} valid days (need ≥{min_obs}). "
            f"Check data quality or relax min_bars_per_day."
        )

    return np.log(rv), pd.DatetimeIndex(dates)


def compute_log_rv_from_daily_close(
    file_path: str | Path,
    *,
    window_days: int = 5,
    min_year: int = 1990,
    non_overlapping: bool = True,
    time_col: str = "time",
    close_col: str = "close",
    utc: bool = True,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Compute log(RV) from daily close prices using windowed close-to-close returns.

    Returns a non-overlapping (by default) windowed RV series to reduce induced
    autocorrelation bias in downstream Hurst estimation.
    """
    file_path = Path(file_path)
    df = pd.read_csv(file_path)

    # Allow TradingView 'Close' column
    if close_col not in df.columns and "Close" in df.columns:
        close_col = "Close"

    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df["datetime"] = _coerce_datetime(df, time_col=time_col, utc=utc)
    df = df.sort_values("datetime").dropna(subset=["datetime", close_col]).reset_index(drop=True)

    df = df[df["datetime"].dt.year >= int(min_year)].reset_index(drop=True)

    prices = df[close_col].values.astype(float)
    if len(prices) < (window_days + 1):
        return np.array([], dtype=float), pd.DatetimeIndex([])

    log_returns = np.diff(np.log(prices))
    dates = df["datetime"].values[1:]  # per-return timestamp

    window_days = int(window_days)
    stride = window_days if non_overlapping else 1

    rv_list: list[float] = []
    dates_list: list[pd.Timestamp] = []

    for start in range(0, len(log_returns) - window_days + 1, stride):
        chunk = log_returns[start : start + window_days]
        rv_val = float(np.sum(chunk**2))
        if rv_val > 1e-16:
            rv_list.append(rv_val)
            dates_list.append(pd.Timestamp(dates[start + window_days - 1]))

    if len(rv_list) == 0:
        return np.array([], dtype=float), pd.DatetimeIndex([])

    return np.log(np.asarray(rv_list, dtype=float)), pd.DatetimeIndex(dates_list)


def compute_rolling_annualized_rv_from_prices(
    prices: np.ndarray,
    *,
    window_days: int,
    trading_days_per_year: float = 252.0,
) -> np.ndarray:
    """Rolling annualized close-to-close RV from a price array.

    Output is aligned to the input `prices` index:
      - length == len(prices)
      - first valid value appears at index `window_days`
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    out = np.full(n, np.nan, dtype=float)
    window_days = int(window_days)
    if window_days <= 0 or n < (window_days + 1):
        return out

    log_ret = np.diff(np.log(prices))
    annualize = float(trading_days_per_year) / float(window_days)
    rv = pd.Series(log_ret * log_ret).rolling(window_days).sum() * annualize
    out[1:] = rv.values.astype(float)
    return out


def compute_forward_annualized_rv_from_prices(
    prices: np.ndarray,
    *,
    start_idx: int,
    horizon_days: int = 21,
    trading_days_per_year: float = 252.0,
) -> Optional[float]:
    """Forward annualized RV over the next `horizon_days` from `start_idx`.

    Matches the common backtest definition:
      RV(start) = sum_{i=1..horizon} r_i^2 * (252 / horizon)
    where r_i are close-to-close log-returns.
    """
    prices = np.asarray(prices, dtype=float)
    horizon_days = int(horizon_days)
    start_idx = int(start_idx)
    if horizon_days <= 0:
        return None
    if start_idx < 0 or (start_idx + horizon_days) >= len(prices):
        return None

    future = prices[start_idx : start_idx + horizon_days + 1]
    log_ret = np.diff(np.log(future))
    return float(np.sum(log_ret * log_ret) * float(trading_days_per_year) / float(horizon_days))


def compute_single_day_annualized_rv_from_intraday_df(
    intraday_day: pd.DataFrame,
    *,
    price_col: str = "close",
    trading_days_per_year: float = 252.0,
    min_bars: int = 10,
) -> Optional[float]:
    """Annualized single-day RV from an intraday DataFrame.

    The input is expected to be already filtered to a single trading day.
    """
    if price_col not in intraday_day.columns:
        return None
    prices = pd.to_numeric(intraday_day[price_col], errors="coerce").dropna().values.astype(float)
    if len(prices) < max(2, int(min_bars)):
        return None
    log_ret = np.diff(np.log(prices))
    if len(log_ret) < max(1, int(min_bars) - 1):
        return None
    rv_day = float(np.sum(log_ret * log_ret))
    return float(rv_day * float(trading_days_per_year))


def compute_intraday_annualized_rv_series_from_file(
    file_path: str | Path,
    *,
    base_bar_min: int,
    target_bar_min: int,
    trading_hours_per_day: float = 6.5,
    trading_days_per_year: float = 252.0,
    window_bars: int = 1,
    time_col: str = "time",
    price_col: str = "close",
    utc: bool = True,
) -> np.ndarray:
    """Annualized intraday RV series aligned to `target_bar_min`.

    - Computes log-returns within each day (no overnight returns).
    - Resamples to `target_bar_min` by summing consecutive base-bar returns.
    - Produces per-bar RV = r_bar^2 annualized by 252 * bars_per_day.
    - Optionally smooths with a rolling mean over `window_bars`.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return np.array([], dtype=float)

    base_bar_min = int(base_bar_min)
    target_bar_min = int(target_bar_min)
    window_bars = int(window_bars)
    if base_bar_min <= 0 or target_bar_min <= 0:
        raise ValueError("base_bar_min and target_bar_min must be positive")
    if window_bars <= 0:
        raise ValueError("window_bars must be >= 1")

    df = pd.read_csv(file_path)
    if time_col not in df.columns or price_col not in df.columns:
        return np.array([], dtype=float)

    df["datetime"] = _coerce_datetime(df, time_col=time_col, utc=utc)
    df = df.sort_values("datetime").reset_index(drop=True)
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["datetime", price_col])

    df["date"] = df["datetime"].dt.date
    df["log_return"] = df.groupby("date")[price_col].transform(lambda x: np.log(x / x.shift(1)))
    df = df.dropna(subset=["log_return"])

    if target_bar_min != base_bar_min:
        if target_bar_min % base_bar_min != 0:
            raise ValueError(
                f"target_bar_min={target_bar_min} must be a multiple of base_bar_min={base_bar_min}"
            )
        k = int(target_bar_min // base_bar_min)
        df["bar_index"] = df.groupby("date").cumcount()
        df["bucket"] = df["bar_index"] // k
        per_bar = df.groupby(["date", "bucket"])["log_return"].sum().rename("r")
    else:
        df["bar_index"] = df.groupby("date").cumcount()
        per_bar = df.set_index(["date", "bar_index"])['log_return'].rename("r")

    bars_per_day_target = int(float(trading_hours_per_day) * 60.0 / float(target_bar_min))
    r = per_bar.values.astype(float)
    if len(r) < max(window_bars, 10):
        return np.array([], dtype=float)

    annualize = float(trading_days_per_year) * float(bars_per_day_target)
    r2 = r * r

    if window_bars == 1:
        return (r2 * annualize).astype(float)

    rv = pd.Series(r2).rolling(window_bars).mean() * annualize
    return rv.dropna().values.astype(float)

def build_realized_vol_training_paths(
    *,
    intraday_file: str | Path,
    segment_length: int,
    stride_ratio: float,
    base_bar_min: int,
    target_bar_min: int,
    trading_hours_per_day: float,
    trading_days_per_year: float,
    train_window_bars: int,
    rv_min: float,
    rv_max: float,
    daily_window_days: int = 5,
    daily_min_year: int = 2000,
    shuffle: bool = True,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build overlapping RV training paths from intraday source 

    Returns:
      - paths: array with shape (n_paths, segment_length)
      - rv_series: filtered 1D RV series used to create the paths
    """
    intraday_rv = compute_intraday_annualized_rv_series_from_file(
        intraday_file,
        base_bar_min=int(base_bar_min),
        target_bar_min=int(target_bar_min),
        trading_hours_per_day=float(trading_hours_per_day),
        trading_days_per_year=float(trading_days_per_year),
        window_bars=int(max(1, train_window_bars)),
        time_col="time",
        price_col="close",
        utc=True,
    )

    if len(intraday_rv) >= int(segment_length) + 1:
        rv_series = intraday_rv
    else:
        raise ValueError(f"Intraday RV insufficient ({len(intraday_rv)} points), ")
   
    rv_series = rv_series[(rv_series > float(rv_min)) & (rv_series < float(rv_max))]
    if len(rv_series) == 0:
        raise ValueError("No valid RV data after filtering.")

    stride = max(1, int(int(segment_length) * float(stride_ratio)))
    paths = [rv_series[i : i + int(segment_length)] for i in range(0, len(rv_series) - int(segment_length), stride)]
    if len(paths) == 0:
        raise ValueError(
            f"Not enough RV data ({len(rv_series)} points) for segment_length={segment_length}. "
            f"Need at least {int(segment_length) + 1}."
        )

    arr = np.asarray(paths, dtype=float)
    if shuffle:
        if seed is None:
            np.random.shuffle(arr)
        else:
            rng = np.random.default_rng(int(seed))
            rng.shuffle(arr)
    return arr, rv_series.astype(float)
