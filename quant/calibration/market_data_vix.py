"""
VIX Term Structure & Futures Loader
=====================================
Unified loader for VIX volatility term structure from two data sources:

1. **CBOE VIX Futures** (data/cboe_vix_futures_full/)
   Individual futures contracts with expiry dates, 2013–present.
   Columns: Trade Date, Futures, Open, High, Low, Close, Settle, ...

2. **TradingView Volatility Indices** (data/market/volatility/)
   CBOE-computed VIX indices at fixed tenures:
     - VIX1D (1-day)   — τ = 1/365   — since 2022
     - VIX9D (9-day)   — τ = 9/365   — since 2011
     - VIX   (30-day)  — τ = 30/365  — since 1990
     - VIX3M (3-month) — τ = 90/365  — since 2007
     - VIX6M (6-month) — τ = 180/365 — since 2008
     - VIX1Y (1-year)  — τ = 365/365 — since 2018

3. **TradingView VIX Futures** (data/market/vix_futures/)
   Continuous front-month (VX1) and 2nd-month (VX2) contracts.
   High-frequency (5m–1h) and daily. VX2 daily since 2004.

The TradingView volatility indices are *directly* the VIX term structure
at fixed tenures — no need to interpolate between futures expiries.
This is the cleanest calibration target for the rBergomi model.

Mathematical framework
----------------------
Under Q, a VIX futures contract expiring at T prices as:

    F^{VIX}(0, T) = E^Q[ VIX_T ]

where

    VIX²_T = (1/τ) · E^Q_T[ ∫_T^{T+τ} V_s ds ]

In the rBergomi model (Bayer, Friz & Gatheral 2016):

    V_s = ξ₀(s) · exp( η Ŵᴴ_s - ½η² Var[Ŵᴴ_s] )

The VIX term structure constrains the forward variance curve ξ₀(t)
and the vol-of-vol parameters (η, H) jointly.

Key identity (Gatheral & Keller-Ressel 2019):
    E^Q[VIX²_T] = (1/τ) ∫_T^{T+τ} ξ₀(s) · exp(η²·K_H(s,T)) ds

where K_H(s,T) is a kernel depending on H. This connects the VIX TS
to model parameters analytically (at the VIX² level) before needing
MC for the convexity adjustment E[VIX] vs sqrt(E[VIX²]).

References
----------
[1] Bayer, Friz & Gatheral (2016). Pricing under rough volatility. QF.
[2] Gatheral (2006). The Volatility Surface, ch. 4.
[3] Rømer (2022). Empirical analysis of rough and classical stochastic
    volatility models to the SPX and VIX markets. QF.
[4] Guennoun, Jacquier, Roome & Shi (2018). Asymptotic behavior of the
    fractional Heston-like model. AMF.
[5] Gatheral & Keller-Ressel (2019). Affine forward variance models. FM.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VIXFuturesTerm:
    """
    A single VIX futures term structure snapshot at a given trade date.

    Attributes
    ----------
    trade_date  : observation date
    expiries    : sorted array of expiry dates
    prices      : VIX futures prices (close or settle) for each expiry
    days_to_exp : days to expiry for each contract
    T_years     : time to expiry in years
    vix_spot    : VIX spot level on that date (if available)
    """
    trade_date: pd.Timestamp
    expiries: np.ndarray
    prices: np.ndarray
    days_to_exp: np.ndarray
    T_years: np.ndarray
    vix_spot: Optional[float] = None

    @property
    def n_contracts(self) -> int:
        return len(self.prices)

    def term_structure_slope(self) -> float:
        """Contango (+) or backwardation (-) slope."""
        if self.n_contracts < 2:
            return 0.0
        return float((self.prices[-1] - self.prices[0]) / (self.T_years[-1] - self.T_years[0]))

    def is_contango(self) -> bool:
        """True if term structure is in contango (upward sloping)."""
        return self.term_structure_slope() > 0


@dataclass
class VIXFuturesData:
    """
    Complete VIX futures dataset parsed from CBOE CSV files.

    Attributes
    ----------
    all_data    : full DataFrame with parsed columns
    trade_dates : unique trade dates
    """
    all_data: pd.DataFrame
    trade_dates: np.ndarray

    def get_term_structure(
        self,
        trade_date: str | pd.Timestamp,
        min_dte: int = 5,
        max_dte: int = 240,
        vix_spot: Optional[float] = None,
    ) -> VIXFuturesTerm:
        """
        Extract VIX futures term structure for a single trade date.

        Parameters
        ----------
        trade_date : target date (exact match or nearest before)
        min_dte : minimum days to expiry (exclude near-expiry contracts)
        max_dte : maximum days to expiry
        vix_spot : optional VIX spot for that date

        Returns
        -------
        VIXFuturesTerm
        """
        td = pd.to_datetime(trade_date)
        df = self.all_data

        # Find closest date on or before target
        available = df[df['trade_date'] <= td]['trade_date'].unique()
        if len(available) == 0:
            raise ValueError(f"No data on or before {td.date()}")
        closest = max(available)

        day_data = df[df['trade_date'] == closest].copy()
        day_data = day_data[
            (day_data['days_to_exp'] >= min_dte) &
            (day_data['days_to_exp'] <= max_dte) &
            (day_data['price'] > 0)
        ]

        if len(day_data) == 0:
            raise ValueError(f"No valid contracts on {closest.date()}")

        day_data = day_data.sort_values('days_to_exp')

        return VIXFuturesTerm(
            trade_date=closest,
            expiries=day_data['expiry_date'].values,
            prices=day_data['price'].values,
            days_to_exp=day_data['days_to_exp'].values,
            T_years=day_data['T_years'].values,
            vix_spot=vix_spot,
        )

    def get_latest_term_structure(self, **kwargs) -> VIXFuturesTerm:
        """Get the most recent term structure."""
        latest = self.trade_dates[-1]
        return self.get_term_structure(latest, **kwargs)

    def get_historical_term_structures(
        self,
        start_date: str = None,
        end_date: str = None,
        freq: str = 'W',
        **kwargs,
    ) -> list[VIXFuturesTerm]:
        """
        Extract term structures at regular intervals (for time-series analysis).

        Parameters
        ----------
        start_date, end_date : date range
        freq : pandas frequency string ('D', 'W', 'M')
        """
        dates = pd.Series(self.trade_dates).sort_values()
        if start_date:
            dates = dates[dates >= pd.to_datetime(start_date)]
        if end_date:
            dates = dates[dates <= pd.to_datetime(end_date)]

        # Resample to requested frequency
        date_index = pd.DatetimeIndex(dates)
        resampled = date_index.to_series().resample(freq).last().dropna()

        term_structures = []
        for d in resampled:
            try:
                ts = self.get_term_structure(d, **kwargs)
                term_structures.append(ts)
            except ValueError:
                continue

        return term_structures


def load_vix_futures(
    data_dir: str = "data/cboe_vix_futures_full",
    use_settle: bool = False,
) -> VIXFuturesData:
    """
    Load and parse CBOE VIX futures CSV data.

    Parameters
    ----------
    data_dir : directory containing vix_futures_all.csv
    use_settle : if True, prefer Settle over Close price.
                 Settle = 0 in many rows, so Close is usually better.

    Returns
    -------
    VIXFuturesData with parsed term structure
    """
    path = Path(data_dir) / "vix_futures_all.csv"
    if not path.exists():
        raise FileNotFoundError(f"VIX futures file not found: {path}")

    df = pd.read_csv(path)

    # Parse dates
    df['trade_date'] = pd.to_datetime(df['Trade Date'], errors='raise', format='mixed').dt.normalize()
    df['expiry_date'] = pd.to_datetime(df['expiration_date'], errors='raise', format='mixed').dt.normalize()

    # Parse price: prefer Close, fallback to Settle
    df['close_px'] = pd.to_numeric(df['Close'], errors='coerce')
    df['settle_px'] = pd.to_numeric(df['Settle'], errors='coerce')

    if use_settle:
        df['price'] = df['settle_px'].where(df['settle_px'] > 0, df['close_px'])
    else:
        df['price'] = df['close_px'].where(df['close_px'] > 0, df['settle_px'])

    # Compute days to expiry and T in years
    df['days_to_exp'] = (df['expiry_date'] - df['trade_date']).dt.days
    df['T_years'] = df['days_to_exp'] / 365.0

    # Parse contract month from "Futures" column for identification
    df['contract_label'] = df['Futures'].astype(str)

    # Clean: remove expired or invalid rows
    df = df[
        (df['days_to_exp'] > 0) &
        (df['price'] > 0) &
        df['price'].notna()
    ].copy()

    # Parse volume and OI
    df['volume'] = pd.to_numeric(df['Total Volume'], errors='coerce').fillna(0)
    df['open_interest'] = pd.to_numeric(df['Open Interest'], errors='coerce').fillna(0)

    # Keep useful columns
    keep_cols = [
        'trade_date', 'expiry_date', 'contract_label',
        'price', 'days_to_exp', 'T_years',
        'volume', 'open_interest',
    ]
    df = df[keep_cols].sort_values(['trade_date', 'days_to_exp']).reset_index(drop=True)

    trade_dates = np.sort(df['trade_date'].unique())

    return VIXFuturesData(all_data=df, trade_dates=trade_dates)


def load_vix_spot_history(
    vix_daily_path: str = "data/market/volatility/vix_daily.csv",
) -> pd.Series:
    """
    Load daily VIX spot (for matching with futures trade dates).

    Tries TradingView first (richest: 1990–present), then market/ fallback.

    Returns
    -------
    pd.Series indexed by date, values = VIX close.
    """
    candidates = [
        Path(vix_daily_path),
        Path("data/market/volatility/vix_daily.csv"),
        Path("data/market/volatility/vix_daily.csv"),
    ]

    path = None
    for p in candidates:
        if p.exists():
            path = p
            break

    if path is None:
        return pd.Series(dtype=float)

    df = pd.read_csv(path)

    # Handle TradingView format (time as UNIX epoch)
    if 'time' in df.columns:
        df['date'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.normalize()
    elif 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date'])
    else:
        df['date'] = pd.to_datetime(df.iloc[:, 0])

    df['close'] = pd.to_numeric(df['close'] if 'close' in df.columns else df['Close'],
                                errors='coerce')

    return df.set_index('date')['close'].dropna().sort_index()


# ═══════════════════════════════════════════════════════════════════
# TradingView VIX Term Structure (fixed-tenure CBOE indices)
# ═══════════════════════════════════════════════════════════════════

# CBOE VIX index family — each measures implied vol over a fixed window τ.
# These are NOT futures but options-implied vol indices published by CBOE.
# VIX1D = 1-day expected vol, VIX9D = 9-day, VIX = 30-day, etc.
VIX_INDEX_TENORS = {
    "vix1d": {"tau_days": 1,   "T_years": 1 / 365},
    "vix9d": {"tau_days": 9,   "T_years": 9 / 365},
    "vix":   {"tau_days": 30,  "T_years": 30 / 365},
    "vix3m": {"tau_days": 90,  "T_years": 90 / 365},
    "vix6m": {"tau_days": 180, "T_years": 180 / 365},
    "vix1y": {"tau_days": 365, "T_years": 365 / 365},
}


@dataclass
class VIXTermStructureSnapshot:
    """
    VIX term structure at a single date, built from CBOE indices.

    This is the most direct calibration target: each VIX index
    measures the market's expectation of integrated variance over
    a specific horizon τ:

        VIX(τ)² = (1/τ) E^Q[ ∫₀^τ V_s ds ]

    so VIX(τ)/100 is the annualized implied vol for horizon τ.
    """
    date: pd.Timestamp
    tenors_days: np.ndarray     # [1, 9, 30, 90, 180, 365]
    tenors_years: np.ndarray    # [1/365, 9/365, ...]
    vix_levels: np.ndarray      # VIX index values (in vol points, e.g. 18.5)
    labels: list                # ["vix1d", "vix9d", "vix", ...]

    @property
    def n_tenors(self) -> int:
        return len(self.vix_levels)

    @property
    def implied_variances(self) -> np.ndarray:
        """Convert VIX levels to implied variances: σ² = (VIX/100)²."""
        return (self.vix_levels / 100.0) ** 2

    @property
    def total_variances(self) -> np.ndarray:
        """Total variance: σ²·τ for each tenor."""
        return self.implied_variances * self.tenors_years

    def term_structure_slope(self) -> float:
        """Slope of VIX term structure (contango > 0, backwardation < 0)."""
        if self.n_tenors < 2:
            return 0.0
        return float(
            (self.vix_levels[-1] - self.vix_levels[0]) /
            (self.tenors_years[-1] - self.tenors_years[0])
        )


def _load_tradingview_csv(path: Path) -> pd.DataFrame:
    """Load a TradingView CSV (time as Unix epoch, OHLC columns)."""
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.normalize()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df[['date', 'close']].dropna().drop_duplicates('date').set_index('date').sort_index()


def load_vix_term_structure(
    data_dir: str = "data/market/volatility",
    freq: str = "daily",
    tenors: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load the full VIX term structure from TradingView indices.

    Returns a DataFrame indexed by date with columns = tenor labels,
    values = VIX level (e.g., 18.5 means 18.5% annualized implied vol).

    Parameters
    ----------
    data_dir : path to volatility directory
    freq : "daily", "1h", "15m", etc.
    tenors : subset of tenor labels to load (None = all available)

    Returns
    -------
    DataFrame with date index and VIX level columns
    """
    base = Path(data_dir)
    if tenors is None:
        tenors = list(VIX_INDEX_TENORS.keys())

    dfs = {}
    for label in tenors:
        fname = f"{label}_{freq}.csv"
        path = base / fname
        if not path.exists():
            continue
        series = _load_tradingview_csv(path)['close']
        series.name = label
        dfs[label] = series

    if not dfs:
        raise FileNotFoundError(
            f"No VIX index files found in {base} for freq={freq}. "
            f"Looked for: {[f'{t}_{freq}.csv' for t in tenors]}"
        )

    # Join on dates (inner join = only dates where all loaded tenors exist)
    result = pd.DataFrame(dfs)
    result = result.dropna()
    result = result.sort_index()

    return result


def get_vix_term_snapshot(
    date: str | pd.Timestamp = None,
    data_dir: str = "data/market/volatility",
) -> VIXTermStructureSnapshot:
    """
    Get VIX term structure for a specific date (latest if None).

    Parameters
    ----------
    date : target date (finds closest available)
    data_dir : volatility data directory

    Returns
    -------
    VIXTermStructureSnapshot
    """
    ts_df = load_vix_term_structure(data_dir)

    if date is None:
        row_date = ts_df.index[-1]
    else:
        target = pd.to_datetime(date).normalize()
        # Find closest date on or before target
        available = ts_df.index[ts_df.index <= target]
        if len(available) == 0:
            available = ts_df.index
        row_date = available[-1]

    row = ts_df.loc[row_date]
    available_labels = [c for c in row.index if c in VIX_INDEX_TENORS and pd.notna(row[c])]

    tenors_days = np.array([VIX_INDEX_TENORS[l]["tau_days"] for l in available_labels])
    tenors_years = np.array([VIX_INDEX_TENORS[l]["T_years"] for l in available_labels])
    vix_levels = np.array([row[l] for l in available_labels])

    return VIXTermStructureSnapshot(
        date=row_date,
        tenors_days=tenors_days,
        tenors_years=tenors_years,
        vix_levels=vix_levels,
        labels=available_labels,
    )


def get_vix_term_snapshot_from_futures(
    date: str | pd.Timestamp = None,
    data_dir: str = "data/cboe_vix_futures_full",
    tenors: list[int] = None,
) -> VIXTermStructureSnapshot:
    """
    Get VIX term structure from CBOE futures (front, 2M, 3M contracts).
    
    This builds the term structure directly from futures prices instead of
    from the stale TradingView indices. Each futures contract is treated
    as a tenor based on its days-to-expiry.

    Parameters
    ----------
    date : target date (finds closest available)
    data_dir : directory with CBOE futures CSV files
    tenors : specific tenor days to extract (if None, use all available contracts)

    Returns
    -------
    VIXTermStructureSnapshot
    """
    # Load CBOE futures
    vix_fut = load_vix_futures(data_dir)
    
    # Get term structure for this date
    if date is None:
        date = vix_fut.trade_dates[-1]
    
    ts = vix_fut.get_term_structure(date, min_dte=5, max_dte=365)
    
    if tenors is None:
        # Use all available contracts
        selected_idx = np.arange(len(ts.prices))
        selected_tenors = ts.days_to_exp
        selected_prices = ts.prices
        selected_labels = [f"fut_{d}d" for d in ts.days_to_exp]
    else:
        # Filter to requested tenors (nearest match)
        selected_idx = []
        selected_tenors = []
        selected_prices = []
        selected_labels = []
        
        for target_tenor in tenors:
            # Find nearest contract
            closest_idx = np.argmin(np.abs(ts.days_to_exp - target_tenor))
            selected_idx.append(closest_idx)
            selected_tenors.append(ts.days_to_exp[closest_idx])
            selected_prices.append(ts.prices[closest_idx])
            selected_labels.append(f"fut_{ts.days_to_exp[closest_idx]}d")
    
    return VIXTermStructureSnapshot(
        date=ts.trade_date,
        tenors_days=np.array(selected_tenors),
        tenors_years=np.array(selected_tenors) / 365.0,
        vix_levels=np.array(selected_prices),
        labels=selected_labels,
    )


# ═══════════════════════════════════════════════════════════════════
# TradingView VIX Futures (continuous contracts VX1, VX2)
# ═══════════════════════════════════════════════════════════════════

def load_vix_futures_continuous(
    data_dir: str = "data/market/vix_futures",
    freq: str = "daily",
) -> pd.DataFrame:
    """
    Load continuous VIX futures (VX1 = front month, VX2 = 2nd month).

    These are pre-rolled continuous contracts from TradingView.
    VX2 daily goes back to 2004 — much longer than CBOE parsed data.

    Returns DataFrame with columns ['vx1', 'vx2'] indexed by date.
    """
    base = Path(data_dir)
    dfs = {}

    for contract in ['vx1', 'vx2']:
        fname = f"{contract}_{freq}.csv"
        path = base / fname
        if not path.exists():
            continue
        series = _load_tradingview_csv(path)['close']
        series.name = contract
        dfs[contract] = series

    if not dfs:
        raise FileNotFoundError(f"No VX futures found in {base} for freq={freq}")

    result = pd.DataFrame(dfs).dropna().sort_index()
    return result


# ═══════════════════════════════════════════════════════════════════
# Unified calibration data assembly
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CalibrationMarketData:
    """
    All market data needed for joint SPX-VIX calibration, assembled
    from multiple sources.

    Attributes
    ----------
    vix_term_structure : VIX term structure snapshot (VIX1D → VIX1Y)
    vix_futures_term   : VIX futures term structure (CBOE contracts)
    vix_spot           : VIX spot level
    spx_surface        : SPX options surface DataFrame
    spx_spot           : SPX/SPY spot price
    risk_free_rate     : SOFR rate
    vvix               : VVIX level (vol-of-vol, for η prior)
    timestamp          : data observation timestamp
    """
    vix_term_structure: Optional[VIXTermStructureSnapshot] = None
    vix_futures_term: Optional[VIXFuturesTerm] = None
    vix_spot: Optional[float] = None
    spx_surface: Optional[pd.DataFrame] = None
    spx_spot: Optional[float] = None
    risk_free_rate: float = 0.05
    vvix: Optional[float] = None
    timestamp: str = ""

    def summary(self) -> str:
        lines = ["╔══════════════════════════════════════════════════╗"]
        lines.append("║      Calibration Market Data Summary             ║")
        lines.append("╚══════════════════════════════════════════════════╝")
        lines.append(f"  Timestamp: {self.timestamp}")
        lines.append(f"  SPX Spot:  ${self.spx_spot:.2f}" if self.spx_spot else "  SPX Spot:  N/A")
        lines.append(f"  VIX Spot:  {self.vix_spot:.2f}" if self.vix_spot else "  VIX Spot:  N/A")
        lines.append(f"  VVIX:      {self.vvix:.2f}" if self.vvix else "  VVIX:      N/A")
        lines.append(f"  Risk-free: {self.risk_free_rate*100:.3f}% (SOFR)")

        if self.vix_term_structure is not None:
            vts = self.vix_term_structure
            lines.append(f"\n  VIX Term Structure ({vts.date.strftime('%Y-%m-%d')}):")
            for i, label in enumerate(vts.labels):
                lines.append(f"    {label:>6}: {vts.vix_levels[i]:>6.2f}  "
                             f"(σ²={vts.implied_variances[i]:.4f})")

        if self.vix_futures_term is not None:
            vft = self.vix_futures_term
            lines.append(f"\n  VIX Futures ({vft.trade_date.strftime('%Y-%m-%d')}): "
                         f"{vft.n_contracts} contracts")
            for i in range(min(6, vft.n_contracts)):
                lines.append(f"    {vft.days_to_exp[i]:>4}d: {vft.prices[i]:>6.2f}")

        if self.spx_surface is not None:
            surf = self.spx_surface
            n_mats = surf['dte'].nunique() if 'dte' in surf.columns else 0
            lines.append(f"\n  SPX Options Surface: {len(surf)} options, "
                         f"{n_mats} maturities")

        return "\n".join(lines)


def assemble_calibration_data(
    options_cache_dir: str = "data/options_cache",
    volatility_dir: str = "data/market/volatility",
    vix_futures_dir: str = "data/market/cboe_vix_futures_full",
    as_of_date: str | None = None,
    verbose: bool = True,
    enforce_sync: bool = False,
    max_days_skew: int = 2,
) -> CalibrationMarketData:
    """
    Assemble all market data needed for joint SPX-VIX calibration.

    Loads from multiple sources with graceful fallbacks:
    1. SPX options surface (from cache)
    2. VIX term structure (TradingView indices)
    3. VIX futures term structure (CBOE)
    4. VIX spot, VVIX (TradingView)
    5. SOFR risk-free rate

     TEMPORAL ALIGNMENT NOTE:
    Each data source is loaded independently and may come from different dates.
    This is OK for a *calibration snapshot* (Q-measure) as long as no huge market
    moves happen between dates (e.g., volatility spike), but for accuracy:
    
    - Set as_of_date explicitly to ensure reproducibility
    - Pass enforce_sync=True to raise error if dates differ by >max_days_skew
    - Check the assembled data's `timestamp` field to verify alignment

    Parameters
    ----------
    options_cache_dir : path to SPY options cache
    volatility_dir : path to TradingView volatility data
    vix_futures_dir : path to CBOE VIX futures
    as_of_date : target date (None = latest; recommend: explicit YYYY-MM-DD)
    verbose : print progress
    enforce_sync : if True, raise error if source dates differ by >max_days_skew
    max_days_skew : tolerance in days for temporal alignment (default: 2)

    Returns
    -------
    CalibrationMarketData
        Note: Check .timestamp and individual source dates for alignment audit.
    """
    import pandas as pd
    from utils.fetcher.options_cache import OptionsDataCache

    data = CalibrationMarketData()
    
    # Track timestamps for alignment check
    source_timestamps = {}

    # 1. SPX options surface
    try:
        cache = OptionsDataCache(options_cache_dir)
        surface_df, meta = cache.load_latest("SPX")
        
        # CLEAN: Remove illiquid/bad data
        n_before = len(surface_df)
        
        # Remove rows with bid = 0 (missing data)
        surface_df = surface_df[surface_df['bid'] > 0].copy()
        n_zero_bid = n_before - len(surface_df)
        
        # Remove rows with bid-ask spread > 50% (extremely illiquid)
        surface_df['spread_pct'] = (surface_df['ask'] - surface_df['bid']) / surface_df['mid'].clip(lower=0.001)
        wide_spread = (surface_df['spread_pct'] > 0.50).sum()
        surface_df = surface_df[surface_df['spread_pct'] <= 0.50].copy()
        
        # Drop the helper column
        surface_df = surface_df.drop(columns=['spread_pct'])
        
        data.spx_surface = surface_df
        data.spx_spot = meta['spot']
        data.timestamp = meta.get('datetime', '')
        source_timestamps['SPY_options'] = pd.to_datetime(data.timestamp)
        if verbose:
            n_mats = surface_df['dte'].nunique()
            print(f"  SPX surface: {len(surface_df)} options (removed {n_zero_bid} zero-bid + {wide_spread} wide-spread), "
                  f"{n_mats} maturities (spot=${meta['spot']:.2f})  {data.timestamp}")
    except Exception as e:
        if verbose:
            print(f"  SPX surface error: {e}")

    # 2. VIX futures (CBOE) — PRIMARY SOURCE for term structure (always up-to-date)
    try:
        vix_fut = load_vix_futures(vix_futures_dir)
        if as_of_date:
            term = vix_fut.get_term_structure(as_of_date, vix_spot=data.vix_spot)
        else:
            term = vix_fut.get_latest_term_structure(vix_spot=data.vix_spot)
        data.vix_futures_term = term
        source_timestamps['VIX_futures'] = term.trade_date
        
        # Build VIX term structure from futures prices (FALLBACK if TradingView stale)
        try:
            vts = get_vix_term_snapshot_from_futures(date=as_of_date, data_dir=vix_futures_dir)
            data.vix_term_structure = vts
            data.vix_spot = float(vts.vix_levels[0]) if len(vts.vix_levels) > 0 else None
            source_timestamps['VIX_indices'] = vts.date
            if verbose:
                print(f"  VIX term structure (from futures): {vts.n_tenors} contracts "
                      f"({vts.date.strftime('%Y-%m-%d')})")
                for i, label in enumerate(vts.labels):
                    print(f"      {label:>6}: {vts.vix_levels[i]:.2f}")
        except Exception as e:
            # Try TradingView fallback
            try:
                vts = get_vix_term_snapshot(date=as_of_date, data_dir=volatility_dir)
                data.vix_term_structure = vts
                data.vix_spot = float(vts.vix_levels[vts.labels.index('vix')]) if 'vix' in vts.labels else None
                source_timestamps['VIX_indices'] = vts.date
                if verbose:
                    print(f"  VIX term structure (TradingView): {vts.n_tenors} tenors "
                          f"({vts.date.strftime('%Y-%m-%d')})")
                    for i, label in enumerate(vts.labels):
                        print(f"      {label:>6}: {vts.vix_levels[i]:.2f}")
            except Exception as e2:
                if verbose:
                    print(f"  VIX term structure error (futures): {e}, (TradingView): {e2}")
    except Exception as e:
        if verbose:
            print(f"  VIX futures error: {e}")

    # 4. VVIX
    try:
        vvix_path = Path(volatility_dir) / "vvix_daily.csv"
        if vvix_path.exists():
            vvix_df = _load_tradingview_csv(vvix_path)
            if as_of_date:
                target = pd.to_datetime(as_of_date).normalize()
                avail = vvix_df[vvix_df.index <= target]
                data.vvix = float(avail['close'].iloc[-1]) if len(avail) > 0 else None
            else:
                data.vvix = float(vvix_df['close'].iloc[-1])
            if verbose and data.vvix is not None:
                print(f"  VVIX: {data.vvix:.2f}")
    except Exception as e:
        if verbose:
            print(f"  VVIX error: {e}")

    # 5. SOFR
    try:
        from quant.loader.sofr_loader import SOFRRateLoader
        sofr = SOFRRateLoader()
        if sofr.is_available:
            data.risk_free_rate = sofr.get_rate()
            if verbose:
                print(f"  SOFR: {data.risk_free_rate*100:.3f}%")
    except Exception as e:
        if verbose:
            print(f"  SOFR error (using default {data.risk_free_rate*100:.1f}%): {e}")

    # ── Temporal Alignment Check ──────────────────────────────────────────────
    if source_timestamps and verbose:
        print(f"\n TEMPORAL ALIGNMENT AUDIT")
        print(f"  {'-' * 50}")
        
        # Normalize all timestamps (remove tz info, time component)
        normalized_ts = {}
        for source, ts in sorted(source_timestamps.items()):
            if isinstance(ts, str):
                ts = pd.to_datetime(ts)
            else:
                ts = pd.to_datetime(ts)
            # Remove timezone and time of day
            ts_norm = ts.tz_localize(None).normalize() if ts.tz is not None else ts.normalize()
            normalized_ts[source] = ts_norm
            print(f"     {source:20} : {ts_norm.strftime('%Y-%m-%d')}")
        
        # Check skew
        if len(normalized_ts) > 1:
            dates = list(normalized_ts.values())
            date_min = min(dates)
            date_max = max(dates)
            skew_days = (date_max - date_min).days
            
            if skew_days > max_days_skew:
                msg = (
                    f"\n   ALERT: Data sources differ by {skew_days} days (max tolerance: {max_days_skew})\n"
                    f"     This may bias calibration if significant market moves occurred.\n"
                    f"     Recommend: Pass explicit as_of_date (e.g., '2026-03-12') for reproducibility."
                )
                print(msg)
                
                if enforce_sync:
                    raise ValueError(
                        f"Temporal alignment failed: {skew_days}d skew > {max_days_skew}d threshold. "
                        f"Pass enforce_sync=False or use explicit as_of_date."
                    )
            else:
                print(f"  [OK] All sources within {skew_days}d (acceptable)")
        print()

    return data
