import pandas as pd
import numpy as np
import jax.numpy as jnp
import os

from utils.config import load_config


class RealizedVolatilityLoader:
    """
    Computes Realized Volatility from S&P 500 intraday price data.
    This captures the TRUE rough behavior (H ~ 0.1) of market volatility.
    
    Theory: RV_t = sqrt(sum(r_i^2)) where r_i are intraday returns
    
    Key distinction:
    - VIX ≈ 30-day integrated implied vol → H ≈ 0.5 (smooth, NOT rough)
    - Realized Vol from 5-min SPX returns → H ≈ 0.05-0.14 (rough!)
    - Gatheral, Jaisson, Rosenbaum (2018): "Volatility is Rough"
    """
    def __init__(self, file_path: str = None, config_path: str = "config/params.yaml",
                 bar_interval_min: int = None):
        self.config = load_config(config_path)
        self.file_path = file_path or self.config['data'].get('rv_source', 'data/SP_SPX, 5.csv')
        self.max_gap_hours = self.config['data'].get('max_gap_hours', 4)
        self.stride_ratio = self.config['data'].get('stride_ratio', 0.5)
        self.trading_hours = self.config['data'].get('trading_hours_per_day', 6.5)
        
        # Auto-detect bar frequency from data or use explicit parameter
        if bar_interval_min is not None:
            self._bar_min = bar_interval_min
        else:
            self._bar_min = self._detect_bar_interval()
        self.bars_per_day = int(self.trading_hours * 60 / self._bar_min)
        print(f"   RV Loader: {self._bar_min}-min bars, {self.bars_per_day} bars/day")
    
    def _detect_bar_interval(self) -> int:
        """Auto-detect bar interval from data timestamps."""
        if not os.path.exists(self.file_path):
            # Fallback to config
            return self.config['data'].get('rv_bar_interval_min', 5)
        try:
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
        except Exception:
            return self.config['data'].get('rv_bar_interval_min', 5)
        
    def _check_temporal_coherence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates temporal coherence and marks segment boundaries."""
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df = df.sort_values('datetime').reset_index(drop=True)
        
        df['time_delta'] = df['datetime'].diff()
        max_acceptable_gap = pd.Timedelta(hours=self.max_gap_hours)
        
        gaps = df[df['time_delta'] > max_acceptable_gap]
        if len(gaps) > 0:
            print(f"   Warning: Found {len(gaps)} gaps > {self.max_gap_hours}h")
            for _, row in gaps.nlargest(3, 'time_delta').iterrows():
                print(f"      Gap: {row['time_delta']} at {row['datetime']}")
        
        df['segment_id'] = (df['time_delta'] > max_acceptable_gap).cumsum()
        n_segments = df['segment_id'].nunique()
        print(f"   Data split into {n_segments} continuous segments")
        
        return df
    
    def get_realized_vol_paths(self, segment_length: int = None, rv_window: int = None):
        """
        Computes rolling Realized Volatility from S&P 500 returns.
        
        Args:
            segment_length: Length of variance paths for training
            rv_window: Window for RV calculation
                       Default from config: 78 (= 1 trading day for 5-min bars)
                       The annualization factor is computed automatically.
        
        Returns:
            JAX array of realized variance paths
        """
        if segment_length is None:
            segment_length = self.config['data']['segment_length']
        if rv_window is None:
            # Config rv_window is calibrated for rv_bar_interval_min (default 5min)
            # If this file's bar interval differs, scale proportionally
            config_rv_window = self.config['data'].get('rv_window', self.bars_per_day)
            config_bar_min = self.config['data'].get('rv_bar_interval_min', 5)
            if self._bar_min != config_bar_min:
                # Scale: rv_window of 78 at 5-min ≈ 13 at 30-min (same real time span)
                rv_window = max(6, int(config_rv_window * config_bar_min / self._bar_min))
                print(f"   Adjusted rv_window: {config_rv_window} ({config_bar_min}min) → {rv_window} ({self._bar_min}min)")
            else:
                rv_window = config_rv_window
            
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Please place S&P 500 data at: {self.file_path}")
            
        print(f"Loading S&P 500 intraday data from {self.file_path}...")
        
        # 1. Load and preprocess
        df = pd.read_csv(self.file_path)
        df = self._check_temporal_coherence(df)
        
        # 2. Compute log-returns
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df = df.dropna(subset=['log_return'])
        
        # 3. Compute Realized Variance per segment (avoid crossing gaps)
        all_rv = []
        
        for seg_id in df['segment_id'].unique():
            seg_data = df[df['segment_id'] == seg_id].copy()
            
            if len(seg_data) < rv_window + 1:
                continue
            
            # Rolling Realized Variance: RV = sum(r^2) over window
            # Annualization: each squared return r_i^2 has E[r_i^2] = sigma^2 * dt_bar
            # where dt_bar = 1 / (252 * bars_per_day) in years.
            # sum_{i in window} r_i^2 ≈ sigma^2 * rv_window * dt_bar
            # To annualize: multiply by (252 * bars_per_day / rv_window)
            # Reference: Andersen & Bollerslev (1998), Barndorff-Nielsen & Shephard (2002)
            annualize_factor = 252 * self.bars_per_day / rv_window
            seg_data['realized_var'] = seg_data['log_return'].rolling(window=rv_window).apply(
                lambda x: np.sum(x**2) * annualize_factor, raw=True
            )
            
            rv_values = seg_data['realized_var'].dropna().values
            all_rv.extend(rv_values)
        
        realized_var = np.array(all_rv)
        
        # 4. Filter outliers (RV should be positive and reasonable)
        # Typical annual variance: 0.01 (10% vol) to 1.0 (100% vol)
        realized_var = realized_var[(realized_var > 0.001) & (realized_var < 2.0)]
        
        print(f"Computed {len(realized_var)} realized variance points")
        print(f"   Mean RV: {np.mean(realized_var):.4f} (Vol: {np.sqrt(np.mean(realized_var))*100:.1f}%)")
        
        # 5. Segment into training paths
        paths = []
        stride = max(1, int(segment_length * self.stride_ratio))
        
        for i in range(0, len(realized_var) - segment_length, stride):
            path = realized_var[i : i + segment_length]
            paths.append(path)
        
        if len(paths) == 0:
            raise ValueError(f"No valid paths. Try reducing segment_length (current: {segment_length})")
            
        paths = np.array(paths)
        np.random.shuffle(paths)
        
        print(f"Dataset ready: {paths.shape[0]} paths of length {segment_length}")
        return jnp.array(paths)


class MarketDataLoader:
    """
    Loads custom Intraday VIX data for Rough Volatility modeling.
    Target: 30-min VIX Variance.
    """
    def __init__(self, file_path: str = None, config_path: str = "config/params.yaml"):
        self.config = load_config(config_path)
        self.file_path = file_path or self.config['data']['source']
        self.max_gap_hours = self.config['data'].get('max_gap_hours', 4)
        self.stride_ratio = self.config['data'].get('stride_ratio', 0.5)

    def _check_temporal_coherence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates temporal coherence of intraday data:
        - Detects and reports gaps (weekends, holidays, missing data)
        - Filters out overnight jumps that could distort the model
        """
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Compute time deltas between consecutive points
        df['time_delta'] = df['datetime'].diff()
        
        # Expected gap for 30-min data
        expected_gap = pd.Timedelta(minutes=30)
        max_acceptable_gap = pd.Timedelta(hours=self.max_gap_hours)
        
        # Identify gaps
        gaps = df[df['time_delta'] > max_acceptable_gap]
        
        if len(gaps) > 0:
            print(f"Warning: Found {len(gaps)} gaps > {self.max_gap_hours}h")
            largest_gaps = gaps.nlargest(5, 'time_delta')
            for _, row in largest_gaps.iterrows():
                print(f"   Gap: {row['time_delta']} at {row['datetime']}")
        else:
            print("Temporal Coherence: No significant gaps detected")
        
        # Mark segment boundaries (where gaps occur)
        # This prevents paths from crossing overnight/weekend boundaries
        df['segment_id'] = (df['time_delta'] > max_acceptable_gap).cumsum()
        
        # Statistics
        n_segments = df['segment_id'].nunique()
        avg_segment_len = len(df) / n_segments
        print(f"   Data split into {n_segments} continuous segments (avg length: {avg_segment_len:.0f} points)")
        
        return df

    def get_realized_vol_paths(self, segment_length: int = None):
        """
        Reads 'time,open,high,low,close' CSV and converts VIX close to Variance paths.
        
        For short segment_length (< max_segment), respects day boundaries.
        For longer paths, bridges across overnight gaps since VIX variance
        is persistent (VIX opens near previous close).
        """
        if segment_length is None:
            segment_length = self.config['data']['segment_length']
            
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Please place your data file at: {self.file_path}")

        print(f"Loading Intraday VIX data from {self.file_path}...")
        
        # 1. Parse CSV
        df = pd.read_csv(self.file_path)
        
        # 2. Temporal Coherence Check
        df = self._check_temporal_coherence(df)
        
        # 3. Compute Variance Proxy
        # VIX is a volatility index in percentage points (e.g., 20.0 means 20%).
        # We convert it to decimal variance: (Close / 100)^2
        df['variance'] = (pd.to_numeric(df['close'], errors='coerce') / 100.0) ** 2
        
        # 4. Filter Data
        # Remove aberrations (VIX < 1.0 or VIX > 200.0 is likely an error)
        df = df[(df['variance'] > 1e-4) & (df['variance'] < 4.0)]
        df = df.dropna(subset=['variance'])

        print(f"Loaded {len(df)} valid intraday points.")
        
        # 5. Check if segments are long enough for requested path length
        seg_lengths = df.groupby('segment_id').size()
        max_seg_len = int(seg_lengths.max()) if len(seg_lengths) > 0 else 0
        
        if max_seg_len >= segment_length:
            # Standard mode: segment within day boundaries
            print(f"   Using intra-segment paths (max segment: {max_seg_len} >= {segment_length})")
            paths = self._segment_within_boundaries(df, segment_length)
        else:
            # Bridge mode: concatenate variance across day boundaries
            # VIX variance is persistent → safe to bridge overnight gaps
            print(f"   Max segment ({max_seg_len}) < path length ({segment_length})")
            print(f"   Bridging across overnight gaps (VIX variance is persistent)")
            paths = self._segment_bridged(df, segment_length)

        if len(paths) == 0:
            raise ValueError(f"No valid paths created. Try reducing segment_length (current: {segment_length})")
            
        paths = np.array(paths)
        
        # Shuffle paths to break time correlation during training
        np.random.shuffle(paths)
        
        print(f"Dataset ready: {paths.shape[0]} paths of length {segment_length}")
        return jnp.array(paths)
    
    def _segment_within_boundaries(self, df, segment_length):
        """Extract paths strictly within segment boundaries."""
        paths = []
        stride = max(1, int(segment_length * self.stride_ratio))
        
        for seg_id in df['segment_id'].unique():
            segment_data = df[df['segment_id'] == seg_id]['variance'].values
            if len(segment_data) < segment_length:
                continue
            for i in range(0, len(segment_data) - segment_length, stride):
                paths.append(segment_data[i : i + segment_length])
        
        return paths
    
    def _segment_bridged(self, df, segment_length):
        """
        Extract paths that bridge across overnight/weekend gaps.
        
        For VIX variance, this is valid because:
          - VIX opens near previous close (no gap in variance level)
          - We're modeling the variance distribution, not return dynamics
          - The signature captures path shape, which is continuous in VIX level
          
        We filter out paths where the overnight jump in variance exceeds
        a threshold (e.g. > 50% relative change) to exclude crisis events
        that would distort training.
        """
        # Concatenate all variance values in time order
        variance_series = df.sort_values('datetime')['variance'].values
        
        # Also track where the gaps are (to check for large jumps)
        gap_mask = df.sort_values('datetime')['time_delta'].dt.total_seconds() > self.max_gap_hours * 3600
        gap_indices = np.where(gap_mask.values)[0]
        
        # Check overnight jumps: relative change at gap points
        max_relative_jump = 0.5  # Filter out >50% overnight VIX jumps
        bad_indices = set()
        for gi in gap_indices:
            if gi > 0 and gi < len(variance_series):
                rel_change = abs(variance_series[gi] - variance_series[gi-1]) / (variance_series[gi-1] + 1e-8)
                if rel_change > max_relative_jump:
                    # Mark a window around this jump as "bad"
                    for offset in range(-2, 3):
                        bad_indices.add(gi + offset)
        
        paths = []
        stride = max(1, int(segment_length * self.stride_ratio))
        
        for i in range(0, len(variance_series) - segment_length, stride):
            # Check if this path window crosses a bad jump
            path_range = set(range(i, i + segment_length))
            if path_range & bad_indices:
                continue
            
            path = variance_series[i : i + segment_length]
            
            # Additional sanity: skip paths with extreme variance ratios
            if np.max(path) / (np.min(path) + 1e-8) > 20:
                continue
            
            paths.append(path)
        
        return paths