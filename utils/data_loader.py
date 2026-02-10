import pandas as pd
import numpy as np
import jax.numpy as jnp
import yaml
import os

def load_config(config_path: str = "config/params.yaml") -> dict:
    """Loads configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class RealizedVolatilityLoader:
    """
    Computes Realized Volatility from S&P 500 intraday price data.
    This captures the TRUE rough behavior (H ~ 0.1) of market volatility.
    
    Theory: RV_t = sqrt(sum(r_i^2)) where r_i are intraday returns
    """
    def __init__(self, file_path: str = "data/SP_SPX, 30.csv", config_path: str = "config/params.yaml"):
        self.config = load_config(config_path)
        self.file_path = file_path
        self.max_gap_hours = self.config['data'].get('max_gap_hours', 4)
        self.stride_ratio = self.config['data'].get('stride_ratio', 0.5)
        
    def _check_temporal_coherence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates temporal coherence and marks segment boundaries."""
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df = df.sort_values('datetime').reset_index(drop=True)
        
        df['time_delta'] = df['datetime'].diff()
        max_acceptable_gap = pd.Timedelta(hours=self.max_gap_hours)
        
        gaps = df[df['time_delta'] > max_acceptable_gap]
        if len(gaps) > 0:
            print(f"Warning: Found {len(gaps)} gaps > {self.max_gap_hours}h")
            for _, row in gaps.nlargest(3, 'time_delta').iterrows():
                print(f"   Gap: {row['time_delta']} at {row['datetime']}")
        
        df['segment_id'] = (df['time_delta'] > max_acceptable_gap).cumsum()
        n_segments = df['segment_id'].nunique()
        print(f"   Data split into {n_segments} continuous segments")
        
        return df
    
    def get_realized_vol_paths(self, segment_length: int = None, rv_window: int = 13):
        """
        Computes rolling Realized Volatility from S&P 500 returns.
        
        Args:
            segment_length: Length of variance paths for training
            rv_window: Window for RV calculation (13 = ~1 trading day for 30-min data)
        
        Returns:
            JAX array of realized variance paths
        """
        if segment_length is None:
            segment_length = self.config['data']['segment_length']
            
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
            # Annualized: multiply by (252 * 13) for 30-min bars
            seg_data['realized_var'] = seg_data['log_return'].rolling(window=rv_window).apply(
                lambda x: np.sum(x**2) * 252 * 13, raw=True
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
        Respects temporal boundaries to avoid overnight jumps in training data.
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

        # 5. Segment into training paths (respecting temporal boundaries)
        paths = []
        stride = max(1, int(segment_length * self.stride_ratio))
        
        for seg_id in df['segment_id'].unique():
            segment_data = df[df['segment_id'] == seg_id]['variance'].values
            
            # Only process segments long enough
            if len(segment_data) < segment_length:
                continue
                
            for i in range(0, len(segment_data) - segment_length, stride):
                path = segment_data[i : i + segment_length]
                paths.append(path)
        
        if len(paths) == 0:
            raise ValueError(f"No valid paths created. Try reducing segment_length (current: {segment_length})")
            
        paths = np.array(paths)
        
        # Shuffle paths to break time correlation during training
        np.random.shuffle(paths)
        
        print(f"Dataset ready: {paths.shape[0]} paths of length {segment_length}")
        return jnp.array(paths)