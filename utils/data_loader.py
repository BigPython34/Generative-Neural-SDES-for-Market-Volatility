import os
import yfinance as yf
import numpy as np
import pandas as pd
import jax.numpy as jnp

class MarketDataLoader:
    """
    Fetches real market data with robust CSV parsing.
    Handles yfinance multi-header format artifacts.
    """
    def __init__(self, ticker="^GSPC", start_date="2010-01-01", end_date="2023-01-01"):
        self.ticker = ticker
        self.start = start_date
        self.end = end_date
        self.cache_file = f"data_{ticker}_{start_date}_{end_date}.csv"

    def get_realized_vol_paths(self, window_size: int = 20, segment_length: int = 100):
        # 1. Load Data
        if os.path.exists(self.cache_file):
            print(f"Loading data from cache: {self.cache_file}...")
            # Read all as string first to avoid type inference errors on headers
            df = pd.read_csv(self.cache_file, index_col=0, low_memory=False)
        else:
            print(f"Downloading {self.ticker} from Yahoo Finance...")
            df = yf.download(self.ticker, start=self.start, end=self.end, progress=False)
            df.to_csv(self.cache_file)
        
        # 2. Extract Price Series
        # yfinance CSVs often have 3 header rows, confusing pandas.
        # We try standard column names, then fallback to index location.
        try:
            if 'Close' in df.columns:
                prices = df['Close']
            elif 'Adj Close' in df.columns:
                prices = df['Adj Close']
            else:
                prices = df.iloc[:, 0] # Fallback
        except:
            prices = df.iloc[:, 0]

        # Ensure we have a 1D Series, not a DataFrame (common cause of TypeError)
        if isinstance(prices, pd.DataFrame):
            # If multiple columns found, take the first one
            prices = prices.iloc[:, 0]
            
        # 3. Clean and Typecast
        # Force numeric conversion. Non-numeric headers/metadata become NaN.
        prices = pd.to_numeric(prices, errors='coerce')
        prices = prices.dropna()

        print(f"Loaded {len(prices)} valid price points.")

        if len(prices) < window_size + segment_length:
             raise ValueError("Insufficient data points after cleaning.")

        # 4. Compute Realized Volatility
        # Log Returns
        log_ret = np.log(prices / prices.shift(1))
        
        # Rolling Volatility (Annualized)
        realized_vol = log_ret.rolling(window=window_size).std() * np.sqrt(252)
        
        # Variance = Vol^2
        variance_data = realized_vol.pow(2).replace([np.inf, -np.inf], np.nan).dropna().values
        
        # 5. Segment into training paths
        paths = []
        stride = 5 
        for i in range(0, len(variance_data) - segment_length, stride):
            path = variance_data[i : i + segment_length]
            paths.append(path)
            
        paths = np.array(paths)
        
        # Clip extreme outliers (e.g. 2020 crash) for training stability
        if len(paths) > 0:
            cap = np.percentile(paths, 99)
            paths = np.clip(paths, 0, cap)
        
        print(f"Dataset ready: {paths.shape[0]} paths of length {segment_length}.")
        return jnp.array(paths)