"""
Options Data Cache Manager
==========================
Stores and manages historical options data locally.
Avoids repeated API calls to Yahoo Finance.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import json

class OptionsDataCache:
    """
    Caches options data locally for faster access and historical analysis.
    """
    
    def __init__(self, cache_dir: str = "data/options_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.metadata_file = os.path.join(cache_dir, "metadata.json")
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> dict:
        """Load cache metadata."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"snapshots": []}
    
    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def save_surface(self, surface_df: pd.DataFrame, ticker: str, spot: float):
        """
        Save a volatility surface snapshot.
        
        Args:
            surface_df: DataFrame with options data
            ticker: Ticker symbol
            spot: Spot price at snapshot time
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ticker}_surface_{timestamp}.csv"
        filepath = os.path.join(self.cache_dir, filename)
        
        # Save data as CSV (more portable)
        surface_df.to_csv(filepath, index=False)
        
        # Update metadata
        snapshot_info = {
            "filename": filename,
            "ticker": ticker,
            "spot": spot,
            "timestamp": timestamp,
            "datetime": datetime.now().isoformat(),
            "n_options": len(surface_df),
            "maturities": sorted(surface_df['dte'].unique().tolist()),
            "strike_range": [float(surface_df['strike'].min()), float(surface_df['strike'].max())]
        }
        
        self.metadata["snapshots"].append(snapshot_info)
        self._save_metadata()
        
        print(f"Saved surface snapshot: {filename}")
        print(f"  {len(surface_df)} options, {len(snapshot_info['maturities'])} maturities")
        
        return filepath
    
    def load_latest(self, ticker: str = "SPY") -> tuple:
        """Load the most recent surface for a ticker."""
        ticker_snapshots = [s for s in self.metadata["snapshots"] if s["ticker"] == ticker]
        
        if not ticker_snapshots:
            raise FileNotFoundError(f"No cached data for {ticker}")
        
        latest = max(ticker_snapshots, key=lambda x: x["timestamp"])
        filepath = os.path.join(self.cache_dir, latest["filename"])
        
        df = pd.read_csv(filepath)
        
        print(f"Loaded cached surface from {latest['datetime']}")
        print(f"  Spot at snapshot: ${latest['spot']:.2f}")
        
        return df, latest
    
    def load_by_date(self, ticker: str, date: str) -> tuple:
        """Load surface closest to a specific date."""
        ticker_snapshots = [s for s in self.metadata["snapshots"] if s["ticker"] == ticker]
        
        target = datetime.strptime(date, "%Y-%m-%d")
        closest = min(ticker_snapshots, key=lambda x: abs(
            datetime.fromisoformat(x["datetime"]) - target
        ))
        
        filepath = os.path.join(self.cache_dir, closest["filename"])
        df = pd.read_csv(filepath)
        
        return df, closest
    
    def list_snapshots(self, ticker: str = None) -> pd.DataFrame:
        """List all cached snapshots."""
        snapshots = self.metadata["snapshots"]
        
        if ticker:
            snapshots = [s for s in snapshots if s["ticker"] == ticker]
        
        if not snapshots:
            print("No snapshots in cache.")
            return pd.DataFrame()
        
        df = pd.DataFrame(snapshots)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df.sort_values('datetime', ascending=False)


class EnhancedOptionsLoader:
    """
    Enhanced options data loader with more maturities, wider strikes, and caching.
    """
    
    def __init__(self, ticker: str = "SPY", cache: OptionsDataCache = None):
        self.ticker = ticker
        self.cache = cache or OptionsDataCache()
        self.spot = None
        self.tick = yf.Ticker(ticker)
        
    def get_spot_price(self) -> float:
        """Get current spot price."""
        self.spot = self.tick.history(period="1d")['Close'].iloc[-1]
        return self.spot
    
    def get_all_expirations(self) -> list:
        """Get all available expiration dates."""
        return list(self.tick.options)
    
    def get_full_surface(self, max_dte: int = 180, min_volume: int = 1,
                        moneyness_range: float = 0.30, save_cache: bool = True) -> pd.DataFrame:
        """
        Load comprehensive volatility surface with many maturities.
        
        Args:
            max_dte: Maximum days to expiry to include
            min_volume: Minimum volume filter
            moneyness_range: Maximum |log(K/S)| to include (0.30 = ±30%)
            save_cache: Whether to save to local cache
        """
        self.spot = self.get_spot_price()
        expirations = self.get_all_expirations()
        
        print(f"{self.ticker} Spot: ${self.spot:.2f}")
        print(f"Loading {len(expirations)} available expirations (max {max_dte} DTE)...")
        
        all_data = []
        loaded_dtes = []
        
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            dte = (exp_date - datetime.now()).days
            
            if dte <= 0 or dte > max_dte:
                continue
            
            try:
                chain = self.tick.option_chain(exp_str)
                
                # Process calls
                calls = chain.calls[['strike', 'lastPrice', 'bid', 'ask', 
                                    'impliedVolatility', 'volume', 'openInterest']].copy()
                calls['type'] = 'call'
                
                # Process puts
                puts = chain.puts[['strike', 'lastPrice', 'bid', 'ask',
                                  'impliedVolatility', 'volume', 'openInterest']].copy()
                puts['type'] = 'put'
                
                # Combine
                df = pd.concat([calls, puts], ignore_index=True)
                df['expiration'] = exp_str
                df['dte'] = dte
                df['T'] = dte / 365.0
                df['moneyness'] = np.log(df['strike'] / self.spot)
                df['mid'] = (df['bid'] + df['ask']) / 2
                
                # Filter
                df = df[
                    (df['volume'] >= min_volume) &
                    (df['bid'] > 0) &
                    (df['impliedVolatility'] > 0.01) &
                    (df['impliedVolatility'] < 3.0) &
                    (abs(df['moneyness']) <= moneyness_range)
                ]
                
                if len(df) > 0:
                    all_data.append(df)
                    loaded_dtes.append(dte)
                    
            except Exception as e:
                continue
        
        if not all_data:
            raise ValueError("No options data loaded!")
        
        surface = pd.concat(all_data, ignore_index=True)
        
        # Use OTM options for cleaner IV
        otm_mask = ((surface['type'] == 'call') & (surface['strike'] >= self.spot)) | \
                   ((surface['type'] == 'put') & (surface['strike'] <= self.spot))
        surface_otm = surface[otm_mask].copy()
        
        # Remove duplicates (same strike/dte)
        surface_otm = surface_otm.sort_values('volume', ascending=False)\
                                  .drop_duplicates(['strike', 'dte'])\
                                  .sort_values(['dte', 'moneyness'])
        
        print(f"Surface loaded: {len(surface_otm)} OTM options")
        print(f"  DTEs: {sorted(surface_otm['dte'].unique().tolist())}")
        print(f"  Strikes: ${surface_otm['strike'].min():.0f} - ${surface_otm['strike'].max():.0f}")
        
        # Save to cache
        if save_cache:
            self.cache.save_surface(surface_otm, self.ticker, self.spot)
        
        return surface_otm
    
    def get_smile(self, dte: int, surface: pd.DataFrame = None) -> pd.DataFrame:
        """Extract smile for a specific maturity."""
        if surface is None:
            surface = self.get_full_surface()
        
        # Find closest DTE
        available_dtes = surface['dte'].unique()
        closest_dte = min(available_dtes, key=lambda x: abs(x - dte))
        
        smile = surface[surface['dte'] == closest_dte].sort_values('moneyness')
        return smile


def download_and_cache_options():
    """Utility function to download and cache options data."""
    
    print("="*70)
    print("   OPTIONS DATA DOWNLOADER & CACHE")
    print("="*70)
    
    cache = OptionsDataCache()
    loader = EnhancedOptionsLoader("SPY", cache)
    
    # Load comprehensive surface
    surface = loader.get_full_surface(
        max_dte=180,        # Up to 6 months
        min_volume=1,       # Include less liquid options
        moneyness_range=0.25,  # ±25% strikes
        save_cache=True
    )
    
    # Show summary
    print("\nSurface Summary:")
    print(surface.groupby('dte').agg({
        'strike': ['min', 'max', 'count'],
        'impliedVolatility': ['mean', 'std']
    }).round(3))
    
    # List all cached snapshots
    print("\nCached Snapshots:")
    print(cache.list_snapshots())
    
    return surface


if __name__ == "__main__":
    download_and_cache_options()
