import jax.numpy as jnp
import numpy as np
from scipy.stats import kurtosis

def compute_acf(x, lags=20):
    """
    Computes Auto-Correlation Function for a 1D array.
    Used to check memory/persistence of volatility.
    """
    n = len(x)
    mean = np.mean(x)
    var = np.var(x)
    xp = x - mean
    corr = np.correlate(xp, xp, mode='full')[n-1:] / (var * n)
    return corr[:lags]

def print_distribution_stats(name: str, data: np.ndarray):
    """
    Prints vital statistics including high-order moments (Kurtosis).
    """
    flat_data = data.flatten()
    
    # Calculate ACF on the first path as a sample
    acf_sample = compute_acf(data[0], lags=5)
    
    print(f"--- DIAGNOSTIC: {name} ---")
    print(f"   Mean           : {np.mean(flat_data):.5f}")
    print(f"   Median         : {np.median(flat_data):.5f}")
    print(f"   Max            : {np.max(flat_data):.5f}")
    print(f"   Kurtosis       : {kurtosis(flat_data):.5f} (Target > 3.0 for fat tails)")
    print(f"   ACF (lag 1-5)  : {np.round(acf_sample[1:], 3)}")
    print("-" * 30)