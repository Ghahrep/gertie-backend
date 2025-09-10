# tools/fractal_tools.py - Clean Fractal Analysis Tools
"""
Fractal Analysis Tools - Clean Architecture
==========================================

Advanced mathematical tools for fractal analysis of financial time series.
Includes Hurst exponent, DFA, multifractal spectrum, and fBm simulation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Any
from scipy import stats
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning, message="The values in the array are all the same")

# Optional dependencies with graceful fallbacks
try:
    from fbm import FBM
    HAS_FBM = True
except ImportError:
    HAS_FBM = False
    print("Warning: fbm not installed. fBm simulation unavailable. `pip install fbm`")

def calculate_hurst(
    series: pd.Series,
    min_window: int = 10,
    max_window: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calculate the Hurst exponent using Rescaled Range (R/S) analysis.

    Parameters:
    -----------
    series : pd.Series
        Time series data (e.g., price returns, log prices).
    min_window : int, default=10
        Minimum window size for R/S calculation.
    max_window : int or None, default=None
        Maximum window size. If None, uses len(series)//2.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the Hurst exponent, its interpretation, and metadata.
    """
    series_clean = series.dropna()
    n = len(series_clean)

    if n < 50:
        raise ValueError(f"Series too short for reliable Hurst calculation (has {n}, needs 50)")

    if max_window is None:
        max_window = n // 2

    # Ensure min_window is less than max_window and max_window is feasible
    min_window = max(2, min_window)
    max_window = min(n - 1, max_window)
    if min_window >= max_window:
        raise ValueError("min_window must be smaller than max_window.")

    windows = np.unique(np.logspace(np.log10(min_window), np.log10(max_window), 20).astype(int))
    
    rs_values = []
    valid_windows = []

    for window in windows:
        # Check if window size is valid
        if window > n:
            continue

        # Vectorized R/S calculation for better performance
        sub_series_matrix = np.lib.stride_tricks.as_strided(
            series_clean.values,
            shape=(n - window + 1, window),
            strides=(series_clean.values.strides[0], series_clean.values.strides[0])
        )
        
        mean_centered = sub_series_matrix - sub_series_matrix.mean(axis=1, keepdims=True)
        cum_dev = mean_centered.cumsum(axis=1)
        R = cum_dev.max(axis=1) - cum_dev.min(axis=1)
        S = sub_series_matrix.std(axis=1)

        # Filter out sub-series with zero standard deviation
        valid_mask = S > 1e-10
        if np.any(valid_mask):
            rs_window = (R[valid_mask] / S[valid_mask])
            rs_values.append(rs_window.mean())
            valid_windows.append(window)

    if len(rs_values) < 5:
        raise ValueError(f"Insufficient valid windows ({len(rs_values)}) for Hurst regression.")

    # Use log-log regression to find the slope (Hurst exponent)
    log_rs = np.log10(rs_values)
    log_windows = np.log10(valid_windows)
    
    slope, intercept, r_value, _, _ = stats.linregress(log_windows, log_rs)
    
    hurst_exponent = np.clip(slope, 0.01, 0.99)

    # Interpretation based on Hurst value
    if hurst_exponent < 0.45:
        interpretation = "Mean-Reverting"
    elif hurst_exponent > 0.55:
        interpretation = "Trending"
    else:
        interpretation = "Random Walk"

    return {
        "hurst_exponent": hurst_exponent,
        "interpretation": interpretation,
        "r_squared": r_value**2,
        "num_windows": len(valid_windows),
        "log_log_slope": slope,
        "log_log_intercept": intercept
    }


def calculate_dfa(
    series: pd.Series,
    min_box_size: int = 10,
    max_box_size: Optional[int] = None,
    order: int = 1
) -> Dict[str, Any]:
    """
    Calculate the scaling exponent using Detrended Fluctuation Analysis (DFA).

    Parameters:
    -----------
    series : pd.Series
        Time series data.
    min_box_size : int, default=10
        Minimum box size for DFA calculation.
    max_box_size : int or None, default=None
        Maximum box size. If None, uses len(series)//4.
    order : int, default=1
        Order of polynomial detrending.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the DFA exponent (alpha), its interpretation, and metadata.
    """
    series_clean = series.dropna()
    n = len(series_clean)

    if n < 100:
        raise ValueError(f"Series too short for reliable DFA (has {n}, needs 100)")
        
    if max_box_size is None:
        max_box_size = n // 4
        
    profile = np.cumsum(series_clean.values - series_clean.mean())
    box_sizes = np.unique(np.logspace(np.log10(min_box_size), np.log10(max_box_size), 25).astype(int))
    
    fluctuations = []
    valid_box_sizes = []
    
    for box_size in box_sizes:
        if box_size > n:
            continue
        
        # Reshape into non-overlapping boxes
        reshaped_profile = profile[:(n // box_size) * box_size].reshape(-1, box_size)
        x = np.arange(box_size)
        
        # Fit polynomial trends to each box
        coeffs = np.polyfit(x, reshaped_profile.T, order)
        trends = np.polyval(coeffs, x)
        
        # Calculate RMS of detrended fluctuations
        detrended = reshaped_profile - trends.T
        fluctuation = np.sqrt(np.mean(detrended**2, axis=1))
        
        fluctuations.append(np.mean(fluctuation))
        valid_box_sizes.append(box_size)

    if len(fluctuations) < 5:
        raise ValueError("Insufficient valid box sizes for DFA calculation.")

    log_sizes = np.log10(valid_box_sizes)
    log_flucts = np.log10(fluctuations)
    
    alpha, _, r_value, _, _ = stats.linregress(log_sizes, log_flucts)

    # Interpretation for returns series (stationary input)
    if alpha < 0.45:
        interpretation = "Anti-correlated (Negative Memory)"
    elif alpha > 0.55:
        interpretation = "Correlated (Positive Memory)"
    else:
        interpretation = "Uncorrelated (No Memory)"

    return {
        "dfa_alpha": alpha,
        "interpretation": interpretation,
        "r_squared": r_value**2
    }


def calculate_multifractal_spectrum(
    series: pd.Series,
    q_range: Tuple[float, float] = (-5, 5),
    q_step: float = 0.5
) -> Dict[str, Any]:
    """
    Calculate the multifractal spectrum f(α) vs α.

    Parameters:
    -----------
    series : pd.Series
        Time series data.
    q_range : Tuple[float, float], default=(-5, 5)
        Range of q values for multifractal analysis.
    q_step : float, default=0.5
        Step size for q values.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the full spectrum, q values, and summary statistics.
    """
    if len(series) < 256:
        raise ValueError("Series too short for reliable multifractal analysis (minimum 256 points)")
        
    measure = np.abs(np.diff(series.dropna().values)) + 1e-10  # Add epsilon for stability
    n = len(measure)
    
    q_values = np.arange(q_range[0], q_range[1] + q_step, q_step)
    q_values = q_values[q_values != 0]  # Avoid q=0
    
    box_sizes = np.unique(np.logspace(np.log10(10), np.log10(n // 8), 15).astype(int))
    
    tau_q = []
    
    for q in q_values:
        log_partitions = []
        valid_log_sizes = []
        for box_size in box_sizes:
            n_boxes = n // box_size
            if n_boxes < 3: 
                continue
            
            box_measures = np.sum(measure[:n_boxes*box_size].reshape(n_boxes, box_size), axis=1)
            box_measures = box_measures[box_measures > 0]
            if len(box_measures) < 3: 
                continue
            
            probabilities = box_measures / np.sum(box_measures)
            Z_q = np.sum(probabilities**q)
            
            if Z_q > 0:
                log_partitions.append(np.log(Z_q))
                valid_log_sizes.append(np.log(box_size))
        
        if len(valid_log_sizes) >= 5:
            slope, _, _, _, _ = stats.linregress(valid_log_sizes, log_partitions)
            tau_q.append((q - 1) * slope)
        else:
            tau_q.append(np.nan)

    tau_q = np.array(tau_q)
    valid_mask = ~np.isnan(tau_q)
    q_valid = q_values[valid_mask]
    tau_valid = tau_q[valid_mask]

    if len(q_valid) < 5:
        raise ValueError("Insufficient valid τ(q) values for spectrum calculation")
        
    alpha_q = np.gradient(tau_valid, q_valid)
    f_alpha = q_valid * alpha_q - tau_valid
    
    spectrum_width = alpha_q.max() - alpha_q.min()
    
    return {
        'alpha': alpha_q, 
        'f_alpha': f_alpha, 
        'q_values': q_valid, 
        'tau_q': tau_valid,
        'spectrum_width': spectrum_width,
        'multifractality_level': 'High' if spectrum_width > 0.25 else 'Low'
    }


def generate_fbm_path(
    initial_price: float,
    hurst: float,
    days: int,
    volatility: float = 0.2,
    drift: float = 0.05
) -> pd.Series:
    """
    Generate a simulated asset price path using Fractional Brownian Motion.
    
    Parameters:
    -----------
    initial_price : float
        Starting price for the simulation.
    hurst : float
        Hurst exponent (0 < H < 1).
    days : int
        Number of days to simulate.
    volatility : float, default=0.2
        Annual volatility.
    drift : float, default=0.05
        Annual drift rate.
    
    Returns:
    --------
    pd.Series
        A pandas Series of simulated prices with a DatetimeIndex.
    """
    if not HAS_FBM:
        raise ImportError("fbm package required for fBm simulation. `pip install fbm`")
        
    if not 0 < hurst < 1:
        raise ValueError("Hurst exponent must be between 0 and 1")
    
    dt = 1 / 252  # Daily time step
    fbm_generator = FBM(n=days, hurst=hurst, length=1, method='daviesharte')
    fbm_sample = fbm_generator.fbm()
    
    drift_term = (drift - 0.5 * volatility**2) * dt
    vol_term = volatility * np.sqrt(dt) 
    
    returns = drift_term + vol_term * np.diff(fbm_sample, prepend=0)
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Return a pandas Series with a proper index
    date_index = pd.date_range(
        start=pd.Timestamp.now().normalize(), 
        periods=len(prices),
        freq='D'
    )
    
    return pd.Series(prices, index=date_index, name=f"fBm_H{hurst:.2f}")


def calculate_detrended_cross_correlation(
    series1: pd.Series,
    series2: pd.Series,
    min_box_size: int = 10,
    max_box_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calculate Detrended Cross-Correlation Analysis (DCCA) between two series.
    
    Parameters:
    -----------
    series1, series2 : pd.Series
        Two time series to analyze for cross-correlation.
    min_box_size : int, default=10
        Minimum box size.
    max_box_size : int or None, default=None
        Maximum box size.
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing DCCA results.
    """
    # Align series
    aligned = pd.concat([series1, series2], axis=1).dropna()
    if len(aligned) < 100:
        raise ValueError("Insufficient aligned data for DCCA")
        
    x1, x2 = aligned.iloc[:, 0].values, aligned.iloc[:, 1].values
    n = len(x1)
    
    if max_box_size is None:
        max_box_size = n // 4
    
    # Create profiles
    profile1 = np.cumsum(x1 - np.mean(x1))
    profile2 = np.cumsum(x2 - np.mean(x2))
    
    box_sizes = np.unique(np.logspace(np.log10(min_box_size), np.log10(max_box_size), 20).astype(int))
    
    dcca_coeffs = []
    valid_sizes = []
    
    for box_size in box_sizes:
        if box_size > n:
            continue
            
        # Calculate detrended covariance
        n_boxes = n // box_size
        reshaped_1 = profile1[:n_boxes * box_size].reshape(-1, box_size)
        reshaped_2 = profile2[:n_boxes * box_size].reshape(-1, box_size)
        
        x = np.arange(box_size)
        
        # Fit linear trends
        coeffs1 = np.polyfit(x, reshaped_1.T, 1)
        coeffs2 = np.polyfit(x, reshaped_2.T, 1)
        
        trends1 = np.polyval(coeffs1, x).T
        trends2 = np.polyval(coeffs2, x).T
        
        detrended1 = reshaped_1 - trends1
        detrended2 = reshaped_2 - trends2
        
        # Calculate cross-correlation coefficient
        cross_fluct = np.mean(detrended1 * detrended2, axis=1)
        auto_fluct1 = np.mean(detrended1**2, axis=1)
        auto_fluct2 = np.mean(detrended2**2, axis=1)
        
        # DCCA coefficient
        rho = np.mean(cross_fluct) / np.sqrt(np.mean(auto_fluct1) * np.mean(auto_fluct2))
        
        dcca_coeffs.append(rho)
        valid_sizes.append(box_size)
    
    return {
        "dcca_coefficients": dcca_coeffs,
        "box_sizes": valid_sizes,
        "average_correlation": np.mean(dcca_coeffs),
        "correlation_stability": np.std(dcca_coeffs)
    }