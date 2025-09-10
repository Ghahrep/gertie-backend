# tools/strategy_tools.py - Complete Strategy & Analysis Tools
"""
Strategy Design and Analysis Tools - Complete Version  
====================================================

Clean financial strategy tools with all functions included.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


def calculate_hurst(
    series: pd.Series,
    min_window: int = 10,
    max_window: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calculate Hurst exponent using Rescaled Range (R/S) analysis.
    """
    series_clean = series.dropna()
    n = len(series_clean)

    if n < 50:
        raise ValueError(f"Series too short for reliable Hurst calculation (has {n}, needs 50)")

    if max_window is None:
        max_window = n // 2

    min_window = max(2, min_window)
    max_window = min(n - 1, max_window)
    if min_window >= max_window:
        raise ValueError("min_window must be smaller than max_window.")

    windows = np.unique(np.logspace(np.log10(min_window), np.log10(max_window), 20).astype(int))
    
    rs_values = []
    valid_windows = []

    for window in windows:
        if window > n:
            continue

        # Vectorized R/S calculation
        try:
            sub_series_matrix = np.lib.stride_tricks.as_strided(
                series_clean.values,
                shape=(n - window + 1, window),
                strides=(series_clean.values.strides[0], series_clean.values.strides[0])
            )
            
            mean_centered = sub_series_matrix - sub_series_matrix.mean(axis=1, keepdims=True)
            cum_dev = mean_centered.cumsum(axis=1)
            R = cum_dev.max(axis=1) - cum_dev.min(axis=1)
            S = sub_series_matrix.std(axis=1)

            valid_mask = S > 1e-10
            if np.any(valid_mask):
                rs_window = (R[valid_mask] / S[valid_mask])
                rs_values.append(rs_window.mean())
                valid_windows.append(window)
        except Exception:
            continue

    if len(rs_values) < 5:
        raise ValueError(f"Insufficient valid windows ({len(rs_values)}) for Hurst regression.")

    # Log-log regression to find Hurst exponent
    log_rs = np.log10(rs_values)
    log_windows = np.log10(valid_windows)
    
    slope, intercept, r_value, _, _ = stats.linregress(log_windows, log_rs)
    
    hurst_exponent = np.clip(slope, 0.01, 0.99)

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


def design_mean_reversion_strategy(
    price_data: pd.DataFrame,
    hurst_threshold: float = 0.45
) -> Dict[str, Any]:
    """
    Screen assets for mean-reverting candidates based on Hurst exponent.
    """
    if price_data.empty: 
        return {"success": False, "error": "Price data is empty."}

    hurst_results = []
    for ticker in price_data.columns:
        try:
            price_series = price_data[ticker].dropna()
            if len(price_series) < 100: 
                continue
            
            hurst_dict = calculate_hurst(price_series)
            if hurst_dict:
                h_value = hurst_dict['hurst_exponent']
                hurst_results.append({'ticker': ticker, 'hurst': round(h_value, 4)})
        except Exception as e:
            print(f"Could not process {ticker} for Hurst calculation: {e}")
            continue

    candidates = [res for res in hurst_results if res['hurst'] < hurst_threshold]
    sorted_candidates = sorted(candidates, key=lambda x: x['hurst'])
    
    if not sorted_candidates:
        return {"success": False, "error": "No suitable mean-reverting assets found."}
    
    return {
        "success": True,
        "strategy_type": "Mean-Reversion",
        "candidates": sorted_candidates,
    }


def design_momentum_strategy(
    asset_prices: pd.DataFrame, 
    lookback_period: int = 252
) -> Dict[str, Any]:
    """
    Screen assets for momentum candidates based on total return over lookback period.
    """
    if asset_prices.empty or len(asset_prices) < 200:
        return {"success": False, "error": "Insufficient price data. At least 200 days required."}

    actual_lookback = min(lookback_period, len(asset_prices) - 1)

    # Calculate total return over lookback period
    returns = asset_prices.pct_change().dropna()
    momentum_scores = (1 + returns.tail(actual_lookback)).prod() - 1
    
    # Rank assets from highest to lowest momentum
    ranked_assets = momentum_scores.sort_values(ascending=False)
    
    candidates = []
    for ticker, score in ranked_assets.items():
        if score > 0:
            candidates.append({
                "ticker": str(ticker), 
                "momentum_score_pct": round(score * 100, 2)
            })

    if not candidates:
        return {"success": False, "error": "No assets with positive momentum found."}

    return {
        "success": True,
        "strategy_type": "Momentum", 
        "candidates": candidates
    }


def detect_hmm_regimes(
    returns: pd.Series, 
    n_regimes: int = 2, 
    n_init: int = 10 
) -> Optional[Dict[str, Any]]:
    """
    Detect market regimes using Gaussian HMM with volatility-sorted output.
    """
    try:
        from hmmlearn import hmm
    except ImportError:
        print("Warning: hmmlearn not installed. Skipping HMM regime detection.")
        return None
        
    if len(returns) < n_regimes * 25:
        raise ValueError(f"Not enough data for HMM. Has {len(returns)}, needs {n_regimes * 25}")

    returns_clean = returns.dropna()
    feature_matrix = returns_clean.values.reshape(-1, 1)

    try:
        model = hmm.GaussianHMM(
            n_components=n_regimes, 
            covariance_type="full", 
            n_iter=1000,
            random_state=42
        )
        model.fit(feature_matrix)
    except Exception as e:
        print(f"HMM model fitting failed: {e}")
        return None

    hidden_states = model.predict(feature_matrix)

    # Standardize regime labels by sorting by volatility
    regime_volatilities = [np.sqrt(model.covars_[i][0][0]) for i in range(n_regimes)]
    vol_sorted_indices = np.argsort(regime_volatilities)
    
    label_map = {old_label: new_label for new_label, old_label in enumerate(vol_sorted_indices)}
    standardized_states = np.array([label_map[s] for s in hidden_states])
    
    regime_series = pd.Series(standardized_states, index=returns_clean.index, name='hmm_regime')
    
    # Create characteristics dictionary with sorted labels
    characteristics = {}
    for i in range(n_regimes):
        original_index = vol_sorted_indices[i]
        characteristics[i] = {
            'mean_return': model.means_[original_index][0],
            'volatility': np.sqrt(model.covars_[original_index][0][0])
        }
        
    # Reorder transition matrix
    transition_matrix = model.transmat_[np.ix_(vol_sorted_indices, vol_sorted_indices)]

    return {
        "regime_series": regime_series,
        "regime_characteristics": characteristics,
        "transition_matrix": transition_matrix,
        "current_regime": regime_series.iloc[-1],
        "fitted_model": model
    }


def analyze_chat_for_biases(chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Analyze user chat history to detect potential behavioral biases.
    """
    biases_found = {}
    user_messages = [msg['content'].lower() for msg in chat_history if msg.get('role') == 'user']
    
    # Loss Aversion / Panic Selling Detection
    loss_aversion_keywords = ['panic', 'sell everything', 'market crash', 'get out', 'afraid of losing']
    if any(any(kw in msg for kw in loss_aversion_keywords) for msg in user_messages):
        biases_found['Loss Aversion'] = {
            "finding": "Detected language related to panic or selling based on fear.",
            "suggestion": "Consider sticking to your long-term plan. Emotional decisions during downturns can often hurt performance."
        }

    # Herding / FOMO Detection
    herding_keywords = ['everyone is buying', 'hot stock', 'get in on', 'don\'t want to miss out', 'fomo']
    if any(any(kw in msg for kw in herding_keywords) for msg in user_messages):
        biases_found['Herding Behavior (FOMO)'] = {
            "finding": "Detected language suggesting a desire to follow the crowd or chase popular trends.",
            "suggestion": "Ensure investment decisions are based on your own research and strategy, not just popularity."
        }

    # Overconfidence / Frequent Rebalancing Detection
    rebalance_queries = [msg for msg in user_messages if 'rebalance' in msg]
    if len(rebalance_queries) > 2:
        biases_found['Over-trading / Overconfidence'] = {
            "finding": "Noticed multiple requests for rebalancing in a short period.",
            "suggestion": "Frequent trading can increase costs and may not always lead to better results. Ensure each change aligns with your long-term goals."
        }
        
    if not biases_found:
        return {"success": True, "summary": "No strong behavioral biases were detected in the recent conversation."}

    return {"success": True, "biases_detected": biases_found}


def apply_market_shock(
    returns: pd.DataFrame, 
    shock_scenario: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply predefined shock to historical returns series.
    """
    shocked_returns = returns.copy()
    shock_type = shock_scenario.get("type", "none")
    
    print(f"Applying shock type: {shock_type}")

    if shock_type == "market_crash":
        impact = shock_scenario.get("impact_pct", -0.20)
        shock_day = shocked_returns.sample(1).index
        shocked_returns.loc[shock_day] = impact
    
    elif shock_type == "interest_rate_spike":
        impact = shock_scenario.get("impact_pct", -0.05)
        shocked_returns = shocked_returns + (impact / len(shocked_returns))

    return shocked_returns


def calculate_dfa(
    series: pd.Series,
    min_box_size: int = 10,
    max_box_size: Optional[int] = None,
    order: int = 1
) -> Dict[str, Any]:
    """
    Calculate the scaling exponent using Detrended Fluctuation Analysis (DFA).
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

    # Interpretation for returns series
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