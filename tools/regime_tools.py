# tools/regime_tools.py - Clean Market Regime Analysis Tools
"""
Market Regime Analysis Tools - Clean Architecture
===============================================

Tools for identifying and forecasting market regimes using HMM and Hurst analysis.
Stripped of agent dependencies for direct function calls.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import warnings

# Optional dependencies with graceful fallbacks
try:
    from hmmlearn import hmm
    HAS_HMM = True
except ImportError:
    HAS_HMM = False
    print("Warning: hmmlearn not found. HMM functions will be unavailable. `pip install hmmlearn`")

try:
    from .fractal_tools import calculate_hurst
    HAS_FRACTAL = True
except ImportError:
    HAS_FRACTAL = False
    print("Warning: fractal_tools not found. Hurst regime analysis will be unavailable.")

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='hmmlearn')


def detect_hmm_regimes(
    returns: pd.Series, 
    n_regimes: int = 2, 
    max_iter: int = 1000,
    random_state: int = 42
) -> Optional[Dict[str, Any]]:
    """
    Detect market regimes using a Gaussian HMM with standardized, volatility-sorted output.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series for regime detection.
    n_regimes : int, default=2
        Number of regimes to detect.
    max_iter : int, default=1000
        Maximum iterations for HMM fitting.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Returns:
    --------
    Dict[str, Any] or None
        Dictionary containing regime results or None if fitting fails.
    """
    if not HAS_HMM:
        raise ImportError("hmmlearn is not installed. `pip install hmmlearn`")
        
    if len(returns) < n_regimes * 25:
        raise ValueError(f"Not enough data for HMM. Has {len(returns)}, needs {n_regimes * 25}")

    returns_clean = returns.dropna()
    feature_matrix = returns_clean.values.reshape(-1, 1)

    try:
        model = hmm.GaussianHMM(
            n_components=n_regimes, 
            covariance_type="full", 
            n_iter=max_iter,
            random_state=random_state
        )
        model.fit(feature_matrix)
    except Exception as e:
        print(f"HMM model fitting failed: {e}")
        return None

    hidden_states = model.predict(feature_matrix)

    # Standardize regime labels by sorting by volatility
    # This ensures that regime 0 is always the lowest volatility state
    regime_volatilities = [np.sqrt(model.covars_[i][0][0]) for i in range(n_regimes)]
    vol_sorted_indices = np.argsort(regime_volatilities)
    
    # Create a mapping from old, arbitrary labels to new, sorted labels
    label_map = {old_label: new_label for new_label, old_label in enumerate(vol_sorted_indices)}
    standardized_states = np.array([label_map[s] for s in hidden_states])
    
    regime_series = pd.Series(standardized_states, index=returns_clean.index, name='hmm_regime')
    
    # Create characteristics dictionary with the new, sorted labels
    characteristics = {}
    for i in range(n_regimes):
        original_index = vol_sorted_indices[i]
        characteristics[i] = {
            'mean_return': float(model.means_[original_index][0]),
            'volatility': float(np.sqrt(model.covars_[original_index][0][0])),
            'regime_label': f"Regime_{i}_Vol{np.sqrt(model.covars_[original_index][0][0]):.3f}"
        }
        
    # Reorder the transition matrix to match the new labels
    transition_matrix = model.transmat_[np.ix_(vol_sorted_indices, vol_sorted_indices)]

    return {
        "regime_series": regime_series,
        "regime_characteristics": characteristics,
        "transition_matrix": transition_matrix.tolist(),
        "current_regime": int(regime_series.iloc[-1]),
        "model_score": float(model.score(feature_matrix)),
        "n_regimes": n_regimes
    }


def analyze_hurst_regimes(
    series: pd.Series, 
    window: int = 100
) -> Optional[Dict[str, Any]]:
    """
    Perform rolling Hurst analysis to identify memory-based regimes.

    Parameters:
    -----------
    series : pd.Series
        Price or return series with a datetime index.
    window : int, default=100
        Rolling window size for Hurst calculation.

    Returns:
    --------
    Dict[str, Any] or None
        A dictionary containing the results DataFrame and a summary of regime frequencies.
    """
    if not HAS_FRACTAL:
        raise ImportError("calculate_hurst function from fractal_tools not available.")
        
    if len(series) < window:
        return None

    try:
        # Apply rolling Hurst calculation
        rolling_hurst_values = series.rolling(window).apply(
            lambda x: calculate_hurst(pd.Series(x)).get('hurst_exponent', np.nan),
            raw=False
        ).dropna()

        df = pd.DataFrame({'hurst': rolling_hurst_values})
        
        conditions = [
            df['hurst'] < 0.45,
            (df['hurst'] >= 0.45) & (df['hurst'] <= 0.55),
            df['hurst'] > 0.55
        ]
        choices = ['Mean-Reverting', 'Random Walk', 'Trending']
        df['hurst_regime'] = np.select(conditions, choices, default='N/A')
        
        # Add summary for easier interpretation
        regime_counts = df['hurst_regime'].value_counts(normalize=True).round(4)
        summary = {
            "time_in_mean_reverting_pct": float(regime_counts.get("Mean-Reverting", 0) * 100),
            "time_in_random_walk_pct": float(regime_counts.get("Random Walk", 0) * 100),
            "time_in_trending_pct": float(regime_counts.get("Trending", 0) * 100),
            "current_regime": str(df['hurst_regime'].iloc[-1]),
            "current_hurst": float(df['hurst'].iloc[-1])
        }
        
        return {
            "results_df": df,
            "summary": summary
        }
        
    except Exception as e:
        print(f"Error in Hurst regime analysis: {e}")
        return None


def forecast_regime_transition_probability(
    hmm_results: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Calculates the probability of transitioning from the current regime to all others.

    Parameters:
    -----------
    hmm_results : Dict[str, Any]
        The complete result dictionary returned by `detect_hmm_regimes`.

    Returns:
    --------
    Dict[str, Any] or None
        A dictionary detailing the transition probabilities with added context.
    """
    try:
        trans_mat = np.array(hmm_results['transition_matrix'])
        current_regime = hmm_results['current_regime']
        characteristics = hmm_results['regime_characteristics']
        
        transition_probs = trans_mat[current_regime, :]
        
        # Create detailed forecast
        forecast = {
            'from_regime': {
                'index': int(current_regime),
                'volatility': round(characteristics[current_regime]['volatility'], 6),
                'mean_return': round(characteristics[current_regime]['mean_return'], 6)
            },
            'transition_forecast': []
        }
        
        for i, prob in enumerate(transition_probs):
            forecast['transition_forecast'].append({
                'to_regime_index': i,
                'probability': round(float(prob), 4),
                'volatility': round(characteristics[i]['volatility'], 6),
                'mean_return': round(characteristics[i]['mean_return'], 6),
                'regime_label': characteristics[i]['regime_label']
            })
            
        # Add most likely transition
        most_likely_idx = np.argmax(transition_probs)
        forecast['most_likely_transition'] = {
            'to_regime': int(most_likely_idx),
            'probability': round(float(transition_probs[most_likely_idx]), 4)
        }
            
        return forecast
        
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error forecasting regime transition: {e}")
        return None


def detect_volatility_regimes(
    returns: pd.Series,
    window: int = 30,
    threshold_low: float = 0.15,
    threshold_high: float = 0.25
) -> Dict[str, Any]:
    """
    Simple volatility-based regime detection using rolling standard deviation.
    
    Parameters:
    -----------
    returns : pd.Series
        Return series.
    window : int, default=30
        Rolling window for volatility calculation.
    threshold_low : float, default=0.15
        Lower threshold for low volatility regime.
    threshold_high : float, default=0.25
        Upper threshold for high volatility regime.
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with volatility regime results.
    """
    try:
        # Calculate rolling volatility (annualized)
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        
        conditions = [
            rolling_vol < threshold_low,
            (rolling_vol >= threshold_low) & (rolling_vol <= threshold_high),
            rolling_vol > threshold_high
        ]
        choices = ['Low Volatility', 'Medium Volatility', 'High Volatility']
        
        vol_regimes = pd.Series(
            np.select(conditions, choices, default='Unknown'),
            index=rolling_vol.index,
            name='volatility_regime'
        )
        
        # Calculate regime statistics
        regime_stats = {}
        for regime in choices:
            mask = vol_regimes == regime
            if mask.sum() > 0:
                regime_returns = returns[mask]
                regime_stats[regime] = {
                    'frequency': float(mask.sum() / len(vol_regimes.dropna())),
                    'avg_return': float(regime_returns.mean()),
                    'avg_volatility': float(regime_returns.std() * np.sqrt(252)),
                    'sharpe_ratio': float(regime_returns.mean() / regime_returns.std() * np.sqrt(252)) if regime_returns.std() > 0 else 0.0
                }
        
        return {
            "regime_series": vol_regimes.dropna(),
            "regime_statistics": regime_stats,
            "current_regime": str(vol_regimes.dropna().iloc[-1]) if len(vol_regimes.dropna()) > 0 else "Unknown",
            "current_volatility": float(rolling_vol.dropna().iloc[-1]) if len(rolling_vol.dropna()) > 0 else 0.0,
            "thresholds": {
                "low": threshold_low,
                "high": threshold_high
            }
        }
        
    except Exception as e:
        return {
            "error": f"Error in volatility regime detection: {str(e)}",
            "regime_series": pd.Series(),
            "regime_statistics": {},
            "current_regime": "Unknown"
        }


def analyze_regime_persistence(regime_series: pd.Series) -> Dict[str, Any]:
    """
    Analyze how persistent different regimes are (average duration).
    
    Parameters:
    -----------
    regime_series : pd.Series
        Series of regime labels.
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with persistence statistics.
    """
    try:
        regime_changes = regime_series != regime_series.shift(1)
        regime_groups = regime_changes.cumsum()
        
        durations = {}
        for regime in regime_series.unique():
            regime_mask = regime_series == regime
            regime_group_ids = regime_groups[regime_mask].unique()
            
            regime_durations = []
            for group_id in regime_group_ids:
                duration = (regime_groups == group_id).sum()
                regime_durations.append(duration)
            
            if regime_durations:
                durations[str(regime)] = {
                    'avg_duration': float(np.mean(regime_durations)),
                    'median_duration': float(np.median(regime_durations)),
                    'max_duration': int(np.max(regime_durations)),
                    'min_duration': int(np.min(regime_durations)),
                    'num_episodes': len(regime_durations)
                }
        
        return {
            "persistence_stats": durations,
            "total_regime_changes": int((regime_series != regime_series.shift(1)).sum() - 1),  # -1 for first NaN
            "avg_regime_duration": float(len(regime_series) / (regime_changes.sum() - 1)) if regime_changes.sum() > 1 else float(len(regime_series))
        }
        
    except Exception as e:
        return {
            "error": f"Error analyzing regime persistence: {str(e)}",
            "persistence_stats": {},
            "total_regime_changes": 0
        }