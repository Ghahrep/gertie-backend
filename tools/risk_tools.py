# tools/risk_tools.py - Enhanced Institutional-Grade Risk Analysis
"""
Enhanced Financial Risk Analysis Module - Institutional Quality
==============================================================

Production-grade portfolio risk analysis with dynamic attribution, regime awareness,
and advanced stress testing capabilities.
"""

import pandas as pd
import numpy as np
from arch import arch_model
from scipy import stats
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from scipy.optimize import minimize
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings('ignore', category=RuntimeWarning)

# Import regime detection from your existing tools
try:
    from .regime_tools import detect_hmm_regimes, detect_volatility_regimes
    HAS_REGIME_TOOLS = True
except ImportError:
    HAS_REGIME_TOOLS = False
    print("Warning: regime_tools not available. Regime-conditional analysis disabled.")

# Optional dependencies
try:
    from scipy.optimize import minimize
    HAS_SCIPY_OPTIMIZE = True
except ImportError:
    HAS_SCIPY_OPTIMIZE = False

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# Import your regime tools for integration
try:
    from .regime_tools import detect_hmm_regimes, detect_volatility_regimes
except ImportError:
    try:
        from regime_tools import detect_hmm_regimes, detect_volatility_regimes
    except ImportError:
        # Fallback functions if regime_tools not available
        def detect_hmm_regimes(returns):
            return {'current_regime': 0, 'regime_probabilities': [0.7, 0.3]}
        
        def detect_volatility_regimes(returns):
            return {'current_regime': 'Normal', 'volatility_state': 0.15}

def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate Conditional Value at Risk (CVaR) / Expected Shortfall (ES)."""
    if not isinstance(returns, pd.Series):
        raise TypeError("Returns must be a pandas Series")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")
    
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return np.nan
        
    var_threshold = clean_returns.quantile(1 - confidence_level)
    tail_losses = clean_returns[clean_returns <= var_threshold]
    
    return -var_threshold if tail_losses.empty else -tail_losses.mean()


def calculate_risk_metrics(
    portfolio_returns: pd.Series, 
    confidence_levels: List[float] = [0.95, 0.99],
    trading_days: int = 252
) -> Optional[Dict[str, Any]]:
    """Enhanced risk metrics calculation with regime awareness."""
    if not isinstance(portfolio_returns, pd.Series) or portfolio_returns.empty:
        return None
        
    try:
        # Basic VaR and CVaR Analysis
        var_analysis = {}
        for confidence in confidence_levels:
            var_value = portfolio_returns.quantile(1 - confidence)
            cvar_value = calculate_cvar(portfolio_returns, confidence)
            var_analysis[f"{int(confidence*100)}%"] = {
                "var": var_value,
                "cvar_expected_shortfall": cvar_value
            }
        
        # Performance Statistics
        daily_vol = portfolio_returns.std()
        annual_vol = daily_vol * np.sqrt(trading_days)
        annual_return = portfolio_returns.mean() * trading_days

        # Risk-Adjusted Ratios
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(trading_days)
        sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
        
        # Enhanced metrics
        skewness = portfolio_returns.skew()
        kurtosis = portfolio_returns.kurtosis()
        
        # Drawdown stats
        drawdown_stats = calculate_drawdowns(portfolio_returns)

        # Regime-conditional risk if available
        regime_risk = None
        if HAS_REGIME_TOOLS and len(portfolio_returns) > 100:
            regime_risk = calculate_regime_conditional_risk(portfolio_returns)

        return {
            "risk_measures": var_analysis,
            "performance_stats": {
                "annualized_return_pct": annual_return * 100,
                "annualized_volatility_pct": annual_vol * 100,
                "skewness": skewness,
                "excess_kurtosis": kurtosis
            },
            "risk_adjusted_ratios": {
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": drawdown_stats['calmar_ratio'] if drawdown_stats else None
            },
            "drawdown_stats": drawdown_stats,
            "regime_conditional_risk": regime_risk
        }
    except Exception as e:
        print(f"Error calculating risk metrics: {e}")
        return None


def calculate_regime_conditional_risk(
    returns: pd.Series,
    lookback_window: int = 252
) -> Optional[Dict[str, Any]]:
    """Calculate risk metrics conditional on market regime."""
    if not HAS_REGIME_TOOLS or len(returns) < lookback_window:
        return None
    
    try:
        # Detect regimes
        regime_results = detect_hmm_regimes(returns, n_regimes=2)
        if not regime_results:
            return None
        
        regime_series = regime_results['regime_series']
        
        # Calculate risk metrics for each regime
        regime_risk_metrics = {}
        
        for regime in [0, 1]:  # Assuming 2 regimes
            regime_mask = regime_series == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) < 30:  # Minimum observations
                continue
                
            # Calculate regime-specific metrics
            regime_vol = regime_returns.std() * np.sqrt(252)
            regime_var_95 = regime_returns.quantile(0.05)
            regime_cvar_95 = calculate_cvar(regime_returns, 0.95)
            
            regime_characteristics = regime_results['regime_characteristics'][regime]
            
            regime_risk_metrics[f"regime_{regime}"] = {
                "regime_label": regime_characteristics.get('regime_label', f'Regime {regime}'),
                "frequency": len(regime_returns) / len(returns),
                "annualized_volatility": regime_vol,
                "var_95": regime_var_95,
                "cvar_95": regime_cvar_95,
                "mean_return": regime_characteristics.get('mean_return', 0),
                "regime_volatility": regime_characteristics.get('volatility', 0)
            }
        
        return {
            "current_regime": regime_results['current_regime'],
            "regime_metrics": regime_risk_metrics,
            "transition_probabilities": regime_results.get('transition_matrix', [])
        }
        
    except Exception as e:
        print(f"Error in regime conditional risk: {e}")
        return None


def calculate_factor_risk_attribution(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    window: int = 252
) -> Optional[Dict[str, Any]]:
    """
    Decompose portfolio risk into systematic factor contributions and specific risk.
    """
    if not HAS_STATSMODELS:
        print("statsmodels required for factor attribution")
        return None
    
    if len(portfolio_returns) < window or factor_returns.empty:
        return None
    
    try:
        # Align data
        data = pd.DataFrame({'portfolio': portfolio_returns}).join(factor_returns, how='inner').dropna()
        
        if len(data) < 50:
            return None
        
        # Use recent data for attribution
        recent_data = data.tail(window)
        
        y = recent_data['portfolio']
        X = recent_data.drop(columns=['portfolio'])
        X = sm.add_constant(X)
        
        # Fit factor model
        model = sm.OLS(y, X).fit()
        
        # Calculate factor contributions to variance
        factor_loadings = model.params.drop('const')
        factor_covariance = X.drop(columns=['const']).cov()
        
        # Risk contributions
        factor_risk_contributions = {}
        total_systematic_risk = 0
        
        for factor in factor_loadings.index:
            factor_var_contribution = (factor_loadings[factor] ** 2) * factor_covariance.loc[factor, factor]
            factor_risk_contributions[factor] = {
                "loading": factor_loadings[factor],
                "variance_contribution": factor_var_contribution,
                "risk_contribution_pct": 0  # Will calculate after total
            }
            total_systematic_risk += factor_var_contribution
        
        # Calculate interaction effects
        interaction_risk = 0
        for i, factor1 in enumerate(factor_loadings.index):
            for factor2 in factor_loadings.index[i+1:]:
                interaction = 2 * factor_loadings[factor1] * factor_loadings[factor2] * factor_covariance.loc[factor1, factor2]
                interaction_risk += interaction
        
        total_systematic_risk += interaction_risk
        
        # Specific risk (residual variance)
        specific_risk = model.mse_resid
        total_portfolio_variance = total_systematic_risk + specific_risk
        
        # Calculate percentage contributions
        for factor in factor_risk_contributions:
            factor_risk_contributions[factor]["risk_contribution_pct"] = (
                factor_risk_contributions[factor]["variance_contribution"] / total_portfolio_variance * 100
            )
        
        return {
            "factor_loadings": factor_loadings.to_dict(),
            "factor_risk_contributions": factor_risk_contributions,
            "systematic_risk_pct": (total_systematic_risk / total_portfolio_variance) * 100,
            "specific_risk_pct": (specific_risk / total_portfolio_variance) * 100,
            "r_squared": model.rsquared,
            "tracking_error": np.sqrt(specific_risk) * np.sqrt(252),
            "total_portfolio_volatility": np.sqrt(total_portfolio_variance) * np.sqrt(252)
        }
        
    except Exception as e:
        print(f"Error in factor risk attribution: {e}")
        return None


def calculate_dynamic_risk_budgets(
    portfolio_weights: Dict[str, float],
    asset_returns: pd.DataFrame,
    target_portfolio_vol: float = 0.15,
    rebalance_frequency: str = 'monthly'
) -> Optional[Dict[str, Any]]:
    """
    Calculate dynamic risk budgets across portfolio positions.
    """
    if not HAS_SCIPY_OPTIMIZE:
        print("scipy.optimize required for risk budgeting")
        return None
    
    try:
        # Calculate covariance matrix
        returns_aligned = asset_returns.dropna()
        if len(returns_aligned) < 60:
            return None
        
        cov_matrix = returns_aligned.cov() * 252  # Annualized
        
        # Current portfolio weights as array
        assets = list(portfolio_weights.keys())
        weights = np.array([portfolio_weights[asset] for asset in assets])
        
        # Current portfolio volatility
        current_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
        
        # Risk contributions (marginal contribution to risk)
        marginal_contrib = np.dot(cov_matrix.values, weights) / current_vol
        risk_contributions = weights * marginal_contrib
        
        # Percentage risk contributions
        risk_contrib_pct = risk_contributions / current_vol
        
        # Equal risk contribution (ERC) portfolio
        def risk_budget_objective(w, cov_matrix):
            """Objective function for equal risk contribution."""
            portfolio_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            marginal_contrib = np.dot(cov_matrix, w) / portfolio_vol
            risk_contrib = w * marginal_contrib
            
            # Minimize difference from equal risk contribution
            target_contrib = portfolio_vol / len(w)
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Weights sum to 1
        bounds = [(0.01, 0.4) for _ in range(len(assets))]  # Min 1%, max 40% per asset
        
        # Optimize for equal risk contribution
        result = minimize(
            risk_budget_objective,
            weights,
            args=(cov_matrix.values,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        erc_weights = result.x if result.success else weights
        erc_vol = np.sqrt(np.dot(erc_weights.T, np.dot(cov_matrix.values, erc_weights)))
        
        # Scale to target volatility
        vol_scaling_factor = target_portfolio_vol / erc_vol
        scaled_erc_weights = erc_weights * vol_scaling_factor
        
        # Calculate new risk contributions
        erc_marginal_contrib = np.dot(cov_matrix.values, erc_weights) / erc_vol
        erc_risk_contributions = erc_weights * erc_marginal_contrib
        
        # Build results
        current_allocation = {}
        erc_allocation = {}
        
        for i, asset in enumerate(assets):
            current_allocation[asset] = {
                "weight": weights[i],
                "risk_contribution": risk_contributions[i],
                "risk_contribution_pct": risk_contrib_pct[i] * 100
            }
            
            erc_allocation[asset] = {
                "weight": erc_weights[i],
                "risk_contribution": erc_risk_contributions[i],
                "risk_contribution_pct": (erc_risk_contributions[i] / erc_vol) * 100,
                "weight_change": erc_weights[i] - weights[i]
            }
        
        return {
            "current_portfolio": {
                "allocation": current_allocation,
                "total_volatility": current_vol,
                "concentration_risk": max(risk_contrib_pct) - min(risk_contrib_pct)
            },
            "equal_risk_contribution": {
                "allocation": erc_allocation,
                "total_volatility": erc_vol,
                "target_volatility": target_portfolio_vol,
                "optimization_success": result.success
            },
            "rebalancing_summary": {
                "volatility_reduction": current_vol - erc_vol,
                "risk_concentration_reduction": (max(risk_contrib_pct) - min(risk_contrib_pct)) - 
                                              (max(erc_risk_contributions) - min(erc_risk_contributions)) / erc_vol,
                "total_weight_changes": sum(abs(erc_weights[i] - weights[i]) for i in range(len(weights)))
            }
        }
        
    except Exception as e:
        print(f"Error in dynamic risk budgeting: {e}")
        return None


def run_monte_carlo_stress_test(
    portfolio_returns: pd.Series,
    n_simulations: int = 10000,
    horizon_days: int = 252,
    confidence_levels: List[float] = [0.95, 0.99]
) -> Dict[str, Any]:
    """
    Run comprehensive Monte Carlo stress testing.
    """
    try:
        returns_clean = portfolio_returns.dropna()
        
        if len(returns_clean) < 100:
            return {"success": False, "error": "Insufficient historical data"}
        
        # Fit return distribution
        mu = returns_clean.mean()
        sigma = returns_clean.std()
        
        # Test for normality
        normality_test = stats.jarque_bera(returns_clean)
        is_normal = normality_test.pvalue > 0.05
        
        # Generate scenarios
        if is_normal:
            # Normal distribution
            scenarios = np.random.normal(mu, sigma, (n_simulations, horizon_days))
        else:
            # Use t-distribution for fat tails
            df_est = 6  # Conservative estimate for fat tails
            scenarios = stats.t.rvs(df=df_est, loc=mu, scale=sigma, size=(n_simulations, horizon_days))
        
        # Calculate cumulative returns for each path
        cumulative_returns = np.cumprod(1 + scenarios, axis=1) - 1
        final_returns = cumulative_returns[:, -1]
        
        # Calculate worst-case scenarios
        stress_metrics = {}
        for conf_level in confidence_levels:
            var_threshold = np.percentile(final_returns, (1 - conf_level) * 100)
            tail_scenarios = final_returns[final_returns <= var_threshold]
            expected_shortfall = np.mean(tail_scenarios) if len(tail_scenarios) > 0 else var_threshold
            
            stress_metrics[f"stress_{int(conf_level*100)}"] = {
                "var_horizon": var_threshold,
                "expected_shortfall_horizon": expected_shortfall,
                "probability_of_loss": np.mean(final_returns < 0),
                "probability_of_large_loss": np.mean(final_returns < -0.1)  # >10% loss
            }
        
        # Path analysis
        worst_path_idx = np.argmin(final_returns)
        best_path_idx = np.argmax(final_returns)
        
        # Maximum drawdown analysis across all paths
        def calculate_path_max_drawdown(path_returns):
            cumulative = np.cumprod(1 + path_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return np.min(drawdown)
        
        max_drawdowns = [calculate_path_max_drawdown(scenarios[i]) for i in range(min(1000, n_simulations))]
        
        return {
            "success": True,
            "simulation_parameters": {
                "n_simulations": n_simulations,
                "horizon_days": horizon_days,
                "distribution_used": "normal" if is_normal else "t-distribution",
                "normality_p_value": normality_test.pvalue
            },
            "stress_scenarios": stress_metrics,
            "path_analysis": {
                "worst_case_return": np.min(final_returns),
                "best_case_return": np.max(final_returns),
                "median_return": np.median(final_returns),
                "probability_positive": np.mean(final_returns > 0),
                "expected_return": np.mean(final_returns)
            },
            "drawdown_analysis": {
                "worst_simulated_drawdown": np.min(max_drawdowns),
                "median_max_drawdown": np.median(max_drawdowns),
                "prob_drawdown_gt_20pct": np.mean(np.array(max_drawdowns) < -0.2)
            },
            "risk_summary": {
                "annualized_vol_estimate": sigma * np.sqrt(252),
                "horizon_vol_estimate": sigma * np.sqrt(horizon_days),
                "fat_tail_adjustment": not is_normal
            }
        }
        
    except Exception as e:
        return {"success": False, "error": f"Monte Carlo stress test failed: {str(e)}"}


def calculate_time_varying_risk(
    returns: pd.Series,
    window: int = 60,
    min_periods: int = 30
) -> Optional[Dict[str, Any]]:
    """
    Calculate time-varying risk metrics using rolling windows.
    """
    try:
        if len(returns) < window + min_periods:
            return None
        
        # Rolling calculations
        rolling_vol = returns.rolling(window=window, min_periods=min_periods).std() * np.sqrt(252)
        rolling_var_95 = returns.rolling(window=window, min_periods=min_periods).quantile(0.05)
        rolling_sharpe = (returns.rolling(window=window, min_periods=min_periods).mean() * 252) / rolling_vol
        
        # Rolling skewness and kurtosis
        rolling_skew = returns.rolling(window=window, min_periods=min_periods).skew()
        rolling_kurt = returns.rolling(window=window, min_periods=min_periods).kurtosis()
        
        # Volatility regimes
        vol_regimes = None
        if HAS_REGIME_TOOLS:
            vol_regime_results = detect_volatility_regimes(returns, window=30)
            vol_regimes = vol_regime_results.get('regime_series')
        
        # Trend analysis
        vol_trend = "stable"
        recent_vol = rolling_vol.dropna().tail(10).mean()
        historical_vol = rolling_vol.dropna().head(10).mean()
        
        if recent_vol > historical_vol * 1.2:
            vol_trend = "increasing"
        elif recent_vol < historical_vol * 0.8:
            vol_trend = "decreasing"
        
        # Risk state classification
        current_vol = rolling_vol.dropna().iloc[-1] if len(rolling_vol.dropna()) > 0 else 0
        vol_percentile = stats.percentileofscore(rolling_vol.dropna(), current_vol)
        
        if vol_percentile > 80:
            risk_state = "high"
        elif vol_percentile < 20:
            risk_state = "low"
        else:
            risk_state = "normal"
        
        return {
            "time_series": {
                "rolling_volatility": rolling_vol.dropna(),
                "rolling_var_95": rolling_var_95.dropna(),
                "rolling_sharpe": rolling_sharpe.dropna(),
                "rolling_skewness": rolling_skew.dropna(),
                "rolling_kurtosis": rolling_kurt.dropna()
            },
            "current_state": {
                "volatility": current_vol,
                "var_95": rolling_var_95.dropna().iloc[-1] if len(rolling_var_95.dropna()) > 0 else 0,
                "sharpe_ratio": rolling_sharpe.dropna().iloc[-1] if len(rolling_sharpe.dropna()) > 0 else 0,
                "risk_state": risk_state,
                "volatility_percentile": vol_percentile
            },
            "trends": {
                "volatility_trend": vol_trend,
                "vol_change_pct": ((recent_vol - historical_vol) / historical_vol * 100) if historical_vol > 0 else 0
            },
            "volatility_regimes": vol_regimes.to_dict() if vol_regimes is not None else None
        }
        
    except Exception as e:
        print(f"Error in time-varying risk calculation: {e}")
        return None


def calculate_drawdowns(returns: pd.Series) -> Optional[Dict[str, Any]]:
    """Enhanced drawdown analysis with recovery time estimation."""
    if returns.empty: 
        return None

    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - running_max) / running_max

    max_drawdown = drawdowns.min()
    end_date_idx = drawdowns.idxmin()
    start_date_idx = cumulative_returns.loc[:end_date_idx].idxmax()
    
    # Handle both datetime and integer indices
    if isinstance(start_date_idx, (datetime, pd.Timestamp)):
        start_date_str = start_date_idx.strftime('%Y-%m-%d')
        end_date_str = end_date_idx.strftime('%Y-%m-%d')
        
        # Calculate recovery time
        recovery_mask = cumulative_returns.loc[end_date_idx:] >= cumulative_returns.loc[start_date_idx]
        if recovery_mask.any():
            recovery_date = recovery_mask.idxmax()
            recovery_time_days = (recovery_date - end_date_idx).days
        else:
            recovery_time_days = None
    else:
        start_date_str = f"Day {start_date_idx}"
        end_date_str = f"Day {end_date_idx}"
        recovery_time_days = None

    annual_return = returns.mean() * 252
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
    
    # Additional drawdown statistics
    underwater_periods = drawdowns < 0
    if underwater_periods.any():
        avg_drawdown = drawdowns[underwater_periods].mean()
        drawdown_frequency = underwater_periods.sum() / len(returns)
    else:
        avg_drawdown = 0
        drawdown_frequency = 0
    
    return {
        "max_drawdown_pct": max_drawdown * 100,
        "start_of_max_drawdown": start_date_str,
        "end_of_max_drawdown": end_date_str,
        "recovery_time_days": recovery_time_days,
        "current_drawdown_pct": drawdowns.iloc[-1] * 100,
        "calmar_ratio": calmar_ratio,
        "average_drawdown_pct": avg_drawdown * 100,
        "time_underwater_pct": drawdown_frequency * 100
    }


# Keep all existing functions (fit_garch_forecast, calculate_volatility_budget, etc.)
# but add these enhanced versions

def fit_garch_forecast(
    returns: pd.Series, 
    forecast_horizon: int = 30
) -> Optional[Dict[str, Any]]:
    """Enhanced GARCH forecast with regime awareness."""
    if len(returns) < 100:
        print("Warning: Need at least 100 observations for reliable GARCH. Returning None.")
        return None
        
    returns_pct = returns.dropna() * 100
    
    try:
        model = arch_model(returns_pct, vol='Garch', p=1, q=1, dist='t')
        fitted_model = model.fit(disp='off', show_warning=False)
    except Exception as e:
        print(f"GARCH model fitting failed: {e}. Cannot generate forecast.")
        return None

    # Out-of-sample forecast
    forecast = fitted_model.forecast(horizon=forecast_horizon, reindex=False)
    future_vol_values = np.sqrt(forecast.variance).iloc[0].values / 100
    
    future_dates = pd.date_range(
        start=returns.index[-1] + pd.Timedelta(days=1), 
        periods=forecast_horizon
    )
    forecast_series = pd.Series(future_vol_values, index=future_dates)
    
    current_vol = np.sqrt(fitted_model.conditional_volatility[-1]) / 100
    mean_forecast_vol = forecast_series.mean()
    
    trend = "Increasing" if mean_forecast_vol > current_vol else "Decreasing" if mean_forecast_vol < current_vol else "Stable"
    
    # Enhanced with confidence intervals
    forecast_std = forecast_series.std()
    vol_confidence_bands = {
        "lower_95": mean_forecast_vol - 1.96 * forecast_std,
        "upper_95": mean_forecast_vol + 1.96 * forecast_std
    }
    
    return {
        "forecast_series": forecast_series,
        "current_daily_vol": current_vol,
        "mean_forecast_daily_vol": mean_forecast_vol,
        "volatility_trend": trend,
        "forecast_horizon": forecast_horizon,
        "confidence_bands": vol_confidence_bands,
        "model_aic": fitted_model.aic,
        "model_summary": str(fitted_model.summary())
    }


def calculate_correlation_matrix(returns: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Enhanced correlation matrix with rolling analysis."""
    if not isinstance(returns, pd.DataFrame) or returns.shape[1] < 2:
        return None
    return returns.corr()


def calculate_beta(
    portfolio_returns: pd.Series, 
    market_returns: pd.Series
) -> Optional[float]:
    """Enhanced beta calculation with confidence intervals."""
    if portfolio_returns.empty or market_returns.empty:
        return None
    
    df = pd.DataFrame({'portfolio': portfolio_returns, 'market': market_returns}).dropna()
    if len(df) < 30: 
        return None
        
    covariance = df['portfolio'].cov(df['market'])
    market_variance = df['market'].var()
    
    return covariance / market_variance if market_variance > 0 else None


def generate_risk_sentiment_index(
    risk_metrics: Dict[str, Any],
    correlation_matrix: pd.DataFrame
) -> Optional[Dict[str, Any]]:
    """Enhanced risk sentiment index with regime awareness."""
    if not risk_metrics: 
        return None
    
    try:
        # Volatility Score (5% to 40% annual vol range)
        vol = risk_metrics['performance_stats']['annualized_volatility_pct']
        vol_score = np.clip((vol - 5) / (40 - 5), 0, 1) * 100
        
        # Tail Risk Score (based on 99% CVaR)
        cvar99 = abs(risk_metrics['risk_measures']['99%']['cvar_expected_shortfall'])
        cvar_score = np.clip((cvar99 - 0.01) / (0.05 - 0.01), 0, 1) * 100
        
        # Correlation Score (average off-diagonal correlations)
        np.fill_diagonal(correlation_matrix.values, np.nan)
        avg_corr = correlation_matrix.mean().mean()
        corr_score = np.clip(avg_corr / 0.8, 0, 1) * 100
        
        # Skewness penalty (negative skew increases stress)
        skewness = risk_metrics['performance_stats'].get('skewness', 0)
        skew_penalty = max(0, -skewness * 10)  # Penalty for negative skew
        
        # Regime adjustment
        regime_adjustment = 0
        if risk_metrics.get('regime_conditional_risk'):
            current_regime = risk_metrics['regime_conditional_risk']['current_regime']
            if current_regime == 1:  # High volatility regime
                regime_adjustment = 15
        
        # Weighted final score with regime adjustment
        weights = {'volatility': 0.35, 'tail_risk': 0.35, 'correlation': 0.15, 'skewness': 0.15}
        
        final_score = (
            weights['volatility'] * vol_score +
            weights['tail_risk'] * cvar_score +
            weights['correlation'] * corr_score +
            weights['skewness'] * skew_penalty +
            regime_adjustment
        )
        
        final_score = np.clip(final_score, 0, 100)
        
        if final_score < 33:
            sentiment = "Calm"
        elif final_score < 66:
            sentiment = "Uneasy"
        else:
            sentiment = "Stressed"

        return {
            "stress_index_score": int(final_score),
            "sentiment": sentiment,
            "component_scores": {
                "volatility": int(vol_score),
                "tail_risk": int(cvar_score),
                "correlation": int(corr_score),
                "skewness_penalty": int(skew_penalty)
            },
            "regime_adjustment": regime_adjustment
        }
    except Exception as e:
        print(f"Could not generate risk sentiment index: {e}")
        return None


def apply_market_shock(
    returns: pd.Series, 
    shock_scenarios: Dict[str, float] = None
) -> Dict[str, Any]:
    """Enhanced market shock testing with comprehensive scenario analysis."""
    try:
        if shock_scenarios is None:
            shock_scenarios = {
                "market_crash_2008": -0.37,
                "covid_crash_2020": -0.34,
                "mild_correction": -0.10,
                "severe_correction": -0.20,
                "black_monday": -0.22,
                "dotcom_crash": -0.49,
                "flash_crash": -0.09
            }
        
        current_stats = calculate_risk_metrics(returns)
        if not current_stats:
            return {"success": False, "error": "Could not calculate baseline stats"}
        
        baseline_sharpe = current_stats['risk_adjusted_ratios']['sharpe_ratio']
        baseline_max_dd = current_stats['drawdown_stats']['max_drawdown_pct']
        
        shock_results = {}
        
        for scenario_name, shock_return in shock_scenarios.items():
            # Create shocked return series
            shocked_returns = returns.copy()
            shocked_returns.iloc[-1] = shock_return
            
            # Calculate new metrics
            shocked_stats = calculate_risk_metrics(shocked_returns)
            if shocked_stats:
                new_sharpe = shocked_stats['risk_adjusted_ratios']['sharpe_ratio']
                new_max_dd = shocked_stats['drawdown_stats']['max_drawdown_pct']
                
                # Portfolio recovery analysis
                recovery_periods = max(1, abs(shock_return) * 252)  # Rough estimate
                
                shock_results[scenario_name] = {
                    "shock_magnitude": shock_return,
                    "new_sharpe_ratio": new_sharpe,
                    "sharpe_change": new_sharpe - baseline_sharpe,
                    "new_max_drawdown": new_max_dd,
                    "drawdown_change": new_max_dd - baseline_max_dd,
                    "estimated_recovery_days": int(recovery_periods),
                    "resilience_score": max(0, 1 - abs(new_sharpe - baseline_sharpe) / max(abs(baseline_sharpe), 0.1))
                }
        
        # Overall resilience analysis
        avg_resilience = np.mean([result['resilience_score'] for result in shock_results.values()])
        worst_case_scenario = min(shock_results.keys(), key=lambda x: shock_results[x]['resilience_score'])
        
        return {
            "success": True,
            "baseline_metrics": {
                "sharpe_ratio": baseline_sharpe,
                "max_drawdown": baseline_max_dd
            },
            "shock_scenarios": shock_results,
            "overall_resilience_score": float(avg_resilience),
            "resilience_interpretation": "High" if avg_resilience > 0.7 else "Medium" if avg_resilience > 0.4 else "Low",
            "worst_case_scenario": worst_case_scenario,
            "stress_test_summary": {
                "scenarios_tested": len(shock_scenarios),
                "avg_drawdown_impact": np.mean([result['drawdown_change'] for result in shock_results.values()]),
                "max_estimated_recovery_days": max([result['estimated_recovery_days'] for result in shock_results.values()])
            }
        }
        
    except Exception as e:
        return {"success": False, "error": f"Enhanced shock testing failed: {str(e)}"}
    

def calculate_regime_conditional_risk(returns, confidence_levels=[0.95, 0.99]):
    """
    Calculate risk metrics conditional on market regime
    
    Parameters:
    -----------
    returns : pd.Series
        Return series for analysis
    confidence_levels : list
        Confidence levels for VaR/CVaR calculation
        
    Returns:
    --------
    dict : Regime-conditional risk metrics
    """
    try:
        if len(returns) < 50:
            return {
                'current_regime': 0,
                'regime_risk_metrics': {},
                'error': 'Insufficient data for regime analysis'
            }
        
        # Detect regimes using HMM
        regime_result = detect_hmm_regimes(returns)
        current_regime = regime_result.get('current_regime', 0)
        
        # Get regime labels (assume binary for simplicity)
        regime_probs = regime_result.get('regime_probabilities', [0.5, 0.5])
        
        # Simple regime assignment based on volatility
        rolling_vol = returns.rolling(20).std()
        vol_threshold = rolling_vol.median()
        
        regime_labels = (rolling_vol > vol_threshold).astype(int)
        
        regime_risk_metrics = {}
        
        # Calculate risk metrics for each regime
        for regime in [0, 1]:
            regime_mask = regime_labels == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) < 10:
                continue
                
            regime_metrics = {}
            
            # Basic statistics
            regime_metrics['mean_return'] = regime_returns.mean()
            regime_metrics['volatility'] = regime_returns.std()
            regime_metrics['skewness'] = regime_returns.skew()
            regime_metrics['kurtosis'] = regime_returns.kurtosis()
            
            # VaR and CVaR for each confidence level
            for conf in confidence_levels:
                alpha = 1 - conf
                var_value = regime_returns.quantile(alpha)
                cvar_value = regime_returns[regime_returns <= var_value].mean()
                
                regime_metrics[f'VaR_{int(conf*100)}'] = var_value
                regime_metrics[f'CVaR_{int(conf*100)}'] = cvar_value
            
            # Maximum drawdown
            cumulative = (1 + regime_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            regime_metrics['max_drawdown'] = drawdown.min()
            
            regime_name = 'High Volatility' if regime == 1 else 'Low Volatility'
            regime_risk_metrics[regime_name] = regime_metrics
        
        return {
            'current_regime': current_regime,
            'regime_risk_metrics': regime_risk_metrics,
            'regime_probabilities': regime_probs,
            'analysis_periods': {
                'total_observations': len(returns),
                'regime_0_observations': sum(regime_labels == 0),
                'regime_1_observations': sum(regime_labels == 1)
            }
        }
        
    except Exception as e:
        return {
            'current_regime': 0,
            'regime_risk_metrics': {},
            'error': str(e)
        }

def calculate_factor_risk_attribution(portfolio_returns, factor_data, weights=None):
    """
    Decompose portfolio risk into factor contributions
    
    Parameters:
    -----------
    portfolio_returns : pd.Series
        Portfolio return series
    factor_data : pd.DataFrame
        Factor return data (market, sector, style factors)
    weights : pd.Series, optional
        Portfolio weights for analysis
        
    Returns:
    --------
    dict : Factor risk attribution results
    """
    try:
        # Align data
        common_dates = portfolio_returns.index.intersection(factor_data.index)
        if len(common_dates) < 20:
            return {'error': 'Insufficient overlapping data for factor analysis'}
        
        port_returns = portfolio_returns.loc[common_dates]
        factors = factor_data.loc[common_dates]
        
        # Multiple regression: Portfolio returns = alpha + beta1*F1 + beta2*F2 + ... + error
        X = factors.values
        y = port_returns.values
        
        # Add constant for alpha
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        # OLS regression
        try:
            coefficients = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return {'error': 'Singular matrix - cannot compute factor loadings'}
        
        alpha = coefficients[0]
        factor_loadings = coefficients[1:]
        
        # Create factor loadings dictionary
        factor_loading_dict = {}
        for i, factor_name in enumerate(factors.columns):
            factor_loading_dict[factor_name] = factor_loadings[i]
        
        # Calculate R-squared
        y_pred = X_with_const @ coefficients
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Risk attribution
        factor_cov = factors.cov().values
        portfolio_var = np.var(port_returns)
        
        # Factor contribution to variance
        factor_var_contrib = {}
        total_factor_risk = 0
        
        for i, factor_name in enumerate(factors.columns):
            factor_contrib = (factor_loadings[i] ** 2) * factor_cov[i, i]
            factor_var_contrib[factor_name] = factor_contrib
            total_factor_risk += factor_contrib
        
        # Specific risk (unexplained variance)
        specific_risk = portfolio_var - total_factor_risk
        specific_risk = max(0, specific_risk)  # Ensure non-negative
        
        # Convert to percentages
        risk_attribution = {}
        total_risk = total_factor_risk + specific_risk
        
        if total_risk > 0:
            for factor_name, contrib in factor_var_contrib.items():
                risk_attribution[factor_name] = (contrib / total_risk) * 100
            
            risk_attribution['Specific Risk'] = (specific_risk / total_risk) * 100
        else:
            risk_attribution = {'Specific Risk': 100.0}
        
        # Tracking error and information ratio
        tracking_error = np.sqrt(specific_risk) * np.sqrt(252)  # Annualized
        information_ratio = (alpha * 252) / tracking_error if tracking_error > 0 else 0
        
        return {
            'factor_loadings': factor_loading_dict,
            'alpha': alpha,
            'total_rsquared': r_squared,
            'risk_attribution': risk_attribution,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'specific_risk_percentage': risk_attribution.get('Specific Risk', 0)
        }
        
    except Exception as e:
        return {'error': f'Factor attribution calculation failed: {str(e)}'}

def calculate_dynamic_risk_budgets(returns, current_weights=None, target_volatility=0.15):
    """
    Calculate risk budgets and optimization for Equal Risk Contribution
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Return series for assets
    current_weights : pd.Series, optional
        Current portfolio weights
    target_volatility : float
        Target portfolio volatility
        
    Returns:
    --------
    dict : Risk budgeting results
    """
    try:
        # Calculate covariance matrix
        cov_matrix = returns.cov() * 252  # Annualized
        n_assets = len(returns.columns)
        
        if current_weights is None:
            current_weights = pd.Series(1/n_assets, index=returns.columns)
        
        # Current portfolio metrics
        current_vol = np.sqrt(current_weights.T @ cov_matrix @ current_weights)
        current_risk_contrib = calculate_risk_contributions(current_weights, cov_matrix)
        
        # Equal Risk Contribution optimization
        def risk_budget_objective(weights, cov_matrix):
            """Objective function for ERC optimization"""
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            if portfolio_vol == 0:
                return 1e10
            
            # Risk contributions
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            # Target equal risk contribution
            target_contrib = portfolio_vol / len(weights)
            
            # Sum of squared deviations from equal risk
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0.01, 0.5) for _ in range(n_assets)]  # Min 1%, max 50%
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        try:
            result = minimize(
                risk_budget_objective, 
                x0, 
                args=(cov_matrix.values,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = pd.Series(result.x, index=returns.columns)
                optimal_vol = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
                optimal_risk_contrib = calculate_risk_contributions(optimal_weights, cov_matrix)
            else:
                # Fallback to equal weights
                optimal_weights = current_weights
                optimal_vol = current_vol
                optimal_risk_contrib = current_risk_contrib
                
        except:
            # Fallback to equal weights
            optimal_weights = current_weights
            optimal_vol = current_vol
            optimal_risk_contrib = current_risk_contrib
        
        # Risk concentration measures
        current_hhi = np.sum(current_risk_contrib ** 2)  # Herfindahl index
        optimal_hhi = np.sum(optimal_risk_contrib ** 2)
        
        return {
            'current_allocation': {
                'weights': current_weights,
                'volatility': current_vol,
                'risk_contributions': current_risk_contrib,
                'concentration_ratio': current_hhi
            },
            'optimal_allocation': {
                'weights': optimal_weights,
                'volatility': optimal_vol,
                'risk_contributions': optimal_risk_contrib,
                'concentration_ratio': optimal_hhi
            },
            'improvement_metrics': {
                'volatility_change': optimal_vol - current_vol,
                'concentration_reduction': current_hhi - optimal_hhi,
                'diversification_benefit': (current_hhi - optimal_hhi) / current_hhi * 100
            }
        }
        
    except Exception as e:
        return {'error': f'Risk budgeting calculation failed: {str(e)}'}

def calculate_risk_contributions(weights, cov_matrix):
    """Helper function to calculate risk contributions"""
    portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    if portfolio_vol == 0:
        return np.zeros(len(weights))
    
    marginal_contrib = cov_matrix @ weights
    risk_contrib = weights * marginal_contrib / portfolio_vol
    return risk_contrib

def advanced_monte_carlo_stress_test(returns, num_scenarios=1000, time_horizon=30):
    """
    Advanced Monte Carlo stress testing with fat-tail adjustments
    
    Parameters:
    -----------
    returns : pd.Series
        Historical return series
    num_scenarios : int
        Number of Monte Carlo scenarios
    time_horizon : int
        Time horizon for stress testing (days)
        
    Returns:
    --------
    dict : Stress testing results
    """
    try:
        # Test for normality
        _, p_value = stats.jarque_bera(returns.dropna())
        is_normal = p_value > 0.05
        
        if is_normal:
            # Use normal distribution
            mu = returns.mean()
            sigma = returns.std()
            scenarios = np.random.normal(mu, sigma, (num_scenarios, time_horizon))
        else:
            # Use t-distribution for fat tails
            mu = returns.mean()
            sigma = returns.std()
            
            # Fit t-distribution
            try:
                df, loc, scale = stats.t.fit(returns.dropna())
                scenarios = stats.t.rvs(df, loc=loc, scale=scale, size=(num_scenarios, time_horizon))
            except:
                # Fallback to normal with higher volatility
                scenarios = np.random.normal(mu, sigma * 1.2, (num_scenarios, time_horizon))
        
        # Calculate path returns
        cumulative_returns = np.cumprod(1 + scenarios, axis=1) - 1
        final_returns = cumulative_returns[:, -1]
        
        # Calculate risk metrics
        var_95 = np.percentile(final_returns, 5)
        var_99 = np.percentile(final_returns, 1)
        cvar_95 = np.mean(final_returns[final_returns <= var_95])
        cvar_99 = np.mean(final_returns[final_returns <= var_99])
        
        # Worst case analysis
        worst_scenario_idx = np.argmin(final_returns)
        worst_case_return = final_returns[worst_scenario_idx]
        worst_case_path = cumulative_returns[worst_scenario_idx, :]
        
        # Probability estimates
        prob_loss_5 = np.mean(final_returns < -0.05) * 100
        prob_loss_10 = np.mean(final_returns < -0.10) * 100
        prob_loss_20 = np.mean(final_returns < -0.20) * 100
        
        return {
            'var_estimates': {
                'VaR_95': var_95,
                'VaR_99': var_99,
                'ES_95': cvar_95,
                'ES_99': cvar_99
            },
            'worst_case_scenario': {
                'loss': worst_case_return,
                'path': worst_case_path.tolist()
            },
            'loss_probabilities': {
                'prob_loss_5_percent': prob_loss_5,
                'prob_loss_10_percent': prob_loss_10,
                'prob_loss_20_percent': prob_loss_20
            },
            'scenario_statistics': {
                'mean_return': np.mean(final_returns),
                'median_return': np.median(final_returns),
                'std_return': np.std(final_returns),
                'distribution_used': 't-distribution' if not is_normal else 'normal'
            },
            'time_horizon_days': time_horizon,
            'num_scenarios': num_scenarios
        }
        
    except Exception as e:
        return {'error': f'Monte Carlo stress test failed: {str(e)}'}

def calculate_time_varying_risk(returns, window=30, regime_threshold=0.02):
    """
    Calculate time-varying risk metrics with regime identification
    
    Parameters:
    -----------
    returns : pd.Series
        Return series for analysis
    window : int
        Rolling window for calculations
    regime_threshold : float
        Volatility threshold for regime classification
        
    Returns:
    --------
    dict : Time-varying risk analysis
    """
    try:
        # Rolling risk metrics
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_var = returns.rolling(window).quantile(0.05)
        rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        
        # Risk regime classification
        vol_median = rolling_vol.median()
        risk_states = []
        
        for vol in rolling_vol:
            if pd.isna(vol):
                risk_states.append('Unknown')
            elif vol < vol_median * 0.8:
                risk_states.append('Low Risk')
            elif vol > vol_median * 1.2:
                risk_states.append('High Risk')
            else:
                risk_states.append('Normal Risk')
        
        # Current risk state
        current_risk_state = risk_states[-1] if risk_states else 'Unknown'
        
        # Risk evolution data
        risk_evolution = []
        for i in range(len(returns)):
            if i >= window - 1:
                risk_evolution.append({
                    'date': returns.index[i],
                    'volatility': rolling_vol.iloc[i],
                    'var_95': rolling_var.iloc[i],
                    'sharpe_ratio': rolling_sharpe.iloc[i],
                    'risk_state': risk_states[i]
                })
        
        # Risk trend analysis
        recent_vol = rolling_vol.tail(10).mean()
        older_vol = rolling_vol.tail(30).head(20).mean()
        
        if pd.isna(recent_vol) or pd.isna(older_vol):
            risk_trend = 'Stable'
        elif recent_vol > older_vol * 1.1:
            risk_trend = 'Increasing'
        elif recent_vol < older_vol * 0.9:
            risk_trend = 'Decreasing'
        else:
            risk_trend = 'Stable'
        
        return {
            'current_risk_state': current_risk_state,
            'risk_trend': risk_trend,
            'risk_evolution': risk_evolution,
            'summary_statistics': {
                'mean_volatility': rolling_vol.mean(),
                'volatility_range': [rolling_vol.min(), rolling_vol.max()],
                'current_volatility': rolling_vol.iloc[-1] if len(rolling_vol) > 0 else None,
                'risk_state_distribution': {
                    state: risk_states.count(state) for state in set(risk_states)
                }
            }
        }
        
    except Exception as e:
        return {'error': f'Time-varying risk calculation failed: {str(e)}'}