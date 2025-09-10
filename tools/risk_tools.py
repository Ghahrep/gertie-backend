# tools/risk_tools.py - Clean Financial Risk Analysis
"""
Financial Risk Analysis Module - Clean Version
==============================================

Production-grade portfolio risk analysis tools stripped of all agent complexity.
Direct function calls for your new clean backend architecture.
"""

import pandas as pd
import numpy as np
from arch import arch_model
from scipy import stats
from typing import List, Dict, Any, Optional
from datetime import datetime
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


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
    """
    Calculates comprehensive risk and return metrics for a portfolio.
    """
    if not isinstance(portfolio_returns, pd.Series) or portfolio_returns.empty:
        return None
        
    try:
        # VaR and CVaR Analysis
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
        
        # Integrate drawdown stats
        drawdown_stats = calculate_drawdowns(portfolio_returns)

        return {
            "risk_measures": var_analysis,
            "performance_stats": {
                "annualized_return_pct": annual_return * 100,
                "annualized_volatility_pct": annual_vol * 100,
            },
            "risk_adjusted_ratios": {
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": drawdown_stats['calmar_ratio'] if drawdown_stats else None
            },
            "drawdown_stats": drawdown_stats
        }
    except Exception as e:
        print(f"Error calculating risk metrics: {e}")
        return None


def calculate_drawdowns(returns: pd.Series) -> Optional[Dict[str, Any]]:
    """
    Analyze historical portfolio drawdowns.
    """
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
    else:
        start_date_str = f"Day {start_date_idx}"
        end_date_str = f"Day {end_date_idx}"

    annual_return = returns.mean() * 252
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
    
    return {
        "max_drawdown_pct": max_drawdown * 100,
        "start_of_max_drawdown": start_date_str,
        "end_of_max_drawdown": end_date_str,
        "current_drawdown_pct": drawdowns.iloc[-1] * 100,
        "calmar_ratio": calmar_ratio
    }


def fit_garch_forecast(
    returns: pd.Series, 
    forecast_horizon: int = 30
) -> Optional[Dict[str, Any]]:
    """
    Fit GARCH(1,1) model and generate volatility forecasts.
    """
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
    
    return {
        "forecast_series": forecast_series,
        "current_daily_vol": current_vol,
        "mean_forecast_daily_vol": mean_forecast_vol,
        "volatility_trend": trend,
        "forecast_horizon": forecast_horizon,
        "model_summary": str(fitted_model.summary())
    }


def calculate_volatility_budget(
    portfolio_returns: pd.Series, 
    target_volatility: float,
    trading_days: int = 252
) -> Optional[Dict[str, float]]:
    """
    Calculate allocation between risky portfolio and risk-free asset
    to achieve target volatility level.
    """
    if not isinstance(portfolio_returns, pd.Series) or portfolio_returns.empty:
        return None
        
    try:
        current_vol = portfolio_returns.std() * np.sqrt(trading_days)
        
        if current_vol == 0:
            return {'risky_asset_weight': 0.0, 'risk_free_asset_weight': 1.0}

        weight_risky = target_volatility / current_vol
        weight_risky = np.clip(weight_risky, 0, 1.5)  # Cap at 50% leverage
        weight_risk_free = 1 - weight_risky
        
        return {
            'risky_asset_weight': round(weight_risky, 4),
            'risk_free_asset_weight': round(weight_risk_free, 4),
            'current_annual_volatility': round(current_vol, 4),
            'target_annual_volatility': round(target_volatility, 4)
        }
    except Exception as e:
        print(f"Error calculating volatility budget: {e}")
        return None


def calculate_correlation_matrix(returns: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Calculate correlation matrix for multiple assets."""
    if not isinstance(returns, pd.DataFrame) or returns.shape[1] < 2:
        return None
    return returns.corr()


def calculate_beta(
    portfolio_returns: pd.Series, 
    market_returns: pd.Series
) -> Optional[float]:
    """Calculate portfolio beta relative to market returns."""
    if portfolio_returns.empty or market_returns.empty:
        return None
    
    df = pd.DataFrame({'portfolio': portfolio_returns, 'market': market_returns}).dropna()
    if len(df) < 30: 
        return None
        
    covariance = df['portfolio'].cov(df['market'])
    market_variance = df['market'].var()
    
    return covariance / market_variance if market_variance > 0 else None


def calculate_tail_risk_copula(
    returns: pd.DataFrame, 
    n_simulations: int = 10000
) -> Optional[pd.DataFrame]:
    """
    Stress test using Student's t-copula for fat tail simulation.
    """
    if not isinstance(returns, pd.DataFrame) or returns.shape[0] < 100:
        print("Error: Requires at least 100 data points for reliable copula fitting.")
        return None

    try:
        # Fit marginal distributions (Student's t) to each asset
        fitted_marginals = {
            asset: stats.t.fit(returns[asset].dropna())
            for asset in returns.columns
        }

        # Transform to uniform distributions
        uniform_returns = pd.DataFrame({
            asset: stats.t.cdf(returns[asset].dropna(), *params)
            for asset, params in fitted_marginals.items()
        })

        # Fit t-copula
        spearman_corr = uniform_returns.corr(method='spearman')
        copula_df = 5  # Lower DF for fatter tails

        # Simulate from copula
        n_assets = len(returns.columns)
        mvt_rvs = stats.multivariate_t.rvs(
            loc=np.zeros(n_assets), 
            shape=spearman_corr.values, 
            df=copula_df, 
            size=n_simulations
        )
        copula_sims_uniform = stats.t.cdf(mvt_rvs, df=copula_df)
        
        # Transform back to asset returns
        simulated_returns = pd.DataFrame({
            asset: stats.t.ppf(copula_sims_uniform[:, i], *fitted_marginals[asset])
            for i, asset in enumerate(returns.columns)
        })

        return simulated_returns
    except Exception as e:
        print(f"Error during copula stress testing: {e}")
        return None


def generate_risk_sentiment_index(
    risk_metrics: Dict[str, Any],
    correlation_matrix: pd.DataFrame
) -> Optional[Dict[str, Any]]:
    """
    Proprietary "Portfolio Stress Sentiment Index" (0-100 scale).
    """
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
        
        # Weighted final score
        weights = {'volatility': 0.4, 'tail_risk': 0.4, 'correlation': 0.2}
        
        final_score = (
            weights['volatility'] * vol_score +
            weights['tail_risk'] * cvar_score +
            weights['correlation'] * corr_score
        )
        
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
                "correlation": int(corr_score)
            }
        }
    except Exception as e:
        print(f"Could not generate risk sentiment index: {e}")
        return None
    

def generate_risk_sentiment_index(
    risk_metrics: Dict[str, Any],
    correlation_matrix: pd.DataFrame
) -> Optional[Dict[str, Any]]:
    """
    Proprietary "Portfolio Stress Sentiment Index" (0-100 scale).
    """
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
        
        # Weighted final score
        weights = {'volatility': 0.4, 'tail_risk': 0.4, 'correlation': 0.2}
        
        final_score = (
            weights['volatility'] * vol_score +
            weights['tail_risk'] * cvar_score +
            weights['correlation'] * corr_score
        )
        
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
                "correlation": int(corr_score)
            }
        }
    except Exception as e:
        print(f"Could not generate risk sentiment index: {e}")
        return None


def apply_market_shock(
    returns: pd.Series, 
    shock_scenarios: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Apply market shock scenarios to test portfolio resilience.
    
    Parameters:
    -----------
    returns : pd.Series
        Historical returns series
    shock_scenarios : Dict[str, float], optional
        Custom shock scenarios. Default includes standard stress tests.
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with shock test results
    """
    try:
        if shock_scenarios is None:
            shock_scenarios = {
                "market_crash_2008": -0.37,  # S&P 500 in 2008
                "covid_crash_2020": -0.34,   # March 2020 crash
                "mild_correction": -0.10,     # 10% correction
                "severe_correction": -0.20,   # 20% correction
                "black_monday": -0.22         # October 1987
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
            shocked_returns.iloc[-1] = shock_return  # Apply shock to last period
            
            # Calculate new metrics
            shocked_stats = calculate_risk_metrics(shocked_returns)
            if shocked_stats:
                new_sharpe = shocked_stats['risk_adjusted_ratios']['sharpe_ratio']
                new_max_dd = shocked_stats['drawdown_stats']['max_drawdown_pct']
                
                shock_results[scenario_name] = {
                    "shock_magnitude": shock_return,
                    "new_sharpe_ratio": new_sharpe,
                    "sharpe_change": new_sharpe - baseline_sharpe,
                    "new_max_drawdown": new_max_dd,
                    "drawdown_change": new_max_dd - baseline_max_dd,
                    "resilience_score": max(0, 1 - abs(new_sharpe - baseline_sharpe) / max(abs(baseline_sharpe), 0.1))
                }
        
        # Calculate overall resilience
        avg_resilience = np.mean([result['resilience_score'] for result in shock_results.values()])
        
        return {
            "success": True,
            "baseline_metrics": {
                "sharpe_ratio": baseline_sharpe,
                "max_drawdown": baseline_max_dd
            },
            "shock_scenarios": shock_results,
            "overall_resilience_score": float(avg_resilience),
            "resilience_interpretation": "High" if avg_resilience > 0.7 else "Medium" if avg_resilience > 0.4 else "Low"
        }
        
    except Exception as e:
        return {"success": False, "error": f"Shock testing failed: {str(e)}"}