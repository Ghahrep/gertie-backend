# tools/portfolio_tools.py - Clean Portfolio Optimization
"""
Portfolio Strategy & Optimization Module - Clean Version
========================================================

Production-grade portfolio optimization and strategy tools.
Direct function calls with no agent dependencies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# Core Library Imports with fallbacks
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.hierarchical_portfolio import HRPOpt
    PYPFOPT_AVAILABLE = True
except ImportError:
    print("Warning: pypfopt not found. pip install PyPortfolioOpt")
    PYPFOPT_AVAILABLE = False

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("Warning: statsmodels not found. pip install statsmodels")
    STATSMODELS_AVAILABLE = False

try:
    from backtesting import Backtest, Strategy
    BACKTESTING_AVAILABLE = True
except ImportError:
    print("Warning: backtesting.py not found. pip install backtesting")
    BACKTESTING_AVAILABLE = False


def optimize_portfolio(
    asset_prices: pd.DataFrame, 
    objective: str = 'MaximizeSharpe',
    risk_free_rate: float = 0.02
) -> Optional[Dict[str, Any]]:
    """
    Construct optimal portfolio using PyPortfolioOpt.
    Objectives: 'MaximizeSharpe', 'MinimizeVolatility', 'HERC'.
    """
    if not PYPFOPT_AVAILABLE:
        raise ImportError("PyPortfolioOpt is not installed.")
    if asset_prices.empty: 
        return None

    returns = expected_returns.returns_from_prices(asset_prices)

    try:
        if objective == 'HERC':
            # Hierarchical Risk Parity
            hrp = HRPOpt(returns)
            hrp.optimize()
            clean_weights = hrp.clean_weights()
            
            perf = hrp.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            expected_return, annual_vol, sharpe = perf

        elif objective in ['MaximizeSharpe', 'MinimizeVolatility']:
            # Mean-Variance Optimization
            mu = expected_returns.mean_historical_return(asset_prices)
            S = risk_models.sample_cov(asset_prices)
            ef = EfficientFrontier(mu, S)
            
            if objective == 'MaximizeSharpe':
                ef.max_sharpe(risk_free_rate=risk_free_rate)
            elif objective == 'MinimizeVolatility':
                ef.min_volatility()
                
            clean_weights = ef.clean_weights()
            expected_return, annual_vol, sharpe = ef.portfolio_performance(
                verbose=False, risk_free_rate=risk_free_rate
            )
        
        else:
            raise ValueError(f"Objective '{objective}' not recognized.")
            
        return {
            "objective_used": objective,
            "optimal_weights": clean_weights,
            "expected_performance": {
                "annual_return_pct": expected_return * 100,
                "annual_volatility_pct": annual_vol * 100,
                "sharpe_ratio": sharpe
            }
        }
    except Exception as e:
        print(f"Portfolio optimization failed for objective '{objective}': {e}")
        return None


def generate_trade_orders(
    current_holdings: Dict[str, float], 
    target_weights: Dict[str, float], 
    total_portfolio_value: float,
    min_trade_amount: float = 100.0
) -> Dict[str, Any]:
    """
    Generate trade orders to rebalance portfolio to target weights.
    
    Parameters:
    -----------
    current_holdings : Dict[str, float]
        Current holdings in dollar amounts {ticker: value}
    target_weights : Dict[str, float] 
        Target allocation weights {ticker: weight} where weights sum to 1.0
    total_portfolio_value : float
        Total portfolio value for rebalancing
    min_trade_amount : float, default=100.0
        Minimum trade size to avoid small transactions
        
    Returns:
    --------
    Dict[str, Any]
        Trade orders and rebalancing summary
    """
    try:
        if not target_weights or sum(target_weights.values()) == 0:
            return {
                "success": False,
                "error": "Invalid target weights",
                "trades": [],
                "summary": {}
            }
        
        # Normalize weights to sum to 1.0
        weight_sum = sum(target_weights.values())
        normalized_weights = {k: v/weight_sum for k, v in target_weights.items()}
        
        trades = []
        rebalancing_summary = {
            "total_portfolio_value": total_portfolio_value,
            "current_allocation": {},
            "target_allocation": {},
            "rebalancing_needed": False
        }
        
        # Calculate current allocation percentages
        for ticker, value in current_holdings.items():
            current_weight = value / total_portfolio_value if total_portfolio_value > 0 else 0
            rebalancing_summary["current_allocation"][ticker] = {
                "value": value,
                "weight_pct": current_weight * 100
            }
        
        # Calculate target allocation and required trades
        total_trade_value = 0
        
        for ticker, target_weight in normalized_weights.items():
            target_value = target_weight * total_portfolio_value
            current_value = current_holdings.get(ticker, 0.0)
            trade_amount = target_value - current_value
            
            rebalancing_summary["target_allocation"][ticker] = {
                "target_value": target_value,
                "target_weight_pct": target_weight * 100
            }
            
            # Only create trade if amount is significant
            if abs(trade_amount) >= min_trade_amount:
                action = "BUY" if trade_amount > 0 else "SELL"
                
                trades.append({
                    "ticker": ticker,
                    "action": action,
                    "amount_usd": abs(trade_amount),
                    "current_value": current_value,
                    "target_value": target_value,
                    "weight_change": target_weight - (current_value / total_portfolio_value if total_portfolio_value > 0 else 0)
                })
                
                total_trade_value += abs(trade_amount)
                rebalancing_summary["rebalancing_needed"] = True
        
        # Calculate trading costs and efficiency
        estimated_cost = total_trade_value * 0.001  # 0.1% trading cost assumption
        efficiency_score = (total_trade_value / total_portfolio_value) if total_portfolio_value > 0 else 0
        
        rebalancing_summary.update({
            "total_trades": len(trades),
            "total_trade_value": total_trade_value,
            "estimated_trading_cost": estimated_cost,
            "efficiency_score": efficiency_score,
            "recommendation": "Execute trades" if efficiency_score > 0.02 else "Consider delaying - minimal rebalancing benefit"
        })
        
        return {
            "success": True,
            "trades": trades,
            "summary": rebalancing_summary,
            "execution_ready": len(trades) > 0
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Trade generation failed: {str(e)}",
            "trades": [],
            "summary": {"error": str(e)}
        }


def perform_factor_analysis(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame
) -> Optional[Dict[str, float]]:
    """
    Perform factor analysis on portfolio returns using specified factors.
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is not installed.")
    
    try:
        # Align data and calculate portfolio excess returns
        data = pd.DataFrame({'portfolio': portfolio_returns}).join(factor_returns).dropna()
        if 'RF' not in data.columns: 
            raise ValueError("factor_returns must include 'RF' column.")
        
        y = data['portfolio'] - data['RF']
        X = data.drop(columns=['portfolio', 'RF'])
        X = sm.add_constant(X)  # Add constant for alpha
        
        # Fit OLS model
        model = sm.OLS(y, X).fit()
        
        # Extract results
        params = model.params.to_dict()
        alpha_daily = params.pop('const')
        alpha_annual_pct = alpha_daily * 252 * 100
        
        return {
            "alpha_annual_pct": alpha_annual_pct,
            "factor_betas": params,
            "r_squared_adj": model.rsquared_adj,
            "p_values": model.pvalues.to_dict()
        }
    except Exception as e:
        print(f"Factor analysis failed: {e}")
        return None


def run_backtest(
    prices: pd.DataFrame,
    strategy_signals: pd.Series,
    initial_cash: float = 100_000,
    commission: float = 0.001
) -> Optional[Dict[str, Any]]:
    """
    Run backtest with pre-calculated strategy signals.
    """
    if not BACKTESTING_AVAILABLE:
        raise ImportError("backtesting.py is not installed.")

    data = prices.copy()
    data['Signal'] = strategy_signals.reindex(data.index, method='ffill').fillna(0)

    class SignalFollowingStrategy(Strategy):
        def init(self): 
            pass
            
        def next(self):
            signal = self.data.Signal[-1]
            if signal == 1 and not self.position:
                self.buy()
            elif signal == 0 and self.position:
                self.position.close()

    try:
        bt = Backtest(data, SignalFollowingStrategy, cash=initial_cash, commission=commission)
        stats = bt.run()
        
        # Build clean result dictionary (avoid non-serializable objects)
        performance_summary = {
            key: value for key, value in stats.items()
            if key not in ['_strategy', '_equity_curve', '_trades']
        }
        
        equity_curve_json = stats._equity_curve.reset_index().to_json(
            orient='records', date_format='iso'
        )
        trades_json = stats._trades.reset_index().to_json(
            orient='records', date_format='iso'
        )

        return {
            "performance_summary": performance_summary,
            "equity_curve_json": equity_curve_json,
            "trades_json": trades_json
        }
    except Exception as e:
        print(f"Backtest failed: {e}")
        return None


def find_optimal_hedge(
    portfolio_returns: pd.Series, 
    hedge_instrument_returns: pd.Series
) -> Optional[Dict[str, float]]:
    """
    Calculate optimal hedge ratio to minimize portfolio variance.
    """
    if portfolio_returns.empty or hedge_instrument_returns.empty:
        return None
        
    try:
        df = pd.DataFrame({
            'portfolio': portfolio_returns, 
            'hedge': hedge_instrument_returns
        }).dropna()
        
        if len(df) < 30:
            print("Warning: Insufficient overlapping data for reliable hedge ratio.")
            return None
            
        covariance = df['portfolio'].cov(df['hedge'])
        hedge_variance = df['hedge'].var()
        
        if hedge_variance == 0:
            return None

        hedge_ratio = -covariance / hedge_variance
        
        original_vol = df['portfolio'].std()
        hedged_portfolio_returns = df['portfolio'] + hedge_ratio * df['hedge']
        hedged_vol = hedged_portfolio_returns.std()
        vol_reduction = original_vol - hedged_vol
        
        return {
            'optimal_hedge_ratio': round(hedge_ratio, 4),
            'original_daily_vol': round(original_vol, 5),
            'hedged_daily_vol': round(hedged_vol, 5),
            'volatility_reduction_pct': round((vol_reduction / original_vol) * 100, 2) if original_vol > 0 else 0
        }
    except Exception as e:
        print(f"Error finding optimal hedge: {e}")
        return None
    

def calculate_portfolio_summary(portfolio_holdings: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate comprehensive portfolio summary statistics.
    
    Parameters:
    -----------
    portfolio_holdings : Dict[str, float]
        Dictionary mapping ticker symbols to dollar values
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with portfolio summary metrics
    """
    try:
        if not portfolio_holdings:
            return {
                "success": False,
                "error": "No portfolio holdings provided",
                "total_value": 0.0,
                "holdings_count": 0,
                "holdings": []
            }
        
        # Calculate basic metrics
        total_value = sum(portfolio_holdings.values())
        holdings_count = len(portfolio_holdings)
        
        # Calculate weights and detailed holdings info
        holdings_details = []
        for ticker, value in portfolio_holdings.items():
            weight = value / total_value if total_value > 0 else 0
            holdings_details.append({
                "ticker": ticker,
                "value": value,
                "weight": weight,
                "weight_pct": weight * 100
            })
        
        # Sort by value (largest first)
        holdings_details.sort(key=lambda x: x["value"], reverse=True)
        
        # Calculate concentration metrics
        top_5_weight = sum(h["weight"] for h in holdings_details[:5])
        top_10_weight = sum(h["weight"] for h in holdings_details[:10])
        
        # Calculate diversification metrics
        weights = np.array([h["weight"] for h in holdings_details])
        herfindahl_index = np.sum(weights**2)
        effective_number_stocks = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        return {
            "success": True,
            "name": "Portfolio",  # Default name
            "total_value": total_value,
            "holdings_count": holdings_count,
            "holdings": holdings_details,
            "concentration_metrics": {
                "top_5_weight_pct": top_5_weight * 100,
                "top_10_weight_pct": top_10_weight * 100,
                "largest_position_pct": holdings_details[0]["weight_pct"] if holdings_details else 0,
                "herfindahl_index": herfindahl_index,
                "effective_number_stocks": effective_number_stocks
            },
            "diversification_score": min(100, effective_number_stocks * 10)  # Simple score out of 100
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Portfolio summary calculation failed: {str(e)}",
            "total_value": 0.0,
            "holdings_count": 0,
            "holdings": []
        }
    
def rebalance_portfolio(
    current_holdings: Dict[str, float],
    target_weights: Dict[str, float],
    rebalance_threshold: float = 0.05,
    cash_available: float = 0.0
) -> Dict[str, Any]:
    """
    Generate rebalancing trades to bring portfolio back to target allocation.
    
    Parameters:
    -----------
    current_holdings : Dict[str, float]
        Current portfolio holdings (ticker -> dollar value)
    target_weights : Dict[str, float]
        Target allocation weights (ticker -> weight between 0 and 1)
    rebalance_threshold : float, default=0.05
        Minimum weight deviation to trigger rebalancing (5%)
    cash_available : float, default=0.0
        Additional cash available for investment
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with rebalancing recommendations
    """
    try:
        total_current_value = sum(current_holdings.values()) + cash_available
        
        if total_current_value <= 0:
            return {
                "success": False,
                "error": "No portfolio value to rebalance",
                "trades": [],
                "rebalancing_needed": False
            }
        
        # Calculate current weights
        current_weights = {}
        for ticker, value in current_holdings.items():
            current_weights[ticker] = value / total_current_value
        
        # Check if rebalancing is needed
        rebalancing_needed = False
        weight_deviations = {}
        
        for ticker in target_weights:
            current_weight = current_weights.get(ticker, 0.0)
            target_weight = target_weights[ticker]
            deviation = abs(current_weight - target_weight)
            weight_deviations[ticker] = deviation
            
            if deviation > rebalance_threshold:
                rebalancing_needed = True
        
        if not rebalancing_needed:
            return {
                "success": True,
                "rebalancing_needed": False,
                "message": "Portfolio is within rebalancing thresholds",
                "trades": [],
                "weight_deviations": weight_deviations
            }
        
        # Generate trades to reach target allocation
        trades = []
        total_target_value = total_current_value
        
        for ticker, target_weight in target_weights.items():
            current_value = current_holdings.get(ticker, 0.0)
            target_value = target_weight * total_target_value
            trade_amount = target_value - current_value
            
            if abs(trade_amount) > total_target_value * rebalance_threshold:
                action = "BUY" if trade_amount > 0 else "SELL"
                trades.append({
                    "ticker": ticker,
                    "action": action,
                    "amount_usd": abs(trade_amount),
                    "current_weight": current_weights.get(ticker, 0.0),
                    "target_weight": target_weight,
                    "weight_deviation": weight_deviations[ticker]
                })
        
        # Calculate post-rebalancing metrics
        total_trade_value = sum(trade["amount_usd"] for trade in trades)
        estimated_cost = total_trade_value * 0.001  # Assume 0.1% trading cost
        
        return {
            "success": True,
            "rebalancing_needed": True,
            "trades": trades,
            "rebalancing_summary": {
                "total_portfolio_value": total_current_value,
                "number_of_trades": len(trades),
                "total_trade_value": total_trade_value,
                "estimated_trading_cost": estimated_cost,
                "net_benefit_threshold": estimated_cost * 2  # Should save 2x trading costs
            },
            "weight_deviations": weight_deviations,
            "recommendation": "Proceed with rebalancing" if total_trade_value > estimated_cost * 2 else "Consider skipping - low benefit vs cost"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Portfolio rebalancing failed: {str(e)}",
            "trades": [],
            "rebalancing_needed": False
        }
    
# Add this function to your tools/portfolio_tools.py file

def calculate_portfolio_summary(portfolio_holdings: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate comprehensive portfolio summary statistics.
    
    Parameters:
    -----------T
    portfolio_holdings : Dict[str, float]
        Dictionary mapping ticker symbols to dollar values
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with portfolio summary metrics
    """
    try:
        if not portfolio_holdings:
            return {
                "success": False,
                "error": "No portfolio holdings provided",
                "total_value": 0.0,
                "holdings_count": 0,
                "holdings": []
            }
        
        # Calculate basic metrics
        total_value = sum(portfolio_holdings.values())
        holdings_count = len(portfolio_holdings)
        
        # Calculate weights and detailed holdings info
        holdings_details = []
        for ticker, value in portfolio_holdings.items():
            weight = value / total_value if total_value > 0 else 0
            holdings_details.append({
                "ticker": ticker,
                "value": value,
                "weight": weight,
                "weight_pct": weight * 100
            })
        
        # Sort by value (largest first)
        holdings_details.sort(key=lambda x: x["value"], reverse=True)
        
        # Calculate concentration metrics
        top_5_weight = sum(h["weight"] for h in holdings_details[:5])
        top_10_weight = sum(h["weight"] for h in holdings_details[:10])
        
        # Calculate diversification metrics
        weights = np.array([h["weight"] for h in holdings_details])
        herfindahl_index = np.sum(weights**2)
        effective_number_stocks = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        return {
            "success": True,
            "name": "Portfolio",  # Default name
            "total_value": total_value,
            "holdings_count": holdings_count,
            "holdings": holdings_details,
            "concentration_metrics": {
                "top_5_weight_pct": top_5_weight * 100,
                "top_10_weight_pct": top_10_weight * 100,
                "largest_position_pct": holdings_details[0]["weight_pct"] if holdings_details else 0,
                "herfindahl_index": herfindahl_index,
                "effective_number_stocks": effective_number_stocks
            },
            "diversification_score": min(100, effective_number_stocks * 10)  # Simple score out of 100
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Portfolio summary calculation failed: {str(e)}",
            "total_value": 0.0,
            "holdings_count": 0,
            "holdings": []
        }
    

def screen_securities(
    screening_criteria: Dict[str, Any],
    universe: Optional[List[str]] = None,
    max_results: int = 10
) -> Dict[str, Any]:
    """
    Screen securities based on specified criteria.
    
    Parameters:
    -----------
    screening_criteria : Dict[str, Any]
        Criteria for screening securities. Can include:
        - factors: List of factors to screen on ['quality', 'value', 'growth']
        - min_market_cap: Minimum market cap
        - sector: Specific sector to focus on
        - exclude_tickers: List of tickers to exclude
        - portfolio_complement: Dict with current portfolio data for complement analysis
        
    universe : Optional[List[str]]
        List of tickers to screen from. If None, uses default universe.
        
    max_results : int
        Maximum number of securities to return
        
    Returns:
    --------
    Dict[str, Any]
        Screening results with recommendations
    """
    
    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import requests
        from io import StringIO
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import warnings
        warnings.filterwarnings('ignore')
        
        # Default universe - S&P 500 tickers
        if universe is None:
            universe = _get_default_screening_universe()
        
        # Get screening factors
        factors = screening_criteria.get('factors', ['quality', 'value', 'growth'])
        exclude_tickers = screening_criteria.get('exclude_tickers', [])
        portfolio_complement = screening_criteria.get('portfolio_complement')
        
        # Filter universe
        screening_universe = [t for t in universe if t not in exclude_tickers]
        
        if len(screening_universe) == 0:
            return {
                'success': False,
                'error': 'No securities in screening universe after filtering'
            }
        
        # Fetch financial data
        print(f"Screening {len(screening_universe)} securities...")
        factor_data = _fetch_factor_data_for_screening(screening_universe)
        
        if factor_data.empty:
            return {
                'success': False,
                'error': 'Could not fetch data for any securities'
            }
        
        # Perform screening based on criteria
        if portfolio_complement:
            results = _screen_for_portfolio_complement(factor_data, portfolio_complement, max_results)
        else:
            results = _screen_by_factors(factor_data, factors, max_results)
        
        return {
            'success': True,
            'screening_type': 'portfolio_complement' if portfolio_complement else 'factor_based',
            'factors_used': factors,
            'universe_size': len(screening_universe),
            'results_count': len(results),
            'recommendations': results,
            'methodology': 'Multi-factor quantitative screening'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Screening failed: {str(e)}'
        }


def _get_default_screening_universe() -> List[str]:
    """Get default screening universe (S&P 500 tickers)"""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        tables = pd.read_html(StringIO(response.text))
        sp500_df = tables[0]
        tickers = sp500_df['Symbol'].tolist()
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        return tickers
    except Exception:
        # Fallback to major tickers
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 
            'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC', 'NFLX'
        ]


def _fetch_single_ticker_data(ticker: str) -> Optional[Dict[str, Any]]:
    """Fetch financial data for a single ticker"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'ticker': ticker,
            'roe': info.get('returnOnEquity'),
            'debt_to_equity': info.get('debtToEquity'),
            'pe_ratio': info.get('trailingPE'),
            'pb_ratio': info.get('priceToBook'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'market_cap': info.get('marketCap'),
            'sector': info.get('sector', 'Unknown')
        }
    except Exception:
        return None


def _fetch_factor_data_for_screening(tickers: List[str]) -> pd.DataFrame:
    """Fetch financial data for multiple tickers in parallel"""
    
    data_list = []
    
    # Use ThreadPoolExecutor for parallel data fetching
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {
            executor.submit(_fetch_single_ticker_data, ticker): ticker 
            for ticker in tickers[:50]  # Limit to 50 for performance
        }
        
        for future in as_completed(future_to_ticker):
            result = future.result()
            if result:
                data_list.append(result)
    
    if not data_list:
        return pd.DataFrame()
    
    df = pd.DataFrame(data_list)
    
    # Calculate factor scores
    df = _calculate_factor_scores_for_screening(df)
    
    return df


def _calculate_factor_scores_for_screening(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate quality, value, and growth factor scores"""
    
    df = df.copy()
    df.set_index('ticker', inplace=True)
    
    # Fill NaN values
    df = df.fillna(method='median')
    
    # Quality Score (ROE high, Debt/Equity low)
    roe_rank = df['roe'].rank(ascending=False, pct=True)
    de_rank = df['debt_to_equity'].rank(ascending=True, pct=True)  # Lower is better
    df['quality_score'] = (roe_rank * 0.6 + de_rank * 0.4).fillna(0.5)
    
    # Value Score (PE low, PB low)
    df['pe_positive'] = df['pe_ratio'].where(df['pe_ratio'] > 0)
    pe_rank = df['pe_positive'].rank(ascending=True, pct=True)  # Lower is better
    pb_rank = df['pb_ratio'].rank(ascending=True, pct=True)     # Lower is better
    df['value_score'] = (pe_rank * 0.5 + pb_rank * 0.5).fillna(0.5)
    
    # Growth Score (Revenue and Earnings growth high)
    rg_rank = df['revenue_growth'].rank(ascending=False, pct=True)
    eg_rank = df['earnings_growth'].rank(ascending=False, pct=True)
    df['growth_score'] = (rg_rank * 0.5 + eg_rank * 0.5).fillna(0.5)
    
    return df.reset_index()


def _screen_by_factors(df: pd.DataFrame, factors: List[str], max_results: int) -> List[Dict[str, Any]]:
    """Screen securities by specified factors"""
    
    # Calculate overall score based on requested factors
    factor_cols = [f'{factor}_score' for factor in factors if f'{factor}_score' in df.columns]
    
    if not factor_cols:
        factor_cols = ['quality_score', 'value_score', 'growth_score']
    
    df['overall_score'] = df[factor_cols].mean(axis=1)
    
    # Sort by overall score and get top results
    top_securities = df.nlargest(max_results, 'overall_score')
    
    results = []
    for _, row in top_securities.iterrows():
        results.append({
            'ticker': row['ticker'],
            'overall_score': round(row['overall_score'], 3),
            'quality_score': round(row.get('quality_score', 0), 3),
            'value_score': round(row.get('value_score', 0), 3),
            'growth_score': round(row.get('growth_score', 0), 3),
            'sector': row.get('sector', 'Unknown'),
            'market_cap': row.get('market_cap'),
            'rationale': f"High {', '.join(factors)} factor scores"
        })
    
    return results


def _screen_for_portfolio_complement(
    df: pd.DataFrame, 
    portfolio_data: Dict[str, Any], 
    max_results: int
) -> List[Dict[str, Any]]:
    """Screen for securities that complement existing portfolio"""
    
    # Calculate current portfolio factor exposure
    holdings = portfolio_data.get('holdings', [])
    total_value = sum(h.get('market_value', 0) for h in holdings)
    
    if total_value == 0:
        return _screen_by_factors(df, ['quality', 'value', 'growth'], max_results)
    
    # Calculate weighted factor exposures
    portfolio_quality = 0
    portfolio_value = 0
    portfolio_growth = 0
    
    for holding in holdings:
        weight = holding.get('market_value', 0) / total_value
        ticker = holding.get('ticker', '')
        
        # Find ticker in factor data
        ticker_data = df[df['ticker'] == ticker]
        if not ticker_data.empty:
            portfolio_quality += weight * ticker_data.iloc[0].get('quality_score', 0.5)
            portfolio_value += weight * ticker_data.iloc[0].get('value_score', 0.5)
            portfolio_growth += weight * ticker_data.iloc[0].get('growth_score', 0.5)
    
    # Calculate complement scores (favor factors where portfolio is weak)
    quality_weight = 1 - portfolio_quality
    value_weight = 1 - portfolio_value
    growth_weight = 1 - portfolio_growth
    
    df['complement_score'] = (
        df['quality_score'] * quality_weight +
        df['value_score'] * value_weight +
        df['growth_score'] * growth_weight
    ) / 3
    
    # Exclude existing holdings
    existing_tickers = [h.get('ticker', '') for h in holdings]
    df_filtered = df[~df['ticker'].isin(existing_tickers)]
    
    # Get top complement securities
    top_complements = df_filtered.nlargest(max_results, 'complement_score')
    
    results = []
    for _, row in top_complements.iterrows():
        results.append({
            'ticker': row['ticker'],
            'overall_score': round(row['complement_score'], 3),
            'quality_score': round(row.get('quality_score', 0), 3),
            'value_score': round(row.get('value_score', 0), 3),
            'growth_score': round(row.get('growth_score', 0), 3),
            'sector': row.get('sector', 'Unknown'),
            'market_cap': row.get('market_cap'),
            'rationale': f"Complements portfolio factor exposure (Q:{portfolio_quality:.2f}, V:{portfolio_value:.2f}, G:{portfolio_growth:.2f})"
        })
    
    return results