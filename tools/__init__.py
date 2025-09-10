# tools/__init__.py - Complete Clean Financial Tools Package
"""
Financial Analysis Tools - Clean Architecture
============================================

Production-grade financial tools extracted from agent complexity.
Direct function calls for fast, reliable financial analysis.

Available Tools:
- Risk Tools: VaR/CVaR, GARCH, drawdown analysis
- Portfolio Tools: Optimization, trade generation, asset allocation  
- Strategy Tools: Backtesting, momentum/mean-reversion strategies
- Behavioral Tools: Bias detection, sentiment analysis, AI summarization
- Fractal Tools: Hurst exponent, DFA, multifractal analysis, fBm simulation
- Regime Tools: HMM regime detection, volatility regimes, persistence analysis
"""

__version__ = "1.0.0"

# Import only functions that actually exist in the tool files
try:
    from .risk_tools import (
        calculate_risk_metrics,
        calculate_drawdowns,
        fit_garch_forecast,
        calculate_correlation_matrix,
        calculate_beta,
        apply_market_shock
    )
except ImportError as e:
    print(f"Warning: Could not import some risk tools: {e}")

try:
    from .portfolio_tools import (
        optimize_portfolio,
        generate_trade_orders,
        run_backtest
    )
    
    # Try to import optional portfolio functions
    try:
        from .portfolio_tools import calculate_portfolio_summary
    except ImportError:
        calculate_portfolio_summary = None
        
    try:
        from .portfolio_tools import rebalance_portfolio
    except ImportError:
        rebalance_portfolio = None
        
    try:
        from .portfolio_tools import screen_securities
    except ImportError:
        screen_securities = None
        
except ImportError as e:
    print(f"Warning: Could not import some portfolio tools: {e}")

try:
    from .strategy_tools import (
        design_mean_reversion_strategy,
        design_momentum_strategy
    )
except ImportError as e:
    print(f"Warning: Could not import some strategy tools: {e}")

try:
    from .behavioral_tools import (
        analyze_chat_for_biases,
        summarize_analysis_results,
        detect_market_sentiment
    )
except ImportError as e:
    print(f"Warning: Could not import behavioral tools: {e}")

try:
    from .fractal_tools import (
        calculate_hurst,
        calculate_dfa,
        calculate_multifractal_spectrum,
        generate_fbm_path,
        calculate_detrended_cross_correlation
    )
except ImportError as e:
    print(f"Warning: Could not import fractal tools: {e}")

try:
    from .regime_tools import (
        detect_hmm_regimes,
        analyze_hurst_regimes,
        forecast_regime_transition_probability,
        detect_volatility_regimes,
        analyze_regime_persistence
    )
except ImportError as e:
    print(f"Warning: Could not import regime tools: {e}")

def get_tool_info():
    """Return information about available tools."""
    return {
        "version": __version__,
        "tool_categories": {
            "risk_analysis": [
                "calculate_risk_metrics", "calculate_drawdowns", 
                "fit_garch_forecast", "calculate_correlation_matrix",
                "calculate_beta", "apply_market_shock"
            ],
            "portfolio_management": [
                "optimize_portfolio", "generate_trade_orders"
            ],
            "strategy_development": [
                "design_mean_reversion_strategy", "design_momentum_strategy",
                "run_backtest"
            ],
            "behavioral_analysis": [
                "analyze_chat_for_biases", "summarize_analysis_results",
                "detect_market_sentiment"
            ],
            "fractal_analysis": [
                "calculate_hurst", "calculate_dfa", "calculate_multifractal_spectrum",
                "generate_fbm_path", "calculate_detrended_cross_correlation"
            ],
            "regime_analysis": [
                "detect_hmm_regimes", "analyze_hurst_regimes", 
                "forecast_regime_transition_probability", "detect_volatility_regimes",
                "analyze_regime_persistence"
            ]
        },
        "description": "Complete financial analysis toolkit for direct integration",
        "note": "Some optional functions may not be available depending on tool file completeness"
    }