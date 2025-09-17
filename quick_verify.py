# quick_verify.py - Quick verification of enhanced tools
"""
Quick verification script to test enhanced backend tools
=======================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def quick_test():
    """Quick test of enhanced functionality"""
    print("Quick Verification of Enhanced Backend Tools")
    print("=" * 45)
    
    # Generate minimal test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    returns = pd.DataFrame(
        np.random.normal(0.001, 0.02, (100, 5)),
        index=dates,
        columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    )
    prices = (1 + returns).cumprod() * 100
    weights = pd.Series(0.2, index=returns.columns)
    
    print(f"Test data: {len(returns)} days, {len(returns.columns)} assets")
    
    # Test imports
    try:
        from tools.strategy_tools import design_risk_adjusted_momentum_strategy
        from tools.risk_tools import calculate_regime_conditional_risk
        print("✅ Enhanced tool imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Test basic functionality
    try:
        # Test enhanced strategy
        print("\nTesting enhanced momentum strategy...")
        momentum_result = design_risk_adjusted_momentum_strategy(
            prices, returns, lookback_days=30
        )
        print(f"✅ Momentum strategy: {len(momentum_result['candidates'])} candidates")
        
        # Test enhanced risk
        print("Testing regime-conditional risk...")
        portfolio_returns = (returns * weights).sum(axis=1)
        risk_result = calculate_regime_conditional_risk(portfolio_returns)
        print(f"✅ Risk analysis: {len(risk_result['regime_risk_metrics'])} regimes detected")
        
        print("\n🎉 Quick verification PASSED - Tools are working!")
        
    except Exception as e:
        print(f"❌ Runtime error: {e}")
        print("Check tool implementations for issues")

if __name__ == "__main__":
    quick_test()