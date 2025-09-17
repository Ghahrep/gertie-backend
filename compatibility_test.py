# compatibility_test.py - Test your risk function with correct input format
"""
Test script to verify your risk_tools function works with proper Series input
"""

import pandas as pd
import numpy as np

def test_risk_function_compatibility():
    """Test your existing risk function with correct input format"""
    
    print("Testing Risk Function Compatibility")
    print("=" * 40)
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    returns_df = pd.DataFrame(
        np.random.normal(0.001, 0.02, (100, 5)),
        index=dates,
        columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    )
    
    # Create portfolio returns Series (what your function expects)
    weights = pd.Series(0.2, index=returns_df.columns)  # Equal weights
    portfolio_returns = (returns_df * weights).sum(axis=1)
    
    print(f"Input type: {type(portfolio_returns)}")
    print(f"Input length: {len(portfolio_returns)}")
    print(f"Input sample: {portfolio_returns.head(3).values}")
    
    try:
        from tools.risk_tools import calculate_risk_metrics
        
        # Test with Series input (correct format)
        result = calculate_risk_metrics(portfolio_returns)
        
        if result is None:
            print("❌ Function returned None")
            
            # Test with DataFrame input (wrong format) 
            result_df = calculate_risk_metrics(returns_df)
            print(f"DataFrame input result: {result_df}")
            
        else:
            print("✅ Function returned valid result")
            print(f"Result type: {type(result)}")
            print(f"Result keys: {list(result.keys())}")
            
            # Show structure
            if 'risk_measures' in result:
                print(f"Risk measures: {list(result['risk_measures'].keys())}")
            
            if 'performance_stats' in result:
                perf_stats = result['performance_stats']
                print(f"Performance stats: {list(perf_stats.keys())}")
                print(f"Annual volatility: {perf_stats.get('annualized_volatility_pct', 'N/A')}%")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Execution error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_risk_function_compatibility()