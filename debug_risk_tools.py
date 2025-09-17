# debug_risk_tools.py - Debug the risk metrics format issue
"""
Debug script to understand the risk metrics output format
"""

import pandas as pd
import numpy as np

def debug_risk_metrics():
    """Debug the risk metrics function output"""
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    returns = pd.DataFrame(
        np.random.normal(0.001, 0.02, (100, 5)),
        index=dates,
        columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    )
    
    print("Debugging Risk Tools Output Format")
    print("=" * 40)
    
    try:
        from tools.risk_tools import calculate_risk_metrics
        
        result = calculate_risk_metrics(returns)
        
        print(f"Return type: {type(result)}")
        print(f"Is dict: {isinstance(result, dict)}")
        
        if hasattr(result, 'keys'):
            print(f"Keys: {list(result.keys())}")
            
            for key, value in result.items():
                print(f"\n{key}:")
                print(f"  Type: {type(value)}")
                if isinstance(value, dict):
                    print(f"  Sub-keys: {list(value.keys())}")
                elif hasattr(value, 'shape'):
                    print(f"  Shape: {value.shape}")
                else:
                    print(f"  Value: {value}")
        else:
            print(f"Result: {result}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_risk_metrics()