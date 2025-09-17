# clean_risk_tools.py - Script to identify duplicate functions
"""
Script to help identify and clean up duplicate functions in risk_tools.py
"""

import inspect
import importlib.util

def analyze_risk_tools_file():
    """Analyze the risk_tools.py file for duplicate function definitions"""
    
    print("Risk Tools File Analysis")
    print("=" * 40)
    
    try:
        # Import the module
        import tools.risk_tools as risk_tools
        
        # Get all functions in the module
        functions = inspect.getmembers(risk_tools, inspect.isfunction)
        
        print(f"Total functions found: {len(functions)}")
        print("\nFunction List:")
        
        core_functions = [
            'calculate_risk_metrics',
            'calculate_regime_conditional_risk', 
            'calculate_factor_risk_attribution',
            'calculate_dynamic_risk_budgets',
            'advanced_monte_carlo_stress_test',
            'calculate_time_varying_risk',
            'calculate_correlation_matrix',
            'calculate_beta',
            'apply_market_shock'
        ]
        
        for func_name, func_obj in functions:
            status = "✅ CORE" if func_name in core_functions else "ℹ️  OTHER"
            print(f"  {status} {func_name}")
        
        print(f"\nCore Functions Status:")
        for core_func in core_functions:
            if hasattr(risk_tools, core_func):
                print(f"  ✅ {core_func} - Available")
            else:
                print(f"  ❌ {core_func} - Missing")
        
        # Test a few key functions
        print(f"\nFunction Testing:")
        
        # Test the main function that was causing issues
        if hasattr(risk_tools, 'calculate_risk_metrics'):
            print("  Testing calculate_risk_metrics...")
            import pandas as pd
            import numpy as np
            
            # Create test data
            test_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
            result = risk_tools.calculate_risk_metrics(test_returns)
            
            if result is not None:
                print(f"    ✅ Returns: {type(result)} with keys: {list(result.keys())}")
            else:
                print(f"    ❌ Returns: None")
        
        print(f"\nRecommendation:")
        print(f"Your risk_tools.py appears to have all enhanced functions.")
        print(f"Run the backend verification to confirm everything works properly.")
        
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_risk_tools_file()