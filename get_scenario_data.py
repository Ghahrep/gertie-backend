# create a file: get_scenario_data.py
import sys
sys.path.append('/path/to/your/backend')

from your_analysis_module import apply_market_shock, get_portfolio_data
import json

# Get portfolio 3 (Main Portfolio) data
portfolio_data = get_portfolio_data(portfolio_id=3)
print("Portfolio Data:", json.dumps(portfolio_data, indent=2))

# Apply -37% market shock
crisis_results = apply_market_shock(portfolio_data, shock_percentage=-37)
print("Crisis Results:", json.dumps(crisis_results, indent=2))

# Calculate individual holding impacts
for holding in portfolio_data['holdings']:
    original_value = holding['market_value']
    crisis_value = original_value * (1 + crisis_results['shock_percentage']/100)
    impact_percent = (crisis_value - original_value) / original_value * 100
    
    print(f"{holding['symbol']}: ${original_value:,.0f} → ${crisis_value:,.0f} ({impact_percent:.1f}%)")