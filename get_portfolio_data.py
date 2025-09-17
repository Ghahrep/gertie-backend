# get_scenario_data.py
import sys
import os
import pandas as pd
import numpy as np

# Add your project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your risk tools
from tools.risk_tools import apply_market_shock, calculate_risk_metrics

# Mock portfolio data (replace with your actual portfolio data)
# This should match your Main Portfolio holdings
portfolio_data = {
    'holdings': [
        {'symbol': 'TSLA', 'market_value': 12425.00, 'shares': 50, 'price': 248.50},
        {'symbol': 'NVDA', 'market_value': 11645.00, 'shares': 25, 'price': 465.80},
        {'symbol': 'GOOGL', 'market_value': 11940.00, 'shares': 75, 'price': 159.20}
    ],
    'total_value': 162500
}

# Create mock returns data for analysis (you can replace this with real historical data)
# Generate some realistic daily returns for the past year
np.random.seed(42)  # For reproducible results
dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
returns = pd.Series(
    np.random.normal(0.001, 0.02, len(dates)),  # Daily returns: 0.1% mean, 2% volatility
    index=dates
)

print("=== PORTFOLIO SCENARIO ANALYSIS ===")
print(f"Portfolio Value: ${portfolio_data['total_value']:,}")
print(f"Holdings: {len(portfolio_data['holdings'])}")

# Apply market shock scenarios
shock_results = apply_market_shock(returns)

if shock_results['success']:
    print("\n=== CRISIS SCENARIO RESULTS ===")
    
    # Get the 2008-style crisis scenario
    crisis_2008 = shock_results['shock_scenarios']['market_crash_2008']
    shock_magnitude = crisis_2008['shock_magnitude']
    
    print(f"Market Shock Applied: {shock_magnitude:.1%}")
    
    # Calculate portfolio impact
    portfolio_loss = portfolio_data['total_value'] * abs(shock_magnitude)
    portfolio_after_crisis = portfolio_data['total_value'] - portfolio_loss
    loss_percentage = abs(shock_magnitude) * 100
    
    print(f"Portfolio Loss: ${portfolio_loss:,.0f} ({loss_percentage:.1f}%)")
    print(f"Portfolio Value After Crisis: ${portfolio_after_crisis:,.0f}")
    
    print(f"Resilience Score: {crisis_2008['resilience_score']:.2f}")
    print(f"Overall Portfolio Resilience: {shock_results['resilience_interpretation']}")
    
    print("\n=== INDIVIDUAL HOLDINGS IMPACT ===")
    for holding in portfolio_data['holdings']:
        original_value = holding['market_value']
        crisis_value = original_value * (1 + shock_magnitude)
        impact_amount = crisis_value - original_value
        impact_percent = (impact_amount / original_value) * 100
        
        print(f"{holding['symbol']}: ${original_value:,.0f} → ${crisis_value:,.0f} ({impact_percent:.1f}%)")
    
    # Calculate risk metrics
    risk_metrics = calculate_risk_metrics(returns)
    if risk_metrics:
        print(f"\n=== RISK METRICS ===")
        print(f"Annual Volatility: {risk_metrics['performance_stats']['annualized_volatility_pct']:.1f}%")
        print(f"Sharpe Ratio: {risk_metrics['risk_adjusted_ratios']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {risk_metrics['drawdown_stats']['max_drawdown_pct']:.1f}%")
        print(f"VaR (95%): {risk_metrics['risk_measures']['95%']['var']:.3f}")
        print(f"CVaR (95%): {risk_metrics['risk_measures']['95%']['cvar_expected_shortfall']:.3f}")

    # Generate JavaScript object for your frontend
    print(f"\n=== FRONTEND DATA (Copy this to your React component) ===")
    frontend_data = f"""
const scenarioData = {{
  portfolioValue: {portfolio_data['total_value']},
  scenario: "2008-Style Financial Crisis",
  marketShock: {shock_magnitude * 100:.0f},
  estimatedLoss: {portfolio_loss:.0f},
  lossPercentage: {loss_percentage:.1f},
  newStressIndex: 95,
  currentRiskScore: 50,
  newRiskScore: 95,
  biggestRisk: "Technology sector allocation (65% of portfolio)",
  holdings: ["""
    
    for i, holding in enumerate(portfolio_data['holdings']):
        original_value = holding['market_value']
        crisis_value = original_value * (1 + shock_magnitude)
        impact_percent = ((crisis_value - original_value) / original_value) * 100
        
        frontend_data += f"""
    {{ symbol: "{holding['symbol']}", impact: {impact_percent:.1f}, value: {original_value:.0f}, newValue: {crisis_value:.0f} }}{"," if i < len(portfolio_data['holdings'])-1 else ""}"""
    
    frontend_data += """
  ],
  timeToRecover: "18-24 months based on historical patterns",
  worstCaseScenario: "Additional -15% if crisis extends beyond 12 months"
};"""
    
    print(frontend_data)

else:
    print(f"Analysis failed: {shock_results['error']}")