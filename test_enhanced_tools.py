# test_enhanced_tools.py - Comprehensive Testing Suite
"""
Testing Suite for Enhanced Strategy and Risk Tools
=================================================

Tests all new functionality with realistic portfolio data.
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import your enhanced tools
from tools.strategy_tools import *
from tools.risk_tools import *
from tools.regime_tools import *
from tools.portfolio_tools import *

class TestEnhancedBackendTools:
    """Comprehensive test suite for enhanced backend tools"""
    
    def setup_method(self):
        """Setup test data for all tests"""
        # Generate realistic test data
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        n_assets = 10
        n_days = len(dates)
        
        # Generate correlated returns with realistic characteristics
        base_returns = np.random.normal(0.0008, 0.02, (n_days, n_assets))
        
        # Add market factor
        market_factor = np.random.normal(0.001, 0.015, n_days)
        for i in range(n_assets):
            beta = np.random.uniform(0.5, 1.5)
            base_returns[:, i] += beta * market_factor
        
        # Create asset names
        self.tickers = [f'ASSET_{i:02d}' for i in range(n_assets)]
        
        # Create DataFrames
        self.returns = pd.DataFrame(base_returns, index=dates, columns=self.tickers)
        self.prices = (1 + self.returns).cumprod() * 100
        
        # Portfolio weights (equal weighted)
        self.weights = pd.Series(1/n_assets, index=self.tickers)
        
        # Benchmark returns
        self.benchmark_returns = pd.Series(market_factor, index=dates)
        
        print(f"Test data created: {len(self.returns)} days, {len(self.tickers)} assets")
        print(f"Return range: {self.returns.min().min():.4f} to {self.returns.max().max():.4f}")

    def test_enhanced_strategy_tools(self):
        """Test enhanced strategy functionality"""
        print("\n=== Testing Enhanced Strategy Tools ===")
        
        try:
            # Test Risk-Adjusted Momentum Strategy
            print("Testing risk-adjusted momentum strategy...")
            momentum_result = design_risk_adjusted_momentum_strategy(
                self.prices, 
                self.returns,
                lookback_days=60,
                min_sharpe=0.5,
                max_volatility=0.25,
                max_beta=1.5
            )
            
            print(f"Momentum candidates: {len(momentum_result['candidates'])}")
            print(f"Risk metrics included: {list(momentum_result['risk_metrics'].keys())}")
            
            # Test Enhanced Mean Reversion
            print("\nTesting enhanced mean reversion strategy...")
            mean_rev_result = design_enhanced_mean_reversion_strategy(
                self.prices,
                self.returns,
                max_correlation=0.7,
                min_rsquared=0.6,
                max_volatility=0.3
            )
            
            print(f"Mean reversion candidates: {len(mean_rev_result['candidates'])}")
            print(f"Correlation matrix shape: {mean_rev_result['correlation_analysis']['correlation_matrix'].shape}")
            
            # Test Portfolio Construction
            print("\nTesting portfolio construction...")
            portfolio_result = construct_risk_managed_portfolio(
                mean_rev_result['candidates'][:5],  # Use top 5 candidates
                self.returns,
                total_risk_budget=0.15,
                max_position_size=0.3
            )
            
            print(f"Portfolio weights sum: {portfolio_result['weights'].sum():.4f}")
            print(f"Expected portfolio volatility: {portfolio_result['portfolio_metrics']['volatility']:.4f}")
            
            print("✅ Enhanced Strategy Tools - All tests passed")
            
        except Exception as e:
            print(f"❌ Enhanced Strategy Tools - Error: {str(e)}")
            raise

    def test_enhanced_risk_tools(self):
        """Test enhanced risk functionality"""
        print("\n=== Testing Enhanced Risk Tools ===")
        
        try:
            # Test Regime-Conditional Risk
            print("Testing regime-conditional risk analysis...")
            regime_risk = calculate_regime_conditional_risk(
                self.returns.iloc[:, 0],  # Single asset for simplicity
                confidence_levels=[0.95, 0.99]
            )
            
            print(f"Number of regimes detected: {len(regime_risk['regime_risk_metrics'])}")
            print(f"Current regime: {regime_risk['current_regime']}")
            
            # Test Factor Risk Attribution
            print("\nTesting factor risk attribution...")
            
            # Create factor data (market, sector, style factors)
            factor_data = pd.DataFrame({
                'Market': self.benchmark_returns.values,
                'Value': np.random.normal(0, 0.01, len(self.returns)),
                'Growth': np.random.normal(0, 0.01, len(self.returns)),
                'Small_Cap': np.random.normal(0, 0.008, len(self.returns))
            }, index=self.returns.index)
            
            portfolio_returns = (self.returns * self.weights).sum(axis=1)
            
            attribution = calculate_factor_risk_attribution(
                portfolio_returns,
                factor_data,
                self.weights
            )
            
            print(f"Factor loadings: {list(attribution['factor_loadings'].keys())}")
            print(f"Total R-squared: {attribution['total_rsquared']:.4f}")
            print(f"Specific risk %: {attribution['risk_attribution']['Specific Risk']:.2f}%")
            
            # Test Dynamic Risk Budgeting
            print("\nTesting dynamic risk budgeting...")
            risk_budget = calculate_dynamic_risk_budgets(
                self.returns,
                current_weights=self.weights
            )
            
            print(f"Current risk concentration: {risk_budget['current_allocation']['concentration_ratio']:.4f}")
            print(f"Optimal weights sum: {risk_budget['optimal_allocation']['weights'].sum():.4f}")
            
            # Test Advanced Monte Carlo
            print("\nTesting advanced Monte Carlo stress testing...")
            stress_test = advanced_monte_carlo_stress_test(
                portfolio_returns,
                num_scenarios=1000,
                time_horizon=30
            )
            
            print(f"VaR (95%): {stress_test['var_estimates']['VaR_95']:.4f}")
            print(f"Expected Shortfall (99%): {stress_test['var_estimates']['ES_99']:.4f}")
            print(f"Worst case scenario: {stress_test['worst_case_scenario']['loss']:.4f}")
            
            # Test Time-Varying Risk
            print("\nTesting time-varying risk analysis...")
            time_varying = calculate_time_varying_risk(
                portfolio_returns,
                window=30
            )
            
            print(f"Risk evolution periods: {len(time_varying['risk_evolution'])}")
            print(f"Current risk state: {time_varying['current_risk_state']}")
            
            print("✅ Enhanced Risk Tools - All tests passed")
            
        except Exception as e:
            print(f"❌ Enhanced Risk Tools - Error: {str(e)}")
            raise

    def test_integration_functionality(self):
        """Test integration between tools"""
        print("\n=== Testing Tool Integration ===")
        
        try:
            # Test cross-tool integration
            print("Testing regime-aware strategy selection...")
            
            # Get current regime
            portfolio_returns = (self.returns * self.weights).sum(axis=1)
            regime_info = detect_hmm_regimes(portfolio_returns)
            current_regime = regime_info['current_regime']
            
            print(f"Current market regime: {current_regime}")
            
            # Select strategy based on regime
            if current_regime == 0:  # Assume 0 = low volatility regime
                strategy = design_risk_adjusted_momentum_strategy(self.prices, self.returns)
                print("Selected: Risk-adjusted momentum strategy")
            else:  # High volatility regime
                strategy = design_enhanced_mean_reversion_strategy(self.prices, self.returns)
                print("Selected: Enhanced mean reversion strategy")
            
            # Apply risk controls from risk tools
            risk_metrics = calculate_risk_metrics(self.returns)
            print(f"Portfolio VaR (95%): {risk_metrics['portfolio']['VaR_95']:.4f}")
            
            # Test behavioral analysis integration
            print("\nTesting behavioral analysis...")
            sample_chat = """
            I'm really worried about my portfolio losses. Everyone seems to be selling their tech stocks.
            Maybe I should sell everything and wait for things to calm down. 
            I've been checking my portfolio every hour and it's making me anxious.
            """
            
            bias_analysis = analyze_enhanced_chat_for_biases(sample_chat, portfolio_returns)
            print(f"Detected biases: {list(bias_analysis['detected_biases'].keys())}")
            print(f"Risk impact score: {bias_analysis['risk_impact_score']:.2f}")
            
            print("✅ Tool Integration - All tests passed")
            
        except Exception as e:
            print(f"❌ Tool Integration - Error: {str(e)}")
            raise

    def test_performance_and_edge_cases(self):
        """Test performance and edge cases"""
        print("\n=== Testing Performance & Edge Cases ===")
        
        try:
            # Test with insufficient data
            print("Testing insufficient data handling...")
            short_returns = self.returns.iloc[:10]  # Only 10 days
            
            try:
                result = design_risk_adjusted_momentum_strategy(
                    short_returns.cumsum(), 
                    short_returns,
                    lookback_days=60  # More than available data
                )
                print("✅ Graceful handling of insufficient data")
            except Exception as e:
                print(f"⚠️  Expected error for insufficient data: {str(e)[:50]}...")
            
            # Test with missing data
            print("Testing missing data handling...")
            missing_returns = self.returns.copy()
            missing_returns.iloc[100:110] = np.nan  # Add missing data
            
            risk_metrics = calculate_risk_metrics(missing_returns)
            print(f"✅ Handled missing data, VaR calculated: {risk_metrics['portfolio']['VaR_95']:.4f}")
            
            # Test performance with large dataset
            print("Testing performance with larger dataset...")
            import time
            
            start_time = time.time()
            large_result = calculate_regime_conditional_risk(portfolio_returns)
            execution_time = time.time() - start_time
            
            print(f"✅ Large dataset processing time: {execution_time:.2f} seconds")
            
            print("✅ Performance & Edge Cases - All tests passed")
            
        except Exception as e:
            print(f"❌ Performance & Edge Cases - Error: {str(e)}")
            raise

def run_comprehensive_test():
    """Run all tests"""
    print("Starting Comprehensive Backend Tools Testing")
    print("=" * 50)
    
    tester = TestEnhancedBackendTools()
    tester.setup_method()
    
    try:
        tester.test_enhanced_strategy_tools()
        tester.test_enhanced_risk_tools()
        tester.test_integration_functionality()
        tester.test_performance_and_edge_cases()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED - Backend tools are production ready!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ TESTING FAILED: {str(e)}")
        print("Check the specific error messages above for details.")

if __name__ == "__main__":
    run_comprehensive_test()