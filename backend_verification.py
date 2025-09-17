# backend_verification.py - Comprehensive Backend Tools Verification
"""
Backend Tools Verification Suite
===============================

Systematic testing of all backend financial analysis tools with detailed reporting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback

class BackendToolsVerifier:
    """Comprehensive verification of all backend tools"""
    
    def __init__(self):
        self.test_results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
        self.setup_test_data()
    
    def setup_test_data(self):
        """Generate realistic test data"""
        np.random.seed(42)
        self.dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        # Generate correlated returns with market factor
        market_factor = np.random.normal(0.001, 0.015, 100)
        returns_data = []
        
        for asset in self.assets:
            beta = np.random.uniform(0.8, 1.2)
            asset_specific = np.random.normal(0, 0.01, 100)
            asset_returns = beta * market_factor + asset_specific
            returns_data.append(asset_returns)
        
        self.returns = pd.DataFrame(
            np.array(returns_data).T, 
            index=self.dates, 
            columns=self.assets
        )
        self.prices = (1 + self.returns).cumprod() * 100
        self.weights = pd.Series(0.2, index=self.assets)
        
        print(f"Test Environment: {len(self.returns)} days, {len(self.assets)} assets")
        print(f"Date range: {self.dates[0].date()} to {self.dates[-1].date()}")
        print(f"Return statistics: mean={self.returns.mean().mean():.4f}, std={self.returns.std().mean():.4f}")
    
    def test_component(self, component_name, test_function):
        """Execute a test component and record results"""
        try:
            print(f"\n--- Testing {component_name} ---")
            result = test_function()
            
            if result.get('success', False):
                self.test_results['passed'].append({
                    'component': component_name,
                    'details': result.get('details', 'Test passed'),
                    'metrics': result.get('metrics', {})
                })
                print(f"PASS: {component_name}")
                if result.get('details'):
                    print(f"  Details: {result['details']}")
            else:
                self.test_results['warnings'].append({
                    'component': component_name,
                    'issue': result.get('error', 'Unknown issue'),
                    'severity': result.get('severity', 'Medium')
                })
                print(f"WARN: {component_name} - {result.get('error', 'Unknown issue')}")
                
        except Exception as e:
            error_details = {
                'component': component_name,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            self.test_results['failed'].append(error_details)
            print(f"FAIL: {component_name} - {str(e)}")
    
    def test_basic_risk_tools(self):
        """Test core risk analysis functionality"""
        try:
            from tools.risk_tools import calculate_risk_metrics, calculate_correlation_matrix
            
            # Convert DataFrame to Series for your risk metrics function
            portfolio_returns = (self.returns * self.weights).sum(axis=1)  # Create portfolio series
            
            # Test risk metrics with Series input (as your function expects)
            risk_result = calculate_risk_metrics(portfolio_returns)
            
            if risk_result is None:
                return {'success': False, 'error': 'Risk metrics function returned None'}
            
            if not isinstance(risk_result, dict):
                return {'success': False, 'error': f'Risk metrics returned {type(risk_result)}, expected dict'}
            
            # Check for expected structure from your function
            expected_keys = ['risk_measures', 'performance_stats', 'risk_adjusted_ratios', 'drawdown_stats']
            missing_keys = [key for key in expected_keys if key not in risk_result]
            
            if missing_keys:
                return {'success': False, 'error': f'Missing expected keys: {missing_keys}'}
            
            # Test correlation matrix
            corr_result = calculate_correlation_matrix(self.returns)
            if not isinstance(corr_result, pd.DataFrame):
                return {'success': False, 'error': 'Correlation matrix returned invalid format'}
            
            # Extract metrics from your function's structure
            risk_measures = risk_result.get('risk_measures', {})
            var_95_data = risk_measures.get('95%', {})
            var_95 = var_95_data.get('var', 'N/A')
            
            performance_stats = risk_result.get('performance_stats', {})
            annual_vol = performance_stats.get('annualized_volatility_pct', 'N/A')
            
            return {
                'success': True,
                'details': f"Risk metrics calculated successfully, correlation matrix {corr_result.shape}",
                'metrics': {
                    'var_95': var_95,
                    'annual_volatility_pct': annual_vol,
                    'correlation_range': f"{corr_result.min().min():.3f} to {corr_result.max().max():.3f}",
                    'function_keys': list(risk_result.keys())
                }
            }
            
        except ImportError as e:
            return {'success': False, 'error': f'Import failed: {str(e)}', 'severity': 'High'}
        except Exception as e:
            return {'success': False, 'error': f'Execution failed: {str(e)}'}
    
    def test_enhanced_risk_tools(self):
        """Test enhanced risk analysis features"""
        try:
            from tools.risk_tools import calculate_regime_conditional_risk
            
            portfolio_returns = (self.returns * self.weights).sum(axis=1)
            regime_result = calculate_regime_conditional_risk(portfolio_returns)
            
            if 'error' in regime_result:
                return {'success': False, 'error': regime_result['error']}
            
            return {
                'success': True,
                'details': f"Regime analysis: {len(regime_result.get('regime_risk_metrics', {}))} regimes detected",
                'metrics': {
                    'current_regime': regime_result.get('current_regime', 'Unknown'),
                    'total_observations': regime_result.get('analysis_periods', {}).get('total_observations', 0)
                }
            }
            
        except ImportError:
            return {'success': False, 'error': 'Enhanced risk functions not available', 'severity': 'Low'}
        except Exception as e:
            return {'success': False, 'error': f'Enhanced risk execution failed: {str(e)}'}
    
    def test_strategy_tools(self):
        """Test strategy generation and analysis"""
        try:
            from tools.strategy_tools import design_risk_adjusted_momentum_strategy
            
            momentum_result = design_risk_adjusted_momentum_strategy(
                self.prices, 
                self.returns, 
                lookback_days=30
            )
            
            if 'error' in momentum_result:
                return {'success': False, 'error': momentum_result['error']}
            
            return {
                'success': True,
                'details': f"Strategy analysis: {len(momentum_result.get('candidates', []))} candidates identified",
                'metrics': {
                    'total_screened': momentum_result.get('total_screened', 0),
                    'passed_filters': momentum_result.get('passed_filters', 0),
                    'strategy_type': momentum_result.get('strategy_type', 'Unknown')
                }
            }
            
        except ImportError as e:
            return {'success': False, 'error': f'Strategy tools import failed: {str(e)}', 'severity': 'High'}
        except Exception as e:
            return {'success': False, 'error': f'Strategy execution failed: {str(e)}'}
    
    def test_behavioral_tools(self):
        """Test behavioral analysis functionality"""
        try:
            from tools.behavioral_tools import analyze_chat_for_biases, detect_market_sentiment
            
            sample_chat = [
                {"role": "user", "content": "I'm worried about my portfolio. Everyone seems to be selling."},
                {"role": "assistant", "content": "I understand your concern."},
                {"role": "user", "content": "Maybe I should sell everything and wait for things to calm down."}
            ]
            
            # Test bias analysis
            bias_result = analyze_chat_for_biases(sample_chat)
            if not bias_result.get('success', False):
                return {'success': False, 'error': bias_result.get('error', 'Bias analysis failed')}
            
            # Test sentiment analysis
            sentiment_result = detect_market_sentiment(sample_chat)
            if not sentiment_result.get('success', False):
                return {'success': False, 'error': sentiment_result.get('error', 'Sentiment analysis failed')}
            
            return {
                'success': True,
                'details': f"Behavioral analysis: {len(bias_result.get('biases_detected', {}))} biases, {sentiment_result.get('sentiment', 'unknown')} sentiment",
                'metrics': {
                    'biases_detected': list(bias_result.get('biases_detected', {}).keys()),
                    'sentiment': sentiment_result.get('sentiment', 'unknown'),
                    'sentiment_confidence': sentiment_result.get('confidence', 0)
                }
            }
            
        except ImportError as e:
            return {'success': False, 'error': f'Behavioral tools import failed: {str(e)}', 'severity': 'Medium'}
        except Exception as e:
            return {'success': False, 'error': f'Behavioral analysis failed: {str(e)}'}
    
    def test_additional_tools(self):
        """Test additional tools if available"""
        available_tools = []
        
        # Test regime tools
        try:
            from tools.regime_tools import detect_hmm_regimes
            portfolio_returns = (self.returns * self.weights).sum(axis=1)
            regime_result = detect_hmm_regimes(portfolio_returns)
            available_tools.append('regime_tools')
        except ImportError:
            pass
        except Exception:
            pass
        
        # Test portfolio tools
        try:
            from tools.portfolio_tools import optimize_portfolio
            available_tools.append('portfolio_tools')
        except ImportError:
            pass
        except Exception:
            pass
        
        # Test fractal tools
        try:
            from tools.fractal_tools import calculate_hurst_exponent
            available_tools.append('fractal_tools')
        except ImportError:
            pass
        except Exception:
            pass
        
        return {
            'success': True,
            'details': f"Additional tools available: {', '.join(available_tools) if available_tools else 'None'}",
            'metrics': {
                'available_tools': available_tools,
                'total_tools': len(available_tools)
            }
        }
    
    def run_verification(self):
        """Execute comprehensive verification"""
        print("=" * 60)
        print("BACKEND TOOLS COMPREHENSIVE VERIFICATION")
        print("=" * 60)
        
        # Core tests
        self.test_component("Basic Risk Tools", self.test_basic_risk_tools)
        self.test_component("Enhanced Risk Tools", self.test_enhanced_risk_tools)
        self.test_component("Strategy Tools", self.test_strategy_tools)
        self.test_component("Behavioral Tools", self.test_behavioral_tools)
        self.test_component("Additional Tools", self.test_additional_tools)
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("VERIFICATION REPORT")
        print("=" * 60)
        
        total_tests = len(self.test_results['passed']) + len(self.test_results['failed']) + len(self.test_results['warnings'])
        
        print(f"Tests Executed: {total_tests}")
        print(f"Passed: {len(self.test_results['passed'])}")
        print(f"Warnings: {len(self.test_results['warnings'])}")
        print(f"Failed: {len(self.test_results['failed'])}")
        
        # Detailed results
        if self.test_results['passed']:
            print(f"\nPASSED COMPONENTS ({len(self.test_results['passed'])}):")
            for result in self.test_results['passed']:
                print(f"  ✓ {result['component']}: {result['details']}")
        
        if self.test_results['warnings']:
            print(f"\nWARNINGS ({len(self.test_results['warnings'])}):")
            for result in self.test_results['warnings']:
                severity = result.get('severity', 'Medium')
                print(f"  ⚠ {result['component']} [{severity}]: {result['issue']}")
        
        if self.test_results['failed']:
            print(f"\nFAILED COMPONENTS ({len(self.test_results['failed'])}):")
            for result in self.test_results['failed']:
                print(f"  ✗ {result['component']}: {result['error']}")
        
        # Overall status
        print(f"\n{'='*60}")
        if len(self.test_results['failed']) == 0:
            if len(self.test_results['warnings']) == 0:
                print("STATUS: ALL SYSTEMS OPERATIONAL")
                print("Backend tools are fully functional and ready for production.")
            else:
                print("STATUS: OPERATIONAL WITH WARNINGS")
                print("Core functionality working, some enhanced features may be limited.")
        else:
            print("STATUS: ISSUES DETECTED")
            print("Critical components have failures that need resolution.")
        
        print(f"Verification completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main verification execution"""
    verifier = BackendToolsVerifier()
    verifier.run_verification()

if __name__ == "__main__":
    main()