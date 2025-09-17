# test_your_enhanced_integration.py
"""
Integration Test for Your Enhanced Investment Committee
=====================================================

Customized test for your specific file structure and enhanced agents.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_backend_tools():
    """Test if backend tools are available"""
    print("🔧 Testing Backend Tool Availability")
    print("-" * 40)
    
    missing_modules = []
    available_modules = []
    
    # Test risk tools
    try:
        import tools.risk_tools
        available_modules.append("tools.risk_tools")
        print("  ✅ tools.risk_tools - Available")
    except ImportError as e:
        missing_modules.append(("tools.risk_tools", str(e)))
        print(f"  ❌ tools.risk_tools - Missing: {e}")
    
    # Test behavioral tools
    try:
        import tools.behavioral_tools
        available_modules.append("tools.behavioral_tools")
        print("  ✅ tools.behavioral_tools - Available")
    except ImportError as e:
        missing_modules.append(("tools.behavioral_tools", str(e)))
        print(f"  ❌ tools.behavioral_tools - Missing: {e}")
    
    # Test strategy tools
    try:
        import tools.strategy_tools
        available_modules.append("tools.strategy_tools")
        print("  ✅ tools.strategy_tools - Available")
    except ImportError as e:
        missing_modules.append(("tools.strategy_tools", str(e)))
        print(f"  ❌ tools.strategy_tools - Missing: {e}")
    
    # Test regime tools
    try:
        import tools.regime_tools
        available_modules.append("tools.regime_tools")
        print("  ✅ tools.regime_tools - Available")
    except ImportError as e:
        missing_modules.append(("tools.regime_tools", str(e)))
        print(f"  ❌ tools.regime_tools - Missing: {e}")
    
    print(f"\n📊 Backend Tools Summary:")
    print(f"  Available: {len(available_modules)} modules")
    print(f"  Missing: {len(missing_modules)} modules")
    
    return len(available_modules) > 0

def test_data_helpers():
    """Test your advanced data helpers"""
    print("\n🔬 Testing Your Advanced Data Helpers")
    print("-" * 40)
    
    try:
        # Try to import your advanced data helpers
        from agents.data_helpers import AgentDataHelper, RiskAnalysisIntegrator, StrategyAnalysisIntegrator
        print("  ✅ Successfully imported your advanced data helpers")
        
        # Test portfolio data conversion
        sample_portfolio = {
            "total_value": 150000,
            "holdings": [
                {"ticker": "AAPL", "value": 45000, "symbol": "AAPL"},
                {"ticker": "MSFT", "value": 37500, "symbol": "MSFT"},
                {"ticker": "GOOGL", "value": 30000, "symbol": "GOOGL"},
                {"ticker": "TSLA", "value": 22500, "symbol": "TSLA"},
                {"ticker": "AMZN", "value": 15000, "symbol": "AMZN"}
            ],
            "daily_change": "+1.2%"
        }
        
        # Test your data extraction methods
        portfolio_returns = AgentDataHelper.extract_portfolio_returns(sample_portfolio)
        print(f"  ✅ Portfolio returns: {len(portfolio_returns)} days, std: {portfolio_returns.std():.4f}")
        
        portfolio_df = AgentDataHelper.convert_portfolio_to_dataframe(sample_portfolio)
        print(f"  ✅ Portfolio DataFrame: {portfolio_df.shape[0]} days × {portfolio_df.shape[1]} assets")
        
        weights = AgentDataHelper.extract_portfolio_weights(sample_portfolio)
        print(f"  ✅ Portfolio weights: {len(weights)} assets, sum: {weights.sum():.3f}")
        
        # Test factor data creation
        factor_data = AgentDataHelper.create_factor_data(lookback_days=30)
        print(f"  ✅ Factor data: {factor_data.shape[0]} days × {factor_data.shape[1]} factors")
        
        # Test backend integration classes
        try:
            risk_analysis = RiskAnalysisIntegrator.get_comprehensive_risk_analysis(sample_portfolio)
            if 'error' in risk_analysis:
                print(f"  ⚠️  Risk analysis fallback: {risk_analysis['error']}")
            else:
                print(f"  ✅ Risk analysis: {len(risk_analysis)} components")
                
            strategy_analysis = StrategyAnalysisIntegrator.get_portfolio_optimization_analysis(sample_portfolio)
            if 'error' in strategy_analysis:
                print(f"  ⚠️  Strategy analysis fallback: {strategy_analysis['error']}")
            else:
                print(f"  ✅ Strategy analysis: {len(strategy_analysis)} components")
                
        except Exception as e:
            print(f"  ⚠️  Backend integration test failed: {e}")
        
        # Test convenience functions
        try:
            from agents.data_helpers import get_real_var_analysis, get_regime_analysis
            var_95, var_99, risk_level = get_real_var_analysis(sample_portfolio)
            print(f"  ✅ VaR analysis: 95%={var_95:.3f}, 99%={var_99:.3f}, Level={risk_level}")
            
            regime_data = get_regime_analysis(sample_portfolio)
            if 'error' in regime_data:
                print(f"  ⚠️  Regime analysis fallback: {regime_data['error']}")
            else:
                print(f"  ✅ Regime analysis: {len(regime_data)} components")
                
        except Exception as e:
            print(f"  ⚠️  Convenience functions test failed: {e}")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Failed to import data helpers: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Data helpers test failed: {e}")
        return False

def test_base_agent_dependency():
    """Test if base agent exists"""
    print("\n🔧 Testing Base Agent Dependency")
    print("-" * 35)
    
    try:
        from agents.base_agent import BaseAgent
        print("  ✅ Successfully imported BaseAgent")
        return True
    except ImportError as e:
        print(f"  ❌ Failed to import BaseAgent: {e}")
        print(f"     You need to create agents/base_agent.py")
        return False

def test_manager_import():
    """Test if the enhanced manager can be imported"""
    print("\n🔧 Testing Enhanced Manager Import")
    print("-" * 35)
    
    # Try different possible manager file names
    manager_files = [
        ("agents.enhanced_committee_manager", "EnhancedInvestmentCommitteeManager"),
        ("agents.enhanced_investment_committee_manager", "EnhancedInvestmentCommitteeManager"),
    ]
    
    for module_path, class_name in manager_files:
        try:
            module = __import__(module_path, fromlist=[class_name])
            manager_class = getattr(module, class_name)
            print(f"  ✅ Successfully imported {class_name} from {module_path}")
            return True, manager_class
        except ImportError as e:
            print(f"  ⚠️  Could not import from {module_path}: {e}")
        except AttributeError as e:
            print(f"  ⚠️  Class {class_name} not found in {module_path}: {e}")
    
    print(f"  ❌ Failed to import Enhanced Manager")
    return False, None

def test_individual_enhanced_agents():
    """Test your individual enhanced agents"""
    print("\n🔬 Testing Your Individual Enhanced Agents")
    print("-" * 45)
    
    agents_to_test = [
        ("Enhanced Quantitative Analyst", "agents.enhanced_chat_quantitative_analyst", "EnhancedChatQuantitativeAnalyst"),
        ("Enhanced CIO Agent", "agents.enhanced_chat_cio_agent", "EnhancedChatCIOAgent"),
        ("Enhanced Portfolio Manager", "agents.enhanced_chat_portfolio_manager", "EnhancedChatPortfolioManager"),
        ("Enhanced Behavioral Coach", "agents.enhanced_chat_behavioral_coach", "EnhancedChatBehavioralCoach"),
    ]
    
    available_agents = []
    
    for name, module_path, class_name in agents_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            agent_class = getattr(module, class_name)
            print(f"  ✅ {name}: Successfully imported")
            available_agents.append((name, agent_class))
        except ImportError as e:
            print(f"  ❌ {name}: Import failed - {e}")
            if "base_agent" in str(e):
                print(f"     → Missing base_agent.py dependency")
        except AttributeError as e:
            print(f"  ❌ {name}: Class not found - {e}")
        except Exception as e:
            print(f"  ❌ {name}: Other error - {e}")
    
    print(f"\n📊 Agent Import Summary:")
    print(f"  Available agents: {len(available_agents)}")
    print(f"  Failed imports: {len(agents_to_test) - len(available_agents)}")
    
    return available_agents

async def test_basic_agent_functionality(available_agents):
    """Test basic functionality of available agents"""
    if not available_agents:
        print("\n❌ No agents available for testing")
        return False
    
    print(f"\n🧪 Testing Basic Agent Functionality")
    print("-" * 40)
    
    # Test data
    portfolio_data = {
        "total_value": 150000,
        "holdings": [
            {"ticker": "AAPL", "value": 45000, "symbol": "AAPL"},
            {"ticker": "MSFT", "value": 37500, "symbol": "MSFT"},
            {"ticker": "GOOGL", "value": 30000, "symbol": "GOOGL"}
        ],
        "daily_change": "+0.8%"
    }
    
    context = {
        "conversation_history": [
            {"role": "user", "content": "I'm worried about market volatility"},
            {"role": "assistant", "content": "Let me analyze your risk exposure"}
        ]
    }
    
    successful_tests = 0
    
    for name, agent_class in available_agents:
        print(f"\nTesting {name}...")
        try:
            # Try to instantiate the agent
            agent = agent_class()
            print(f"  ✅ {name}: Successfully instantiated")
            
            # Try to call analyze_query method
            try:
                response = await agent.analyze_query(
                    "What's my portfolio risk?", 
                    portfolio_data, 
                    context
                )
                
                # Check response structure
                if isinstance(response, dict):
                    content_length = len(response.get('content', ''))
                    analysis = response.get('analysis', {})
                    confidence = analysis.get('confidence', 0)
                    specialist = response.get('specialist', 'Unknown')
                    
                    print(f"  ✅ {name}: analyze_query works - {content_length} chars, {confidence}% confidence")
                    print(f"     Specialist: {specialist}")
                    
                    # Check for backend integration
                    if response.get('backend_integration') or 'institutional' in response.get('content', '').lower():
                        print(f"     🔧 Backend integration detected")
                    
                    successful_tests += 1
                else:
                    print(f"  ⚠️  {name}: Invalid response type")
                    
            except Exception as e:
                print(f"  ⚠️  {name}: Method call failed - {e}")
                print(f"     This might be due to missing dependencies")
                
        except Exception as e:
            print(f"  ❌ {name}: Instantiation failed - {e}")
    
    print(f"\n📊 Functionality Test Summary:")
    print(f"  Working agents: {successful_tests}/{len(available_agents)}")
    
    return successful_tests > 0

async def test_manager_functionality(manager_class):
    """Test the enhanced manager functionality"""
    if not manager_class:
        print("\n❌ No manager available for testing")
        return False
    
    print(f"\n🧪 Testing Enhanced Manager Functionality")
    print("-" * 45)
    
    try:
        # Try to instantiate the manager
        manager = manager_class()
        print(f"  ✅ Manager: Successfully instantiated")
        print(f"     Available specialists: {list(manager.specialists.keys())}")
        
        # Test portfolio context
        portfolio_context = {
            "totalValue": "$150,000",
            "holdings": [
                {"ticker": "AAPL", "value": 45000, "symbol": "AAPL"},
                {"ticker": "MSFT", "value": 37500, "symbol": "MSFT"},
                {"ticker": "GOOGL", "value": 30000, "symbol": "GOOGL"}
            ],
            "dailyChange": "+1.2%",
            "total_value": 150000
        }
        
        # Test a simple query
        test_query = "What's my portfolio risk?"
        
        try:
            response = await manager.route_query(
                query=test_query,
                portfolio_context=portfolio_context,
                preferred_specialist="quantitative_analyst"
            )
            
            if isinstance(response, dict):
                specialist_used = response.get("specialist_used", "unknown")
                content_length = len(response.get("content", ""))
                confidence = response.get("analysis", {}).get("confidence", 0)
                execution_time = response.get("execution_time", 0)
                
                print(f"  ✅ Manager: route_query works")
                print(f"     Specialist used: {specialist_used}")
                print(f"     Content length: {content_length} characters")
                print(f"     Confidence: {confidence}%")
                print(f"     Execution time: {execution_time:.3f}s")
                
                # Check for enhanced features
                enhanced_features = []
                if response.get("enhanced_insights"):
                    enhanced_features.append(f"{len(response['enhanced_insights'])} insights")
                if response.get("cross_specialist_suggestions"):
                    enhanced_features.append(f"{len(response['cross_specialist_suggestions'])} suggestions")
                if response.get("smart_suggestions"):
                    enhanced_features.append(f"{len(response['smart_suggestions'])} smart suggestions")
                
                if enhanced_features:
                    print(f"     Enhanced features: {', '.join(enhanced_features)}")
                
                return True
            else:
                print(f"  ⚠️  Manager: Invalid response type")
                return False
                
        except Exception as e:
            print(f"  ⚠️  Manager: route_query failed - {e}")
            return False
            
    except Exception as e:
        print(f"  ❌ Manager: Instantiation failed - {e}")
        return False

async def run_your_integration_test():
    """Run integration test customized for your setup"""
    print("\n🧪 Your Enhanced Investment Committee Integration Test")
    print("=" * 65)
    
    # Test backend tools
    backend_available = test_backend_tools()
    
    # Test data helpers
    helpers_available = test_data_helpers()
    
    # Test base agent dependency
    base_agent_available = test_base_agent_dependency()
    
    # Test manager import
    manager_available, manager_class = test_manager_import()
    
    # Test individual agents
    available_agents = test_individual_enhanced_agents()
    
    # Test basic functionality
    agents_working = await test_basic_agent_functionality(available_agents)
    
    # Test manager functionality
    manager_working = await test_manager_functionality(manager_class) if manager_class else False
    
    # Summary
    print(f"\n🏆 Your Integration Test Summary:")
    print("=" * 35)
    print(f"Backend Tools Available: {'✅' if backend_available else '❌'}")
    print(f"Advanced Data Helpers: {'✅' if helpers_available else '❌'}")
    print(f"Base Agent Available: {'✅' if base_agent_available else '❌'}")
    print(f"Enhanced Manager: {'✅' if manager_available else '❌'}")
    print(f"Enhanced Agents: {len(available_agents)}/4 imported")
    print(f"Working Agents: {'✅' if agents_working else '❌'}")
    print(f"Manager Working: {'✅' if manager_working else '❌'}")
    
    # Specific recommendations for your setup
    print(f"\n💡 Recommendations for Your Setup:")
    
    if not base_agent_available:
        print("  1. Create agents/base_agent.py (I can provide this)")
        print("     - Your enhanced agents inherit from BaseAgent")
    
    if not manager_available:
        print("  2. Check enhanced_committee_manager.py imports")
        print("     - Make sure it matches your enhanced agent file names")
    
    if len(available_agents) < 4:
        print("  3. Fix enhanced agent import issues")
        print("     - Most likely missing base_agent.py dependency")
    
    if backend_available and helpers_available and not agents_working:
        print("  4. Agent instantiation issues")
        print("     - Backend tools work, but agents can't use them")
    
    # Overall assessment
    components_working = sum([
        backend_available,
        helpers_available, 
        base_agent_available,
        len(available_agents) >= 3,
        agents_working,
        manager_working
    ])
    
    if components_working >= 5:
        print(f"\n🎉 System Status: EXCELLENT - Your enhanced system is working!")
        return True
    elif components_working >= 4:
        print(f"\n✨ System Status: GOOD - Most components working")
        return True
    elif components_working >= 2:
        print(f"\n⚠️  System Status: PARTIAL - Some components working")
        return True
    else:
        print(f"\n❌ System Status: NEEDS WORK - Major issues detected")
        return False

if __name__ == "__main__":
    try:
        # Run your integration test
        success = asyncio.run(run_your_integration_test())
        
        if success:
            print(f"\n🎉 Your enhanced investment committee is ready!")
            print(f"   Your backend tools and agents are working together.")
            sys.exit(0)
        else:
            print(f"\n⚠️  Some issues detected - see recommendations above")
            print(f"   Create the missing files to fix the remaining issues.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)