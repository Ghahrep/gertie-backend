# test_practical_integration.py
"""
Practical Integration Test - Works with Your Actual Files
========================================================

Tests your actual enhanced agents without requiring stub files.
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
    
    available_modules = []
    
    for tool_name in ["risk_tools", "behavioral_tools", "strategy_tools", "regime_tools"]:
        try:
            module = __import__(f"tools.{tool_name}", fromlist=[tool_name])
            available_modules.append(tool_name)
            print(f"  ✅ tools.{tool_name} - Available")
        except ImportError as e:
            print(f"  ❌ tools.{tool_name} - Missing: {e}")
    
    print(f"\n📊 Backend Tools Summary: {len(available_modules)}/4 available")
    return len(available_modules) > 0

def test_data_helpers_directly():
    """Test data helpers by importing individual classes"""
    print("\n🔬 Testing Data Helpers (Direct Import)")
    print("-" * 45)
    
    try:
        # Import specific classes to avoid problematic imports
        import importlib.util
        
        # Load the data helpers module manually
        spec = importlib.util.spec_from_file_location(
            "data_helpers", 
            "agents/data_helpers.py"
        )
        data_helpers = importlib.util.module_from_spec(spec)
        
        try:
            spec.loader.exec_module(data_helpers)
            print("  ✅ Successfully loaded data_helpers.py")
            
            # Test AgentDataHelper if it exists
            if hasattr(data_helpers, 'AgentDataHelper'):
                AgentDataHelper = data_helpers.AgentDataHelper
                print("  ✅ AgentDataHelper class found")
                
                # Test basic functionality
                sample_portfolio = {
                    "total_value": 150000,
                    "holdings": [
                        {"ticker": "AAPL", "value": 45000, "symbol": "AAPL"},
                        {"ticker": "MSFT", "value": 37500, "symbol": "MSFT"}
                    ],
                    "daily_change": "+1.2%"
                }
                
                # Test methods
                try:
                    portfolio_returns = AgentDataHelper.extract_portfolio_returns(sample_portfolio)
                    print(f"  ✅ extract_portfolio_returns: {len(portfolio_returns)} days")
                    
                    portfolio_df = AgentDataHelper.convert_portfolio_to_dataframe(sample_portfolio)
                    print(f"  ✅ convert_portfolio_to_dataframe: {portfolio_df.shape}")
                    
                    weights = AgentDataHelper.extract_portfolio_weights(sample_portfolio)
                    print(f"  ✅ extract_portfolio_weights: {len(weights)} assets")
                    
                    return True
                    
                except Exception as e:
                    print(f"  ⚠️ AgentDataHelper methods failed: {e}")
                    return False
            else:
                print("  ❌ AgentDataHelper class not found")
                return False
                
        except Exception as e:
            print(f"  ❌ Failed to execute data_helpers.py: {e}")
            return False
            
    except Exception as e:
        print(f"  ❌ Failed to load data_helpers.py: {e}")
        return False

def test_enhanced_agents_directly():
    """Test enhanced agents by importing them directly"""
    print("\n🔬 Testing Enhanced Agents (Direct Import)")
    print("-" * 45)
    
    agent_files = {
        "Enhanced Quantitative Analyst": "enhanced_chat_quantitative_analyst.py",
        "Enhanced CIO Agent": "enhanced_chat_cio_agent.py", 
        "Enhanced Portfolio Manager": "enhanced_chat_portfolio_manager.py",
        "Enhanced Behavioral Coach": "enhanced_chat_behavioral_coach.py"
    }
    
    available_agents = []
    
    for name, filename in agent_files.items():
        try:
            import importlib.util
            
            # Load the agent module manually
            spec = importlib.util.spec_from_file_location(
                filename.replace('.py', ''), 
                f"agents/{filename}"
            )
            agent_module = importlib.util.module_from_spec(spec)
            
            try:
                spec.loader.exec_module(agent_module)
                print(f"  ✅ {name}: File loaded successfully")
                
                # Try to find the agent class - handle special cases
                if filename == "enhanced_chat_cio_agent.py":
                    class_name = "EnhancedChatCIOAgent"  # Special case for CIO with capital letters
                else:
                    class_name = filename.replace('.py', '').replace('_', ' ').title().replace(' ', '')
                
                if hasattr(agent_module, class_name):
                    agent_class = getattr(agent_module, class_name)
                    available_agents.append((name, agent_class))
                    print(f"      Class {class_name} found")
                else:
                    print(f"      ⚠️ Class {class_name} not found in module")
                    
            except Exception as e:
                print(f"  ❌ {name}: Failed to execute - {e}")
                if "base_agent" in str(e).lower():
                    print(f"      → Missing base_agent dependency")
                elif "chat_quantitative_analyst" in str(e):
                    print(f"      → Missing base agent stub files")
                    
        except Exception as e:
            print(f"  ❌ {name}: Failed to load - {e}")
    
    print(f"\n📊 Enhanced Agents Summary: {len(available_agents)}/4 loaded")
    return available_agents

def test_enhanced_manager_directly():
    """Test enhanced manager by importing it directly"""
    print("\n🔧 Testing Enhanced Manager (Direct Import)")
    print("-" * 45)
    
    manager_files = [
        "enhanced_committee_manager.py",
        "enhanced_investment_committee_manager.py"
    ]
    
    for filename in manager_files:
        try:
            import importlib.util
            
            # Check if file exists
            filepath = f"agents/{filename}"
            if not os.path.exists(filepath):
                print(f"  ⚠️ {filename}: File not found")
                continue
                
            # Load the manager module manually
            spec = importlib.util.spec_from_file_location(
                filename.replace('.py', ''), 
                filepath
            )
            manager_module = importlib.util.module_from_spec(spec)
            
            try:
                spec.loader.exec_module(manager_module)
                print(f"  ✅ {filename}: File loaded successfully")
                
                # Try to find the manager class
                if hasattr(manager_module, 'EnhancedInvestmentCommitteeManager'):
                    manager_class = manager_module.EnhancedInvestmentCommitteeManager
                    print(f"      EnhancedInvestmentCommitteeManager class found")
                    return True, manager_class
                else:
                    print(f"      ⚠️ EnhancedInvestmentCommitteeManager class not found")
                    
            except Exception as e:
                print(f"  ❌ {filename}: Failed to execute - {e}")
                if "chat_quantitative_analyst" in str(e):
                    print(f"      → Missing base agent dependencies")
                    
        except Exception as e:
            print(f"  ❌ {filename}: Failed to load - {e}")
    
    print(f"  ❌ No working manager found")
    return False, None

async def test_agent_functionality_minimal(available_agents):
    """Test basic functionality with minimal requirements"""
    if not available_agents:
        print("\n❌ No agents available for functionality testing")
        return False
    
    print(f"\n🧪 Testing Agent Functionality (Minimal)")
    print("-" * 45)
    
    portfolio_data = {
        "total_value": 150000,
        "holdings": [
            {"ticker": "AAPL", "value": 45000, "symbol": "AAPL"},
            {"ticker": "MSFT", "value": 37500, "symbol": "MSFT"}
        ],
        "daily_change": "+0.8%"
    }
    
    context = {"conversation_history": []}
    
    working_agents = 0
    
    for name, agent_class in available_agents:
        print(f"\nTesting {name}...")
        try:
            # Try to instantiate
            agent = agent_class()
            print(f"  ✅ Instantiation successful")
            
            # Try to call analyze_query
            try:
                response = await agent.analyze_query(
                    "What's my portfolio risk?",
                    portfolio_data,
                    context
                )
                
                if isinstance(response, dict) and 'content' in response:
                    content_length = len(response['content'])
                    print(f"  ✅ analyze_query works: {content_length} chars")
                    working_agents += 1
                else:
                    print(f"  ⚠️ analyze_query returned invalid response")
                    
            except Exception as e:
                print(f"  ❌ analyze_query failed: {e}")
                
        except Exception as e:
            print(f"  ❌ Instantiation failed: {e}")
    
    print(f"\n📊 Functionality Summary: {working_agents}/{len(available_agents)} working")
    return working_agents > 0

def provide_fix_recommendations():
    """Provide specific recommendations to fix the issues"""
    print(f"\n💡 Specific Fix Recommendations:")
    print("=" * 35)
    
    print(f"\n1. Fix Data Helpers Import Issues:")
    print(f"   Edit agents/data_helpers.py and remove/comment out these lines:")
    print(f"   - Any imports referencing 'chat_quantitative_analyst'")
    print(f"   - Any imports referencing non-existent base agent files")
    print(f"   - Replace with direct backend tool imports")
    
    print(f"\n2. Fix Enhanced Agent Import Issues:")
    print(f"   In each enhanced agent file, check the imports at the top:")
    print(f"   - Make sure .base_agent import works (you have this file)")
    print(f"   - Remove any imports of non-existent base agents")
    print(f"   - Use direct backend tool imports instead")
    
    print(f"\n3. Alternative Approach - Modify Imports:")
    print(f"   Instead of creating stub files, modify your existing files:")
    print(f"   - Comment out problematic imports temporarily")
    print(f"   - Use try/except blocks around imports")
    print(f"   - Import backend tools directly where needed")
    
    print(f"\n4. Quick Fix for Testing:")
    print(f"   Create a simplified version of your agents that doesn't")
    print(f"   depend on the problematic imports, just for testing.")

async def run_practical_test():
    """Run practical test that works with your actual files"""
    print("\n🧪 Practical Integration Test - Your Actual Files")
    print("=" * 55)
    
    # Test each component
    backend_available = test_backend_tools()
    helpers_working = test_data_helpers_directly()
    available_agents = test_enhanced_agents_directly()
    manager_working, manager_class = test_enhanced_manager_directly()
    
    # Test functionality if agents are available
    agents_functional = False
    if available_agents:
        agents_functional = await test_agent_functionality_minimal(available_agents)
    
    # Summary
    print(f"\n🏆 Practical Test Results:")
    print("=" * 25)
    print(f"Backend Tools: {'✅' if backend_available else '❌'}")
    print(f"Data Helpers: {'✅' if helpers_working else '❌'}")
    print(f"Enhanced Agents: {len(available_agents)}/4 loaded")
    print(f"Manager: {'✅' if manager_working else '❌'}")
    print(f"Functionality: {'✅' if agents_functional else '❌'}")
    
    # Determine status
    total_score = sum([
        backend_available,
        helpers_working,
        len(available_agents) > 0,
        manager_working,
        agents_functional
    ])
    
    if total_score >= 4:
        print(f"\n🎉 Status: GOOD - Most components working!")
        print(f"   Your backend integration is solid.")
        return True
    elif total_score >= 2:
        print(f"\n⚠️ Status: NEEDS FIXES - Some components working")
        provide_fix_recommendations()
        return True
    else:
        print(f"\n❌ Status: NEEDS WORK - Import issues need fixing")
        provide_fix_recommendations()
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(run_practical_test())
        
        if success:
            print(f"\n🎯 Next Steps:")
            print(f"   Fix the import issues in your existing files")
            print(f"   Your backend tools and overall structure are solid!")
        else:
            print(f"\n🔧 Focus on fixing import dependencies first")
            
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()