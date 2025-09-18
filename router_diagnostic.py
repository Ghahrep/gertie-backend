
"""
Diagnostic script to verify semantic router version and configuration
"""

import sys
import os

# Add your project root to path if needed
sys.path.append('/path/to/your/project')

def diagnose_router():
    """Diagnose router configuration and version"""
    print("🔍 ROUTER DIAGNOSTIC REPORT")
    print("=" * 50)
    
    try:
        from agents.enhanced_investment_committee_manager import EnhancedInvestmentCommitteeManager
        manager = EnhancedInvestmentCommitteeManager()
        
        print(f"✅ Committee Manager loaded successfully")
        print(f"   Specialists available: {len(manager.specialists)}")
        
        # Check semantic router
        if hasattr(manager, 'semantic_router') and manager.semantic_router:
            router = manager.semantic_router
            print(f"✅ Semantic Router initialized")
            
            # Check router version/analytics
            if hasattr(router, 'get_routing_analytics'):
                analytics = router.get_routing_analytics()
                print(f"   Router Version: {analytics.get('version', 'unknown')}")
                print(f"   Features: {len(analytics.get('routing_features', []))}")
                
                # Check specific tuned features
                features = analytics.get('routing_features', [])
                if 'diversity_preference' in features:
                    print("   ✅ Tuned router features detected")
                else:
                    print("   ⚠️  Original router (no tuned features)")
            
            # Check agent profiles
            if hasattr(router, 'agent_profiles'):
                profiles = router.agent_profiles
                print(f"   Agent profiles configured: {len(profiles)}")
                
                # Check for tuned descriptions
                pm_description = profiles.get('portfolio_manager', {}).get('description', '')
                if 'rebalancing' in pm_description:
                    print("   ✅ Enhanced agent descriptions found")
                else:
                    print("   ⚠️  Basic agent descriptions")
                
                # Check confidence weights
                weights = {name: profile.get('confidence_weight', 1.0) 
                          for name, profile in profiles.items()}
                print(f"   Confidence weights: {weights}")
                
                if weights.get('quantitative_analyst', 1.0) == 1.0:
                    print("   ✅ Tuned weights applied (quant=1.0)")
                else:
                    print("   ⚠️  Original weights (quant=1.2)")
        
        else:
            print("❌ Semantic Router not initialized")
            print("   Falling back to keyword routing")
        
        # Test a simple routing decision
        print(f"\n🧪 ROUTING TEST:")
        test_query = "Should I rebalance my portfolio allocation?"
        
        conversation = manager._get_or_create_enhanced_conversation(
            "test_conv", {"user_id": "test"}, "test"
        )
        
        agent, confidence = manager._enhanced_route_to_specialist_with_confidence(
            test_query, conversation, []
        )
        
        print(f"   Query: '{test_query}'")
        print(f"   Routed to: {agent}")
        print(f"   Confidence: {confidence}%")
        
        if agent == "portfolio_manager":
            print("   ✅ Correct routing (expected portfolio_manager)")
        else:
            print("   ❌ Incorrect routing (expected portfolio_manager)")
            print("   This suggests router tuning hasn't been applied")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(traceback.format_exc())
    
    print(f"\n📁 FILE CHECK:")
    
    # Check if files exist
    files_to_check = [
        "agents/semantic_router.py",
        "agents/enhanced_memory.py", 
        "agents/enhanced_committee_manager.py"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path} exists")
            
            # Check file modification time
            mod_time = os.path.getmtime(file_path)
            import datetime
            mod_date = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            print(f"      Last modified: {mod_date}")
            
        else:
            print(f"   ❌ {file_path} missing")
    
    print(f"\n💡 RECOMMENDATIONS:")
    
    # Check if tuned router is being used
    try:
        from agents.semantic_router import SemanticAgentRouter
        router = SemanticAgentRouter()
        analytics = router.get_routing_analytics()
        
        if analytics.get('version') == 'tuned_v1.1':
            print("   ✅ Tuned router is available")
            print("   ⚠️  But committee manager may not be using it")
            print("   💡 Try restarting Python to reload modules")
        else:
            print("   ⚠️  Router file needs to be updated with tuned version")
            
    except Exception as e:
        print(f"   ⚠️  Could not load semantic router: {e}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    diagnose_router()