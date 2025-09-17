# diagnostic_committee_test.py - Run this to diagnose committee manager issues

import sys
import traceback

def test_committee_manager_imports():
    """Test if all committee manager imports work"""
    print("Testing Committee Manager Imports...")
    
    try:
        print("1. Testing base agent import...")
        from agents.base_agent import BaseAgent
        print("✅ Base agent imported successfully")
    except Exception as e:
        print(f"❌ Base agent import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("2. Testing quantitative analyst import...")
        from agents.chat_quantitative_analyst import ChatQuantitativeAnalyst, QuantAnalystChatIntegration
        print("✅ Quantitative analyst imported successfully")
    except Exception as e:
        print(f"❌ Quantitative analyst import failed: {e}")
        traceback.print_exc()
    
    try:
        print("3. Testing CIO import...")
        from agents.chat_cio_agent import ChatCIOAgent, CIOChatIntegration
        print("✅ CIO imported successfully")
    except Exception as e:
        print(f"❌ CIO import failed: {e}")
        traceback.print_exc()
    
    try:
        print("4. Testing Portfolio Manager import...")
        from agents.chat_portfolio_manager import ChatPortfolioManager, PMChatIntegration
        print("✅ Portfolio Manager imported successfully")
    except Exception as e:
        print(f"❌ Portfolio Manager import failed: {e}")
        traceback.print_exc()
    
    try:
        print("5. Testing Behavioral Coach import...")
        from agents.chat_behavioral_coach import ChatBehavioralCoach, BehavioralChatIntegration
        print("✅ Behavioral Coach imported successfully")
    except Exception as e:
        print(f"❌ Behavioral Coach import failed: {e}")
        traceback.print_exc()
    
    print("\n" + "="*50)

def test_committee_manager_initialization():
    """Test committee manager initialization"""
    print("Testing Committee Manager Initialization...")
    
    try:
        from agents.committee_manager import InvestmentCommitteeManager
        print("✅ Committee manager imported successfully")
        
        manager = InvestmentCommitteeManager()
        print(f"✅ Committee manager initialized with {len(manager.specialists)} specialists")
        
        print("\nRegistered specialists:")
        for specialist_id in manager.specialists.keys():
            print(f"  - {specialist_id}")
        
        print("\nAvailable specialists info:")
        available = manager.get_available_specialists()
        for specialist in available:
            status = "✅" if specialist.get("available", False) else "❌"
            print(f"  {status} {specialist['name']} ({specialist['id']})")
        
        return manager
        
    except Exception as e:
        print(f"❌ Committee manager initialization failed: {e}")
        traceback.print_exc()
        return None

def test_specialist_routing():
    """Test specialist routing"""
    print("\n" + "="*50)
    print("Testing Specialist Routing...")
    
    manager = test_committee_manager_initialization()
    if not manager:
        return
    
    test_queries = [
        ("How risky is my portfolio?", "quantitative_analyst"),
        ("What's the market outlook?", "cio"),
        ("Should I rebalance?", "portfolio_manager"),
        ("I want to panic sell", "behavioral_coach")
    ]
    
    for query, expected_specialist in test_queries:
        try:
            routed_specialist = manager._route_to_specialist(query)
            status = "✅" if expected_specialist in routed_specialist else "❌"
            print(f"  {status} '{query}' → {routed_specialist} (expected: {expected_specialist})")
        except Exception as e:
            print(f"  ❌ Routing failed for '{query}': {e}")

def test_chat_endpoints():
    """Test chat endpoints import"""
    print("\n" + "="*50)
    print("Testing Chat Endpoints...")
    
    try:
        from auth.chat_endpoints import router, committee
        print("✅ Chat endpoints imported successfully")
        
        # Test committee instance
        available = committee.get_available_specialists()
        print(f"✅ Committee in endpoints has {len(available)} specialists")
        
        for specialist in available:
            status = "✅" if specialist.get("available", False) else "❌"
            print(f"  {status} {specialist['name']}")
            
    except Exception as e:
        print(f"❌ Chat endpoints test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Investment Committee Diagnostic Test")
    print("=" * 50)
    
    test_committee_manager_imports()
    test_committee_manager_initialization()
    test_specialist_routing()
    test_chat_endpoints()
    
    print("\n" + "="*50)
    print("Diagnostic Complete!")