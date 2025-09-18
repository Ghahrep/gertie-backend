# test_advanced_memory.py - Test Advanced Memory System
"""
Test script for Week 1 Advanced Memory Implementation
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_advanced_memory_system():
    """Test the advanced memory system functionality"""
    print("🧠 TESTING ADVANCED MEMORY SYSTEM")
    print("=" * 50)
    
    try:
        # Test 1: Basic Memory Creation
        print("\n1️⃣ Testing Memory Creation...")
        from agents.advanced_memory import create_advanced_memory_system, EnhancedConversationMemory
        
        memory_store = create_advanced_memory_system("test_memory")
        print(f"   ✅ Memory store created")
        
        # Test 2: Conversation Memory
        print("\n2️⃣ Testing Conversation Memory...")
        conversation = memory_store.get_conversation("test_conv_001", "test_user_123")
        print(f"   ✅ Conversation memory created: {conversation.conversation_id}")
        
        # Test 3: Add Conversation Turns
        print("\n3️⃣ Testing Conversation Turns...")
        
        # Simulate a series of investment conversations
        test_conversations = [
            {
                "query": "I'm worried about my portfolio's risk level",
                "response": "Your portfolio shows moderate risk with 65/100 risk score. Consider diversification.",
                "specialist": "quantitative_analyst", 
                "confidence": 85,
                "risk_score": 65,
                "collaboration": False
            },
            {
                "query": "Should I rebalance my tech-heavy allocation?",
                "response": "Yes, tech allocation is 45% vs target 25%. Recommend rebalancing.",
                "specialist": "portfolio_manager",
                "confidence": 90,
                "risk_score": 70,
                "collaboration": True,
                "secondary_specialists": ["cio"]
            },
            {
                "query": "I keep making emotional investment decisions",
                "response": "Identified loss aversion bias. Recommend systematic approach.",
                "specialist": "behavioral_coach",
                "confidence": 88,
                "risk_score": 55,
                "collaboration": True,
                "secondary_specialists": ["quantitative_analyst"]
            }
        ]
        
        for i, conv_data in enumerate(test_conversations):
            turn = conversation.add_turn(
                user_query=conv_data["query"],
                agent_response=conv_data["response"],
                specialist=conv_data["specialist"],
                confidence=conv_data["confidence"],
                risk_score=conv_data["risk_score"],
                collaboration_involved=conv_data["collaboration"],
                secondary_specialists=conv_data.get("secondary_specialists", []),
                portfolio_context={"total_value": 500000 + i*10000, "riskLevel": "MODERATE"}
            )
            print(f"   ✅ Added turn {i+1}: {conv_data['specialist']} analysis")
        
        print(f"   📊 Total conversation turns: {len(conversation.turns)}")
        
        # Test 4: Semantic Search
        print("\n4️⃣ Testing Semantic Search...")
        search_results = conversation.semantic_search("portfolio risk", top_k=2)
        print(f"   🔍 Search for 'portfolio risk' found {len(search_results)} results")
        
        for turn, similarity in search_results:
            print(f"   📝 Match: '{turn.user_query[:50]}...' (similarity: {similarity:.2f})")
        
        # Test 5: User Learning Insights
        print("\n5️⃣ Testing User Learning...")
        user_insights = conversation.get_user_learning_insights()
        print(f"   🎓 User expertise level: {user_insights.get('expertise_level', 'unknown')}")
        print(f"   📊 Complexity preference: {user_insights.get('complexity_preference', 0):.2f}")
        collaboration_pref = user_insights.get('collaboration_preference', 0)
        if isinstance(collaboration_pref, (int, float)):
            print(f"   🔄 Collaboration preference: {collaboration_pref:.2f}")
        else:
            print(f"   🔄 Collaboration preference: {collaboration_pref}")
            print(f"   🎯 Learning confidence: {user_insights.get('learning_confidence', 0):.2f}")
        
        # Test 6: Agent Satisfaction Learning
        print("\n6️⃣ Testing Agent Satisfaction Learning...")
        # Simulate user feedback
        conversation.turns[0].user_satisfaction = 0.8  # Good rating for quantitative analyst
        conversation.turns[1].user_satisfaction = 0.9  # Excellent rating for portfolio manager
        conversation.turns[2].user_satisfaction = 0.7  # Good rating for behavioral coach
        
        # Update profiles with satisfaction
        for turn in conversation.turns:
            if turn.user_satisfaction:
                conversation._update_user_profile(turn)
        
        if conversation.user_profile:
            print(f"   📈 Agent satisfaction scores:")
            for agent, score in conversation.user_profile.agent_satisfaction_scores.items():
                print(f"      {agent}: {score:.2f}")
        
        # Test 7: Personalized Routing Weights
        print("\n7️⃣ Testing Personalized Routing...")
        routing_weights = conversation.get_personalized_routing_weights()
        if routing_weights:
            print(f"   ⚖️ Personalized routing weights:")
            for agent, weight in routing_weights.items():
                print(f"      {agent}: {weight:.2f}")
        else:
            print(f"   ℹ️ No personalized weights (insufficient data)")
        
        # Test 8: Portfolio Evolution Tracking
        print("\n8️⃣ Testing Portfolio Evolution...")
        portfolio_insights = conversation.get_portfolio_insights()
        print(f"   📊 Portfolio analysis status: {portfolio_insights.get('status', 'unknown')}")
        
        if portfolio_insights.get("status") == "analysis_complete":
            value_change = portfolio_insights.get("value_change", {})
            if value_change:
                print(f"   💰 Value change: {value_change.get('value_change_percent', 0):.1f}%")
        
        # Test 9: Memory Analytics
        print("\n9️⃣ Testing Memory Analytics...")
        analytics = memory_store.get_memory_analytics()
        print(f"   📊 Memory Analytics:")
        print(f"      Total conversations: {analytics['total_conversations']}")
        print(f"      Total turns: {analytics['total_turns']}")
        print(f"      Avg turns per conversation: {analytics['avg_turns_per_conversation']:.1f}")
        print(f"      Embeddings enabled: {analytics['embeddings_enabled']}")
        
        # Test 10: Integration Test
        print("\n🔟 Testing Integration Readiness...")
        
        # Check if integration components are ready
        integration_status = {
            "memory_store": memory_store is not None,
            "conversation_created": conversation is not None,
            "turns_added": len(conversation.turns) > 0,
            "learning_active": conversation.user_profile is not None,
            "search_working": len(search_results) > 0,
            "insights_available": bool(user_insights)
        }
        
        integration_score = sum(integration_status.values()) / len(integration_status) * 100
        
        print(f"   📋 Integration Status:")
        for component, status in integration_status.items():
            status_icon = "✅" if status else "❌"
            print(f"      {status_icon} {component.replace('_', ' ').title()}")
        
        print(f"   🎯 Integration Score: {integration_score:.0f}%")
        
        # Final Summary
        print(f"\n📊 ADVANCED MEMORY TEST SUMMARY")
        print(f"   • Memory System: {'✅ Active' if analytics['embeddings_enabled'] else '⚠️ Fallback Mode'}")
        print(f"   • Conversations: {analytics['total_conversations']}")
        print(f"   • Learning: {'✅ Active' if conversation.user_profile else '❌ Inactive'}")
        print(f"   • Search: {'✅ Working' if search_results else '❌ Not Working'}")
        print(f"   • Integration: {integration_score:.0f}% Ready")
        
        if integration_score >= 80:
            print(f"   🎉 Advanced Memory System is ready for integration!")
            return True
        else:
            print(f"   ⚠️ Advanced Memory System needs attention before integration")
            return False
            
    except ImportError as e:
        print(f"   ❌ Import Error: {e}")
        print(f"   💡 Make sure advanced_memory.py is in your agents/ directory")
        return False
    except Exception as e:
        print(f"   ❌ Test Error: {e}")
        import traceback
        print(f"   📋 Full traceback:")
        print(traceback.format_exc())
        return False

async def test_memory_integration():
    """Test memory integration with existing committee manager"""
    print(f"\n🔗 TESTING MEMORY INTEGRATION")
    print("=" * 40)
    
    try:
        # Test integration with existing committee manager
        print("\n1️⃣ Testing Committee Manager Integration...")
        
        from agents.enhanced_investment_committee_manager import EnhancedInvestmentCommitteeManager
        
        # Initialize manager (should now include advanced memory)
        manager = EnhancedInvestmentCommitteeManager()
        print(f"   ✅ Committee manager initialized")
        print(f"   🧠 Advanced memory enabled: {getattr(manager, 'advanced_memory_enabled', False)}")
        
        # Test 2: Memory-Enhanced Routing
        if hasattr(manager, 'route_query_with_memory'):
            print("\n2️⃣ Testing Memory-Enhanced Routing...")
            
            test_query = "I'm concerned about my portfolio's risk level given recent market volatility"
            portfolio_context = {
                "totalValue": "$750000",
                "riskLevel": "HIGH",
                "holdings": [
                    {"ticker": "TSLA", "value": 200000},
                    {"ticker": "NVDA", "value": 150000}
                ]
            }
            
            result = await manager.route_query_with_memory(
                query=test_query,
                portfolio_context=portfolio_context,
                user_id="test_user_integration",
                conversation_id="integration_test_001",
                enable_collaboration=True
            )
            
            print(f"   ✅ Memory-enhanced routing completed")
            print(f"   🎯 Specialist: {result.get('specialist_used', 'unknown')}")
            print(f"   🧠 Memory enhanced: {result.get('memory_enhanced', False)}")
            print(f"   👤 Personalized routing: {result.get('personalized_routing', False)}")
            
            if result.get('user_insights'):
                print(f"   💡 User insights available: {len(result['user_insights'])} insights")
            
            if result.get('similar_past_discussions'):
                print(f"   🔍 Similar discussions found: {len(result['similar_past_discussions'])}")
                
        else:
            print("\n2️⃣ Memory-Enhanced Routing Not Available")
            print("   ⚠️ Need to add route_query_with_memory method to committee manager")
        
        # Test 3: User Learning Over Time
        print("\n3️⃣ Testing User Learning Over Multiple Interactions...")
        
        if hasattr(manager, 'advanced_memory_enabled') and manager.advanced_memory_enabled:
            # Simulate multiple interactions to test learning
            user_id = "learning_test_user"
            conversation_id = "learning_test_conv"
            
            learning_scenarios = [
                {
                    "query": "What's my portfolio risk?",
                    "satisfaction": 0.9,  # High satisfaction
                    "description": "Simple risk query - high satisfaction"
                },
                {
                    "query": "I need detailed risk analysis with VaR calculations",
                    "satisfaction": 0.8,  # Good satisfaction for detailed analysis
                    "description": "Complex analysis request"
                },
                {
                    "query": "Should I rebalance my tech allocation?",
                    "satisfaction": 0.7,  # Moderate satisfaction
                    "description": "Portfolio management query"
                }
            ]
            
            for i, scenario in enumerate(learning_scenarios):
                print(f"   📝 Scenario {i+1}: {scenario['description']}")
                
                if hasattr(manager, 'route_query_with_memory'):
                    result = await manager.route_query_with_memory(
                        query=scenario["query"],
                        portfolio_context=portfolio_context,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        user_satisfaction=scenario["satisfaction"]
                    )
                    
                    specialist = result.get('specialist_used', 'unknown')
                    print(f"      → Routed to: {specialist}")
                    
                    # Check if learning is happening
                    if result.get('user_insights'):
                        insights = result['user_insights']
                        if 'expertise_level' in insights:
                            print(f"      → Expertise: {insights['expertise_level']['level']}")
                
            print(f"   ✅ Learning simulation completed")
        else:
            print("   ⚠️ Advanced memory not enabled - cannot test learning")
        
        # Test 4: Memory Analytics
        print("\n4️⃣ Testing Memory Analytics...")
        
        if hasattr(manager, 'get_enhanced_analytics_dashboard'):
            analytics = manager.get_enhanced_analytics_dashboard()
            
            memory_system = analytics.get('memory_system', {})
            print(f"   📊 Memory System Status:")
            print(f"      Advanced Memory: {memory_system.get('advanced_memory_enabled', False)}")
            
            if memory_system.get('memory_analytics'):
                mem_analytics = memory_system['memory_analytics']
                print(f"      Total Conversations: {mem_analytics.get('total_conversations', 0)}")
                print(f"      Total Turns: {mem_analytics.get('total_turns', 0)}")
                print(f"      Embeddings: {mem_analytics.get('embeddings_enabled', False)}")
        
        # Integration Score
        integration_features = {
            "manager_initialized": manager is not None,
            "memory_enabled": getattr(manager, 'advanced_memory_enabled', False),
            "memory_routing": hasattr(manager, 'route_query_with_memory'),
            "analytics_updated": hasattr(manager, 'get_enhanced_analytics_dashboard')
        }
        
        score = sum(integration_features.values()) / len(integration_features) * 100
        
        print(f"\n📊 INTEGRATION TEST SUMMARY:")
        for feature, status in integration_features.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {feature.replace('_', ' ').title()}")
        
        print(f"   🎯 Integration Score: {score:.0f}%")
        
        return score >= 75
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

async def main():
    """Main test runner"""
    print("🚀 ADVANCED MEMORY SYSTEM - WEEK 1 TESTS")
    print("=" * 60)
    
    # Run core memory tests
    memory_success = await test_advanced_memory_system()
    
    # Run integration tests
    integration_success = await test_memory_integration()
    
    # Overall results
    print(f"\n" + "=" * 60)
    print(f"📋 FINAL TEST RESULTS")
    print(f"   🧠 Memory System: {'✅ PASS' if memory_success else '❌ FAIL'}")
    print(f"   🔗 Integration: {'✅ PASS' if integration_success else '❌ FAIL'}")
    
    overall_success = memory_success and integration_success
    
    if overall_success:
        print(f"   🎉 WEEK 1 IMPLEMENTATION: READY FOR PRODUCTION")
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. Install sentence-transformers: pip install sentence-transformers")
        print(f"   2. Update your enhanced_investment_committee_manager.py with integration code")
        print(f"   3. Test with real user interactions")
        print(f"   4. Begin Week 2: Proactive Insights Engine")
    else:
        print(f"   ⚠️ WEEK 1 IMPLEMENTATION: NEEDS ATTENTION")
        print(f"\n🔧 REQUIRED FIXES:")
        if not memory_success:
            print(f"   • Fix core memory system issues")
        if not integration_success:
            print(f"   • Complete integration with committee manager")
    
    return overall_success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        exit(1)