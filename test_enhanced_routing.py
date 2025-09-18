# test_enhanced_routing.py - Test script for enhanced routing system
"""
Test script to validate enhanced routing functionality
Run this to test semantic routing vs keyword routing
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, List

# Add your project root to path if needed
# sys.path.append('/path/to/your/project')

from agents.enhanced_investment_committee_manager import EnhancedInvestmentCommitteeManager

class RoutingTester:
    """Test enhanced routing functionality"""
    
    def __init__(self):
        self.manager = EnhancedInvestmentCommitteeManager()
        self.test_portfolio = {
            "total_value": 250000,
            "holdings": [
                {"ticker": "AAPL", "value": 50000},
                {"ticker": "MSFT", "value": 40000},
                {"ticker": "GOOGL", "value": 35000},
                {"ticker": "TSLA", "value": 30000},
                {"ticker": "NVDA", "value": 25000},
                {"ticker": "SPY", "value": 70000}
            ],
            "riskLevel": "MODERATE",
            "user_id": "test_user_123",
            "conversation_id": "test_conversation_001"
        }
        
        self.test_queries = [
            # Quantitative Analyst queries
            {
                "query": "What's the Value at Risk of my portfolio at 95% confidence?",
                "expected_agent": "quantitative_analyst",
                "category": "Risk Analysis"
            },
            {
                "query": "Can you run a Monte Carlo stress test to see how my portfolio performs in market crashes?",
                "expected_agent": "quantitative_analyst", 
                "category": "Stress Testing"
            },
            {
                "query": "I'm worried about correlation risk between my tech stocks",
                "expected_agent": "quantitative_analyst",
                "category": "Correlation Analysis"
            },
            
            # Portfolio Manager queries
            {
                "query": "Should I rebalance my portfolio allocation?",
                "expected_agent": "portfolio_manager",
                "category": "Portfolio Management"
            },
            {
                "query": "I want to optimize my asset allocation for better risk-adjusted returns",
                "expected_agent": "portfolio_manager",
                "category": "Optimization"
            },
            {
                "query": "What specific trades do you recommend to improve my portfolio?",
                "expected_agent": "portfolio_manager",
                "category": "Trade Recommendations"
            },
            
            # CIO queries
            {
                "query": "What's your strategic outlook for the markets this year?",
                "expected_agent": "cio",
                "category": "Market Outlook"
            },
            {
                "query": "How should I allocate between different asset classes for long-term growth?",
                "expected_agent": "cio",
                "category": "Strategic Allocation"
            },
            {
                "query": "What are the key macroeconomic trends I should consider?",
                "expected_agent": "cio",
                "category": "Macro Analysis"
            },
            
            # Behavioral Coach queries
            {
                "query": "I'm feeling anxious about market volatility and want to sell everything",
                "expected_agent": "behavioral_coach",
                "category": "Emotional Guidance"
            },
            {
                "query": "Help me identify any behavioral biases in my investment decisions",
                "expected_agent": "behavioral_coach",
                "category": "Bias Analysis"
            },
            {
                "query": "I keep panic selling during market downturns, what should I do?",
                "expected_agent": "behavioral_coach",
                "category": "Behavioral Intervention"
            },
            
            # Ambiguous queries (test routing intelligence)
            {
                "query": "My portfolio isn't performing well, what should I do?",
                "expected_agent": "portfolio_manager",  # Could be multiple agents
                "category": "General Performance"
            },
            {
                "query": "I'm concerned about my investment strategy",
                "expected_agent": "cio",  # Strategic concern
                "category": "Strategy Concern"
            },
            {
                "query": "How risky is my current setup?",
                "expected_agent": "quantitative_analyst",
                "category": "Risk Assessment"
            }
        ]
    
    async def test_routing_accuracy(self):
        """Test routing accuracy across different query types"""
        print("=" * 80)
        print("ENHANCED ROUTING ACCURACY TEST")
        print("=" * 80)
        
        print(f"\n🧠 System Features:")
        print(f"   • Semantic Routing: {'✅ Enabled' if self.manager.semantic_router else '❌ Using Keyword Fallback'}")
        print(f"   • Advanced Memory: {'✅ Enabled' if hasattr(self.manager, 'enhanced_conversations') else '❌ Using Basic Memory'}")
        print(f"   • Available Specialists: {len(self.manager.specialists)}")
        
        routing_results = []
        
        for i, test_case in enumerate(self.test_queries, 1):
            print(f"\n📋 Test {i}/{len(self.test_queries)}: {test_case['category']}")
            print(f"   Query: \"{test_case['query']}\"")
            print(f"   Expected: {test_case['expected_agent']}")
            
            try:
                # Test the routing without full execution
                conversation = self.manager._get_or_create_enhanced_conversation(
                    self.test_portfolio["conversation_id"], 
                    self.test_portfolio,
                    self.test_portfolio["user_id"]
                )
                
                analysis_type = self.manager._determine_analysis_type(test_case['query'])
                selected_tools = []  # Skip tool selection for routing test
                
                selected_agent, confidence = self.manager._enhanced_route_to_specialist_with_confidence(
                    test_case['query'], conversation, selected_tools
                )
                
                # Check accuracy
                is_correct = selected_agent == test_case['expected_agent']
                accuracy_indicator = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                
                print(f"   Result: {selected_agent} (confidence: {confidence}%) {accuracy_indicator}")
                
                routing_results.append({
                    "test_case": test_case,
                    "selected_agent": selected_agent,
                    "confidence": confidence,
                    "correct": is_correct,
                    "analysis_type": analysis_type
                })
                
            except Exception as e:
                print(f"   ❌ ERROR: {str(e)}")
                routing_results.append({
                    "test_case": test_case,
                    "selected_agent": "ERROR",
                    "confidence": 0,
                    "correct": False,
                    "error": str(e)
                })
        
        # Calculate overall accuracy
        correct_predictions = sum(1 for result in routing_results if result.get("correct", False))
        total_tests = len(routing_results)
        accuracy_rate = (correct_predictions / total_tests) * 100
        
        print(f"\n📊 ROUTING ACCURACY SUMMARY")
        print(f"   • Total Tests: {total_tests}")
        print(f"   • Correct Predictions: {correct_predictions}")
        print(f"   • Accuracy Rate: {accuracy_rate:.1f}%")
        print(f"   • Routing Method: {'Semantic' if self.manager.semantic_router else 'Keyword-based'}")
        
        return routing_results
    
    async def test_full_query_execution(self):
        """Test full query execution with a sample query"""
        print("\n" + "=" * 80)
        print("FULL QUERY EXECUTION TEST")
        print("=" * 80)
        
        test_query = "What's the risk level of my portfolio and should I be concerned?"
        
        print(f"🚀 Executing full analysis for:")
        print(f"   Query: \"{test_query}\"")
        print(f"   Portfolio Value: ${self.test_portfolio['total_value']:,}")
        print(f"   Holdings: {len(self.test_portfolio['holdings'])} assets")
        
        start_time = datetime.now()
        
        try:
            response = await self.manager.route_query(
                query=test_query,
                portfolio_context=self.test_portfolio,
                user_id=self.test_portfolio["user_id"]
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            print(f"\n✅ EXECUTION SUCCESSFUL")
            print(f"   • Agent Used: {response.get('specialist_used', 'Unknown')}")
            print(f"   • Routing Confidence: {response.get('routing_confidence', 0)}%")
            print(f"   • Analysis Confidence: {response.get('analysis_confidence', 0)}%")
            print(f"   • Execution Time: {execution_time:.2f}s")
            print(f"   • Tools Selected: {len(response.get('tools_selected', []))}")
            print(f"   • Backend Integration: {response.get('backend_integration', False)}")
            print(f"   • Enhancement Version: {response.get('enhancement_version', 'Unknown')}")
            
            # Show partial response content
            content = response.get('content', '')
            if content:
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f"\n📄 Response Preview:")
                print(f"   {preview}")
            
            # Show quick actions
            quick_actions = response.get('quick_actions', [])
            if quick_actions:
                print(f"\n⚡ Quick Actions Generated: {len(quick_actions)}")
                for action in quick_actions[:3]:  # Show first 3
                    print(f"   • {action.get('label', 'Unknown Action')}")
            
            return response
            
        except Exception as e:
            print(f"\n❌ EXECUTION FAILED: {str(e)}")
            import traceback
            print(f"   Full Error: {traceback.format_exc()}")
            return None
    
    async def test_conversation_memory(self):
        """Test conversation memory and context tracking"""
        print("\n" + "=" * 80)
        print("CONVERSATION MEMORY TEST")
        print("=" * 80)
        
        # Simulate a conversation sequence
        conversation_queries = [
            "What's the risk level of my portfolio?",
            "Can you also check the correlation between my holdings?", 
            "Based on that analysis, what should I do?"
        ]
        
        print("🧠 Testing conversation memory with sequential queries:")
        
        for i, query in enumerate(conversation_queries, 1):
            print(f"\n💬 Query {i}: \"{query}\"")
            
            try:
                response = await self.manager.route_query(
                    query=query,
                    portfolio_context=self.test_portfolio,
                    user_id=self.test_portfolio["user_id"]
                )
                
                agent_used = response.get('specialist_used', 'Unknown')
                confidence = response.get('routing_confidence', 0)
                
                print(f"   → Routed to: {agent_used} (confidence: {confidence}%)")
                
                # Check for conversation insights
                insights = response.get('enhanced_insights', [])
                if insights:
                    print(f"   → Generated {len(insights)} insights from conversation history")
                
                suggestions = response.get('cross_specialist_suggestions', [])
                if suggestions:
                    print(f"   → Suggested {len(suggestions)} cross-specialist consultations")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
        
        # Test conversation analytics
        try:
            analytics = self.manager.get_enhanced_analytics_dashboard()
            system_overview = analytics.get('system_overview', {})
            
            print(f"\n📈 Conversation Analytics:")
            print(f"   • Total Conversations: {system_overview.get('total_conversations', 0)}")
            print(f"   • Total Interactions: {system_overview.get('total_turns', 0)}")
            print(f"   • Semantic Routing: {system_overview.get('semantic_routing_enabled', False)}")
            print(f"   • Advanced Memory: {system_overview.get('advanced_memory_enabled', False)}")
            
        except Exception as e:
            print(f"   ❌ Analytics Error: {str(e)}")
    
    async def run_all_tests(self):
        """Run all routing tests"""
        print("🚀 STARTING ENHANCED ROUTING TESTS\n")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test 1: Routing Accuracy
        routing_results = await self.test_routing_accuracy()
        
        # Test 2: Full Execution
        execution_result = await self.test_full_query_execution()
        
        # Test 3: Conversation Memory
        await self.test_conversation_memory()
        
        print("\n" + "=" * 80)
        print("🎯 TEST SUITE COMPLETE")
        print("=" * 80)
        
        # Summary
        if routing_results:
            correct_count = sum(1 for r in routing_results if r.get("correct", False))
            total_count = len(routing_results)
            accuracy = (correct_count / total_count) * 100
            
            print(f"\n📊 SUMMARY:")
            print(f"   • Routing Accuracy: {accuracy:.1f}% ({correct_count}/{total_count})")
            print(f"   • Full Execution: {'✅ Success' if execution_result else '❌ Failed'}")
            print(f"   • System Status: {'🟢 Fully Enhanced' if self.manager.semantic_router else '🟡 Keyword Fallback'}")
            
            if accuracy >= 80:
                print(f"   • Overall Grade: 🏆 EXCELLENT")
            elif accuracy >= 60:
                print(f"   • Overall Grade: ✅ GOOD") 
            else:
                print(f"   • Overall Grade: ⚠️ NEEDS IMPROVEMENT")


async def main():
    """Main test function"""
    tester = RoutingTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    # Run the async test
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        print(traceback.format_exc())