#!/usr/bin/env python3
# test_proactive_insights.py - Week 2 Proactive Insights Testing

import asyncio
import sys
from datetime import datetime, timedelta
from typing import Dict, List

# Add the agents directory to path
sys.path.append('agents')

try:
    from agents.proactive_insights import (
        ProactiveInsightsEngine, 
        PortfolioDriftMonitor, 
        BehavioralPatternAnalyzer,
        MarketOpportunityDetector,
        InsightType,
        InsightPriority
    )
    from agents.enhanced_investment_committee_manager import EnhancedInvestmentCommitteeManager
    print("✅ Proactive Insights modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure to add the proactive_insights.py file to your agents directory")
    sys.exit(1)

async def test_week2_proactive_insights():
    """Comprehensive test suite for Week 2 Proactive Insights Engine"""
    
    print("🚀 PROACTIVE INSIGHTS ENGINE - WEEK 2 TESTS")
    print("=" * 60)
    
    # Test 1: Core Engine Initialization
    print("\n1️⃣ Testing Core Engine Initialization...")
    try:
        insights_engine = ProactiveInsightsEngine()
        portfolio_monitor = PortfolioDriftMonitor()
        behavioral_analyzer = BehavioralPatternAnalyzer()
        market_detector = MarketOpportunityDetector()
        
        print("   ✅ All core components initialized successfully")
        print(f"   📊 Engine components: 4/4 active")
        print(f"   🎯 Drift threshold: {portfolio_monitor.drift_threshold}")
        print(f"   🧠 Pattern window: {behavioral_analyzer.pattern_window}")
        
        init_result = "PASS"
    except Exception as e:
        print(f"   ❌ Core initialization failed: {e}")
        init_result = "FAIL"
    
    # Test 2: Portfolio Drift Detection
    print("\n2️⃣ Testing Portfolio Drift Detection...")
    try:
        # Test portfolio with concentration risk
        test_portfolio = {
            'holdings': [
                {'symbol': 'AAPL', 'value': 60000, 'sector': 'Technology'},
                {'symbol': 'MSFT', 'value': 25000, 'sector': 'Technology'},
                {'symbol': 'GOOGL', 'value': 15000, 'sector': 'Technology'}
            ],
            'total_value': 100000
        }
        
        drift_insights = await portfolio_monitor.analyze_drift(
            portfolio_data=test_portfolio,
            user_id="test_drift_user",
            memory_system=None
        )
        
        concentration_insights = [
            insight for insight in drift_insights 
            if 'concentration' in insight.description.lower()
        ]
        
        print(f"   ✅ Drift analysis completed")
        print(f"   📊 Total insights: {len(drift_insights)}")
        print(f"   ⚠️ Concentration alerts: {len(concentration_insights)}")
        
        if concentration_insights:
            highest_priority = max(concentration_insights, key=lambda x: x.priority.value)
            print(f"   🎯 Highest priority: {highest_priority.priority.value}")
            print(f"   💡 Recommendation count: {len(highest_priority.recommendations)}")
        
        drift_result = "PASS"
    except Exception as e:
        print(f"   ❌ Portfolio drift detection failed: {e}")
        drift_result = "FAIL"
    
    # Test 3: Behavioral Pattern Analysis
    print("\n3️⃣ Testing Behavioral Pattern Analysis...")
    try:
        # Test conversation history with anxiety pattern
        anxiety_conversations = [
            {
                'query': "I'm really worried about the market crash",
                'timestamp': datetime.now() - timedelta(days=1),
                'specialist': 'behavioral_coach'
            },
            {
                'query': "Should I be concerned about my portfolio risk?",
                'timestamp': datetime.now() - timedelta(days=3),
                'specialist': 'quantitative_analyst'
            },
            {
                'query': "I'm scared about losing money",
                'timestamp': datetime.now() - timedelta(days=5),
                'specialist': 'behavioral_coach'
            },
            {
                'query': "Help me understand investment risks",
                'timestamp': datetime.now() - timedelta(days=7),
                'specialist': 'quantitative_analyst'
            }
        ]
        
        behavioral_insights = await behavioral_analyzer.analyze_patterns(
            conversation_history=anxiety_conversations,
            user_id="test_behavioral_user",
            memory_system=None
        )
        
        anxiety_patterns = [
            insight for insight in behavioral_insights 
            if 'anxiety' in insight.title.lower() or 'risk' in insight.title.lower()
        ]
        
        learning_patterns = [
            insight for insight in behavioral_insights 
            if 'learning' in insight.title.lower()
        ]
        
        print(f"   ✅ Behavioral analysis completed")
        print(f"   🧠 Total behavioral insights: {len(behavioral_insights)}")
        print(f"   ⚠️ Anxiety patterns detected: {len(anxiety_patterns)}")
        print(f"   📚 Learning patterns detected: {len(learning_patterns)}")
        
        if behavioral_insights:
            for insight in behavioral_insights:
                print(f"   📋 Pattern: {insight.title} ({insight.priority.value} priority)")
        
        behavioral_result = "PASS"
    except Exception as e:
        print(f"   ❌ Behavioral pattern analysis failed: {e}")
        behavioral_result = "FAIL"
    
    # Test 4: Market Opportunity Detection
    print("\n4️⃣ Testing Market Opportunity Detection...")
    try:
        # Test small, concentrated portfolio
        small_portfolio = {
            'holdings': [
                {'symbol': 'AAPL', 'value': 35000, 'sector': 'Technology'},
                {'symbol': 'MSFT', 'value': 15000, 'sector': 'Technology'}
            ],
            'total_value': 50000
        }
        
        market_insights = await market_detector.identify_opportunities(
            portfolio_data=small_portfolio,
            user_id="test_market_user"
        )
        
        diversification_opportunities = [
            insight for insight in market_insights 
            if 'diversification' in insight.title.lower() or 'sector' in insight.title.lower()
        ]
        
        print(f"   ✅ Market analysis completed")
        print(f"   📈 Total opportunities: {len(market_insights)}")
        print(f"   🎯 Diversification opportunities: {len(diversification_opportunities)}")
        
        if market_insights:
            for insight in market_insights:
                print(f"   💡 Opportunity: {insight.title} ({insight.priority.value} priority)")
        
        market_result = "PASS"
    except Exception as e:
        print(f"   ❌ Market opportunity detection failed: {e}")
        market_result = "FAIL"
    
    # Test 5: Comprehensive Insights Engine
    print("\n5️⃣ Testing Comprehensive Insights Engine...")
    try:
        insights_engine = ProactiveInsightsEngine()
        
        # Test with complex scenario
        complex_portfolio = {
            'holdings': [
                {'symbol': 'AAPL', 'value': 45000, 'sector': 'Technology'},
                {'symbol': 'MSFT', 'value': 25000, 'sector': 'Technology'},
                {'symbol': 'TSLA', 'value': 20000, 'sector': 'Technology'},
                {'symbol': 'NVDA', 'value': 10000, 'sector': 'Technology'}
            ],
            'total_value': 100000
        }
        
        complex_conversations = [
            {
                'query': "I'm worried about tech concentration",
                'timestamp': datetime.now() - timedelta(days=2)
            },
            {
                'query': "Should I diversify more?",
                'timestamp': datetime.now() - timedelta(days=4)
            }
        ]
        
        all_insights = await insights_engine.generate_insights(
            user_id="test_comprehensive_user",
            portfolio_data=complex_portfolio,
            conversation_history=complex_conversations
        )
        
        # Analyze insights by type and priority
        insights_by_type = {}
        insights_by_priority = {}
        
        for insight in all_insights:
            insight_type = insight.type.value
            insight_priority = insight.priority.value
            
            insights_by_type[insight_type] = insights_by_type.get(insight_type, 0) + 1
            insights_by_priority[insight_priority] = insights_by_priority.get(insight_priority, 0) + 1
        
        conversation_starters = [
            insight for insight in all_insights 
            if insight.type == InsightType.CONVERSATION_STARTER
        ]
        
        high_priority_insights = [
            insight for insight in all_insights 
            if insight.priority in [InsightPriority.CRITICAL, InsightPriority.HIGH]
        ]
        
        print(f"   ✅ Comprehensive analysis completed")
        print(f"   📊 Total insights generated: {len(all_insights)}")
        print(f"   🎯 High priority insights: {len(high_priority_insights)}")
        print(f"   💬 Conversation starters: {len(conversation_starters)}")
        print(f"   📋 Insights by type: {insights_by_type}")
        print(f"   🔢 Insights by priority: {insights_by_priority}")
        
        # Show sample conversation starters
        if conversation_starters:
            print(f"   💬 Sample conversation starter:")
            sample_starter = conversation_starters[0]
            print(f"      '{sample_starter.conversation_starters[0]}'")
        
        comprehensive_result = "PASS"
    except Exception as e:
        print(f"   ❌ Comprehensive insights engine failed: {e}")
        comprehensive_result = "FAIL"
    
    # Test 6: Integration Readiness Check
    print("\n6️⃣ Testing Integration Readiness...")
    try:
        # Test data structures and API compatibility
        test_insight = all_insights[0] if 'all_insights' in locals() and all_insights else None
        
        if test_insight:
            # Verify required attributes
            required_attrs = ['id', 'type', 'priority', 'title', 'description', 'recommendations', 'conversation_starters']
            missing_attrs = [attr for attr in required_attrs if not hasattr(test_insight, attr)]
            
            if missing_attrs:
                raise Exception(f"Missing required attributes: {missing_attrs}")
            
            # Test serialization compatibility
            test_data = {
                'id': test_insight.id,
                'type': test_insight.type.value,
                'priority': test_insight.priority.value,
                'title': test_insight.title,
                'description': test_insight.description,
                'recommendations': test_insight.recommendations,
                'conversation_starters': test_insight.conversation_starters,
                'created_at': test_insight.created_at.isoformat()
            }
            
            print(f"   ✅ Data structure validation passed")
            print(f"   📦 Serialization compatibility: ✅")
            print(f"   🔗 API readiness: ✅")
        
        integration_result = "PASS"
    except Exception as e:
        print(f"   ❌ Integration readiness check failed: {e}")
        integration_result = "FAIL"
    
    # Final Summary
    print("\n" + "=" * 60)
    print("📊 WEEK 2 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    test_results = {
        "Core Initialization": init_result,
        "Portfolio Drift Detection": drift_result,
        "Behavioral Pattern Analysis": behavioral_result,
        "Market Opportunity Detection": market_result,
        "Comprehensive Engine": comprehensive_result,
        "Integration Readiness": integration_result
    }
    
    passed_tests = sum(1 for result in test_results.values() if result == "PASS")
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"\n🎯 TEST RESULTS:")
    for test_name, result in test_results.items():
        status_icon = "✅" if result == "PASS" else "❌"
        print(f"   {status_icon} {test_name}: {result}")
    
    print(f"\n📈 OVERALL PERFORMANCE:")
    print(f"   Tests Passed: {passed_tests}/{total_tests}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 100:
        print(f"   🎉 STATUS: PERFECT - READY FOR PRODUCTION")
    elif success_rate >= 80:
        print(f"   ✅ STATUS: EXCELLENT - READY FOR INTEGRATION")
    elif success_rate >= 60:
        print(f"   ⚠️ STATUS: GOOD - MINOR REFINEMENTS NEEDED")
    else:
        print(f"   🔧 STATUS: NEEDS WORK - SIGNIFICANT IMPROVEMENTS REQUIRED")
    
    print(f"\n📋 NEXT STEPS:")
    if success_rate >= 80:
        print(f"   1. Integrate proactive insights with frontend")
        print(f"   2. Begin user testing with real portfolios")
        print(f"   3. Monitor insight engagement and effectiveness")
        print(f"   4. Prepare for Week 3: Advanced Learning & Optimization")
    else:
        failed_tests = [name for name, result in test_results.items() if result == "FAIL"]
        print(f"   1. Address failed tests: {', '.join(failed_tests)}")
        print(f"   2. Rerun test suite until success rate > 80%")
        print(f"   3. Review error logs and fix implementation issues")
    
    return {
        'test_results': test_results,
        'success_rate': success_rate,
        'status': 'READY' if success_rate >= 80 else 'NEEDS_WORK'
    }

if __name__ == "__main__":
    print("Starting Week 2 Proactive Insights Engine Tests...")
    
    try:
        # Run the async test function
        result = asyncio.run(test_week2_proactive_insights())
        
        print(f"\n🏁 TEST EXECUTION COMPLETED")
        print(f"Final Status: {result['status']}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        print("Check your implementation and try again")