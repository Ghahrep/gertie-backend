# agents/proactive_insights.py - Week 2 Proactive Insights Engine
"""
Proactive Insights Engine for Investment Committee
=================================================

Week 2 Implementation: Builds on advanced memory system to provide:
- Portfolio drift detection and alerts
- Behavioral pattern analysis and interventions
- Market opportunity identification
- Proactive conversation starters and recommendations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
import json

logger = logging.getLogger(__name__)

class InsightPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class InsightType(Enum):
    PORTFOLIO_DRIFT = "portfolio_drift"
    BEHAVIORAL_PATTERN = "behavioral_pattern"
    MARKET_OPPORTUNITY = "market_opportunity"
    RISK_ALERT = "risk_alert"
    PERFORMANCE_ANOMALY = "performance_anomaly"
    CONVERSATION_STARTER = "conversation_starter"

@dataclass
class ProactiveInsight:
    """Structured insight with actionable recommendations"""
    id: str
    type: InsightType
    priority: InsightPriority
    title: str
    description: str
    data: Dict
    recommendations: List[str]
    conversation_starters: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    user_id: Optional[str] = None
    
class ProactiveInsightsEngine:
    """Main engine for generating proactive portfolio and behavioral insights"""
    
    def __init__(self, memory_system=None):
        self.memory_system = memory_system
        self.portfolio_monitor = PortfolioDriftMonitor()
        self.behavioral_analyzer = BehavioralPatternAnalyzer()
        self.market_detector = MarketOpportunityDetector()
        self.insight_cache = {}
        self.user_preferences = {}
        
    async def generate_insights(self, user_id: str, portfolio_data: Dict, 
                              conversation_history: List[Dict] = None) -> List[ProactiveInsight]:
        """Generate comprehensive proactive insights for user"""
        insights = []
        
        try:
            # Portfolio drift analysis
            drift_insights = await self.portfolio_monitor.analyze_drift(
                portfolio_data, user_id, self.memory_system
            )
            insights.extend(drift_insights)
            
            # Behavioral pattern analysis
            if conversation_history:
                behavioral_insights = await self.behavioral_analyzer.analyze_patterns(
                    conversation_history, user_id, self.memory_system
                )
                insights.extend(behavioral_insights)
            
            # Market opportunity detection
            market_insights = await self.market_detector.identify_opportunities(
                portfolio_data, user_id
            )
            insights.extend(market_insights)
            
            # Generate conversation starters
            conversation_insights = self._generate_conversation_starters(
                insights, portfolio_data, user_id
            )
            insights.extend(conversation_insights)
            
            # Priority and relevance filtering
            insights = self._filter_and_prioritize(insights, user_id)
            
            # Cache insights for performance
            self.insight_cache[user_id] = {
                'insights': insights,
                'generated_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(hours=6)
            }
            
            logger.info(f"Generated {len(insights)} proactive insights for user {user_id}")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights for user {user_id}: {e}")
            return []
    
    def _generate_conversation_starters(self, insights: List[ProactiveInsight], 
                                       portfolio_data: Dict, user_id: str) -> List[ProactiveInsight]:
        """Generate intelligent conversation starters based on insights"""
        starters = []
        
        # Portfolio-based starters
        total_value = portfolio_data.get('total_value', 0)
        holdings_count = len(portfolio_data.get('holdings', []))
        
        if total_value > 100000 and holdings_count < 5:
            starters.append(ProactiveInsight(
                id=f"starter_diversification_{user_id}",
                type=InsightType.CONVERSATION_STARTER,
                priority=InsightPriority.MEDIUM,
                title="Portfolio Diversification Discussion",
                description="Your portfolio might benefit from broader diversification",
                data={"portfolio_value": total_value, "holdings_count": holdings_count},
                recommendations=[
                    "Consider adding international exposure",
                    "Explore different asset classes",
                    "Review sector concentration"
                ],
                conversation_starters=[
                    "I notice your portfolio has strong positions but limited diversification. Would you like to explore expansion opportunities?",
                    "Your portfolio value suggests you might benefit from broader asset class exposure. Shall we discuss diversification strategies?",
                    "Given your portfolio size, we could explore some advanced diversification techniques. Interested in learning more?"
                ],
                created_at=datetime.now()
            ))
        
        # Insight-based starters
        high_priority_insights = [i for i in insights if i.priority in [InsightPriority.CRITICAL, InsightPriority.HIGH]]
        
        if high_priority_insights:
            insight_starter = ProactiveInsight(
                id=f"starter_priority_{user_id}",
                type=InsightType.CONVERSATION_STARTER,
                priority=InsightPriority.HIGH,
                title="Important Portfolio Updates",
                description="Several high-priority insights require your attention",
                data={"insight_count": len(high_priority_insights)},
                recommendations=[
                    "Review high-priority portfolio alerts",
                    "Consider immediate action on critical insights",
                    "Schedule portfolio review session"
                ],
                conversation_starters=[
                    f"I've identified {len(high_priority_insights)} important updates about your portfolio. Would you like to review them?",
                    "There are some significant developments in your portfolio that we should discuss. Ready to dive in?",
                    "Your portfolio analysis has flagged several items that could benefit from your attention. Shall we take a look?"
                ],
                created_at=datetime.now()
            )
            starters.append(insight_starter)
        
        return starters
    
    def _filter_and_prioritize(self, insights: List[ProactiveInsight], user_id: str) -> List[ProactiveInsight]:
        """Filter and prioritize insights based on user preferences and relevance"""
        # Remove expired insights
        current_time = datetime.now()
        valid_insights = [
            i for i in insights 
            if i.expires_at is None or i.expires_at > current_time
        ]
        
        # Sort by priority and creation time
        priority_order = {
            InsightPriority.CRITICAL: 5,
            InsightPriority.HIGH: 4,
            InsightPriority.MEDIUM: 3,
            InsightPriority.LOW: 2,
            InsightPriority.INFO: 1
        }
        
        valid_insights.sort(
            key=lambda x: (priority_order[x.priority], x.created_at),
            reverse=True
        )
        
        # Limit to top insights to avoid overwhelming user
        return valid_insights[:10]

class PortfolioDriftMonitor:
    """Monitor portfolio allocation drift and identify rebalancing opportunities"""
    
    def __init__(self):
        self.drift_threshold = 0.05  # 5% drift threshold
        self.critical_drift_threshold = 0.15  # 15% critical threshold
        
    async def analyze_drift(self, portfolio_data: Dict, user_id: str, 
                           memory_system=None) -> List[ProactiveInsight]:
        """Analyze portfolio drift and generate alerts"""
        insights = []
        
        try:
            holdings = portfolio_data.get('holdings', [])
            total_value = portfolio_data.get('total_value', 0)
            
            if not holdings or total_value <= 0:
                return insights
            
            # Calculate current weights
            current_weights = {}
            for holding in holdings:
                symbol = holding.get('symbol', 'UNKNOWN')
                value = holding.get('value', 0)
                weight = value / total_value
                current_weights[symbol] = weight
            
            # Get historical allocation from memory if available
            target_weights = await self._get_target_allocation(user_id, memory_system)
            
            if target_weights:
                drift_analysis = self._calculate_drift(current_weights, target_weights)
                insights.extend(self._generate_drift_insights(drift_analysis, user_id))
            
            # Concentration risk analysis
            concentration_insights = self._analyze_concentration_risk(current_weights, user_id)
            insights.extend(concentration_insights)
            
            # Performance attribution drift
            performance_insights = await self._analyze_performance_drift(
                portfolio_data, user_id, memory_system
            )
            insights.extend(performance_insights)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error in drift analysis for user {user_id}: {e}")
            return []
    
    async def _get_target_allocation(self, user_id: str, memory_system) -> Dict[str, float]:
        """Retrieve target allocation from memory system or use default"""
        if memory_system:
            try:
                # Search for previous optimization or allocation discussions
                allocation_context = await memory_system.search_conversations(
                    user_id, "target allocation optimal weights", limit=3
                )
                
                if allocation_context:
                    # Extract allocation data from context
                    # This would need to be implemented based on your memory system
                    return self._extract_allocation_from_context(allocation_context)
            except Exception as e:
                logger.warning(f"Could not retrieve target allocation from memory: {e}")
        
        # Default equal-weight allocation as fallback
        return {}
    
    def _calculate_drift(self, current_weights: Dict, target_weights: Dict) -> Dict:
        """Calculate drift metrics between current and target allocations"""
        drift_data = {
            'total_drift': 0,
            'position_drifts': {},
            'max_drift': 0,
            'max_drift_position': None,
            'positions_over_threshold': []
        }
        
        all_positions = set(current_weights.keys()) | set(target_weights.keys())
        
        for position in all_positions:
            current = current_weights.get(position, 0)
            target = target_weights.get(position, 0)
            drift = abs(current - target)
            
            drift_data['position_drifts'][position] = {
                'current_weight': current,
                'target_weight': target,
                'drift': drift,
                'drift_percentage': drift / target if target > 0 else float('inf')
            }
            
            drift_data['total_drift'] += drift
            
            if drift > drift_data['max_drift']:
                drift_data['max_drift'] = drift
                drift_data['max_drift_position'] = position
            
            if drift > self.drift_threshold:
                drift_data['positions_over_threshold'].append({
                    'position': position,
                    'drift': drift,
                    'current': current,
                    'target': target
                })
        
        return drift_data
    
    def _generate_drift_insights(self, drift_analysis: Dict, user_id: str) -> List[ProactiveInsight]:
        """Generate insights based on drift analysis"""
        insights = []
        
        total_drift = drift_analysis['total_drift']
        positions_over_threshold = drift_analysis['positions_over_threshold']
        
        if total_drift > self.critical_drift_threshold:
            priority = InsightPriority.CRITICAL
            title = "Critical Portfolio Drift Detected"
            description = f"Portfolio has drifted {total_drift:.1%} from target allocation"
        elif total_drift > self.drift_threshold:
            priority = InsightPriority.HIGH
            title = "Portfolio Rebalancing Recommended"
            description = f"Portfolio has drifted {total_drift:.1%} from target allocation"
        else:
            return insights  # No significant drift
        
        recommendations = [
            f"Rebalance {len(positions_over_threshold)} positions that have exceeded drift thresholds",
            "Review and update target allocation if investment thesis has changed",
            "Consider systematic rebalancing schedule to prevent future drift"
        ]
        
        conversation_starters = [
            f"Your portfolio has drifted {total_drift:.1%} from your target allocation. Should we discuss rebalancing?",
            f"I notice {len(positions_over_threshold)} positions have exceeded their target weights. Would you like to review rebalancing options?",
            "Your portfolio allocation has shifted significantly. Shall we explore bringing it back to your target allocation?"
        ]
        
        insights.append(ProactiveInsight(
            id=f"drift_alert_{user_id}_{datetime.now().strftime('%Y%m%d')}",
            type=InsightType.PORTFOLIO_DRIFT,
            priority=priority,
            title=title,
            description=description,
            data=drift_analysis,
            recommendations=recommendations,
            conversation_starters=conversation_starters,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=7),
            user_id=user_id
        ))
        
        return insights
    
    def _analyze_concentration_risk(self, current_weights: Dict, user_id: str) -> List[ProactiveInsight]:
        """Analyze portfolio concentration risk"""
        insights = []
        
        if not current_weights:
            return insights
        
        # Calculate Herfindahl index for concentration
        herfindahl_index = sum(weight ** 2 for weight in current_weights.values())
        max_position = max(current_weights.values()) if current_weights else 0
        
        # High concentration thresholds
        high_concentration_threshold = 0.3  # 30% in single position
        critical_concentration_threshold = 0.5  # 50% in single position
        
        if max_position > critical_concentration_threshold:
            priority = InsightPriority.CRITICAL
            title = "Critical Portfolio Concentration Risk"
            description = f"Largest position represents {max_position:.1%} of portfolio"
        elif max_position > high_concentration_threshold:
            priority = InsightPriority.HIGH
            title = "High Portfolio Concentration Detected"
            description = f"Largest position represents {max_position:.1%} of portfolio"
        elif herfindahl_index > 0.25:  # Moderately concentrated
            priority = InsightPriority.MEDIUM
            title = "Portfolio Concentration Analysis"
            description = f"Portfolio concentration index: {herfindahl_index:.3f}"
        else:
            return insights  # Well diversified
        
        # Find the largest positions
        largest_positions = sorted(
            current_weights.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        recommendations = [
            f"Consider reducing exposure to {largest_positions[0][0]} ({largest_positions[0][1]:.1%})",
            "Diversify into additional positions to reduce concentration risk",
            "Implement position sizing limits to prevent future concentration"
        ]
        
        conversation_starters = [
            f"Your portfolio has significant concentration in {largest_positions[0][0]}. Would you like to discuss diversification strategies?",
            f"I notice {largest_positions[0][0]} makes up {largest_positions[0][1]:.1%} of your portfolio. Should we explore risk management options?",
            "Your portfolio concentration has increased. Shall we review some diversification approaches?"
        ]
        
        insights.append(ProactiveInsight(
            id=f"concentration_alert_{user_id}_{datetime.now().strftime('%Y%m%d')}",
            type=InsightType.RISK_ALERT,
            priority=priority,
            title=title,
            description=description,
            data={
                'herfindahl_index': herfindahl_index,
                'max_position_weight': max_position,
                'largest_positions': largest_positions,
                'concentration_level': 'critical' if max_position > critical_concentration_threshold 
                                     else 'high' if max_position > high_concentration_threshold 
                                     else 'moderate'
            },
            recommendations=recommendations,
            conversation_starters=conversation_starters,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=30),
            user_id=user_id
        ))
        
        return insights
    
    async def _analyze_performance_drift(self, portfolio_data: Dict, user_id: str, 
                                       memory_system) -> List[ProactiveInsight]:
        """Analyze performance attribution drift"""
        insights = []
        
        # This would analyze performance attribution changes over time
        # Implementation would depend on available performance data
        # For now, return empty list
        
        return insights
    
    def _extract_allocation_from_context(self, allocation_context: List[Dict]) -> Dict[str, float]:
        """Extract target allocation from conversation context"""
        # This would parse conversation context to extract allocation data
        # Implementation depends on your memory system format
        # For now, return empty dict
        return {}

class BehavioralPatternAnalyzer:
    """Analyze user behavioral patterns and provide interventions"""
    
    def __init__(self):
        self.pattern_window = timedelta(days=30)  # Analyze last 30 days
        self.intervention_threshold = 3  # Require 3+ instances for pattern
        
    async def analyze_patterns(self, conversation_history: List[Dict], user_id: str, 
                             memory_system=None) -> List[ProactiveInsight]:
        """Analyze behavioral patterns and generate intervention insights"""
        insights = []
        
        try:
            # Analyze conversation patterns
            pattern_analysis = self._analyze_conversation_patterns(conversation_history)
            
            # Detect bias patterns
            bias_patterns = self._detect_bias_patterns(conversation_history)
            
            # Generate behavioral insights
            if pattern_analysis['risk_anxiety_pattern']:
                insights.extend(self._generate_anxiety_intervention(pattern_analysis, user_id))
            
            if pattern_analysis['decision_paralysis_pattern']:
                insights.extend(self._generate_decision_support(pattern_analysis, user_id))
            
            if bias_patterns:
                insights.extend(self._generate_bias_interventions(bias_patterns, user_id))
            
            # Positive pattern reinforcement
            if pattern_analysis['learning_engagement_pattern']:
                insights.extend(self._generate_positive_reinforcement(pattern_analysis, user_id))
            
            return insights
            
        except Exception as e:
            logger.error(f"Error in behavioral pattern analysis for user {user_id}: {e}")
            return []
    
    def _analyze_conversation_patterns(self, conversation_history: List[Dict]) -> Dict:
        """Analyze conversation patterns for behavioral insights"""
        patterns = {
            'risk_anxiety_pattern': False,
            'decision_paralysis_pattern': False,
            'learning_engagement_pattern': False,
            'frequency_analysis': {},
            'topic_analysis': {},
            'sentiment_trends': []
        }
        
        # Analyze recent conversations
        recent_conversations = [
            conv for conv in conversation_history 
            if self._is_recent(conv.get('timestamp', datetime.now()))
        ]
        
        # Risk anxiety pattern detection
        risk_keywords = ['worried', 'concerned', 'scared', 'anxiety', 'fear', 'risk', 'loss']
        risk_mentions = sum(
            1 for conv in recent_conversations 
            if any(keyword in conv.get('query', '').lower() for keyword in risk_keywords)
        )
        
        patterns['risk_anxiety_pattern'] = risk_mentions >= self.intervention_threshold
        
        # Decision paralysis pattern
        decision_keywords = ['should i', 'what if', 'uncertain', 'confused', 'help me decide']
        decision_mentions = sum(
            1 for conv in recent_conversations 
            if any(keyword in conv.get('query', '').lower() for keyword in decision_keywords)
        )
        
        patterns['decision_paralysis_pattern'] = decision_mentions >= self.intervention_threshold
        
        # Learning engagement pattern
        learning_keywords = ['explain', 'understand', 'learn', 'how does', 'why']
        learning_mentions = sum(
            1 for conv in recent_conversations 
            if any(keyword in conv.get('query', '').lower() for keyword in learning_keywords)
        )
        
        patterns['learning_engagement_pattern'] = learning_mentions >= self.intervention_threshold
        
        return patterns
    
    def _detect_bias_patterns(self, conversation_history: List[Dict]) -> List[Dict]:
        """Detect recurring cognitive bias patterns"""
        bias_patterns = []
        
        # This would implement bias detection based on conversation analysis
        # For now, return basic pattern detection
        
        return bias_patterns
    
    def _generate_anxiety_intervention(self, pattern_analysis: Dict, user_id: str) -> List[ProactiveInsight]:
        """Generate intervention for risk anxiety patterns"""
        insights = []
        
        insight = ProactiveInsight(
            id=f"anxiety_intervention_{user_id}_{datetime.now().strftime('%Y%m%d')}",
            type=InsightType.BEHAVIORAL_PATTERN,
            priority=InsightPriority.HIGH,
            title="Risk Anxiety Pattern Detected",
            description="You've expressed concern about portfolio risk multiple times recently",
            data=pattern_analysis,
            recommendations=[
                "Review your risk tolerance and investment objectives",
                "Consider systematic risk assessment to quantify actual risk levels",
                "Implement gradual exposure strategies to build confidence",
                "Schedule regular portfolio reviews to reduce uncertainty"
            ],
            conversation_starters=[
                "I've noticed you've been concerned about portfolio risk lately. Would you like to do a comprehensive risk assessment to put things in perspective?",
                "You've asked about risk several times recently. Shall we work on a systematic approach to managing investment anxiety?",
                "It seems market volatility has been on your mind. Would you like to explore some strategies to feel more confident about your investments?"
            ],
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=14),
            user_id=user_id
        )
        
        insights.append(insight)
        return insights
    
    def _generate_decision_support(self, pattern_analysis: Dict, user_id: str) -> List[ProactiveInsight]:
        """Generate decision support for paralysis patterns"""
        insights = []
        
        insight = ProactiveInsight(
            id=f"decision_support_{user_id}_{datetime.now().strftime('%Y%m%d')}",
            type=InsightType.BEHAVIORAL_PATTERN,
            priority=InsightPriority.MEDIUM,
            title="Decision Support Recommended",
            description="Pattern of decision uncertainty detected in recent conversations",
            data=pattern_analysis,
            recommendations=[
                "Develop systematic decision-making framework",
                "Set clear investment criteria and thresholds",
                "Use scenario analysis to evaluate options",
                "Implement time-boxed decision processes"
            ],
            conversation_starters=[
                "I've noticed you've been facing several investment decisions lately. Would you like to develop a systematic approach to make these easier?",
                "You've asked for help with multiple decisions recently. Shall we create a decision-making framework to streamline this process?",
                "It seems you're working through some investment choices. Would you like tools to make these decisions more straightforward?"
            ],
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=21),
            user_id=user_id
        )
        
        insights.append(insight)
        return insights
    
    def _generate_bias_interventions(self, bias_patterns: List[Dict], user_id: str) -> List[ProactiveInsight]:
        """Generate interventions for detected bias patterns"""
        insights = []
        
        # Implementation would depend on specific bias patterns detected
        # For now, return empty list
        
        return insights
    
    def _generate_positive_reinforcement(self, pattern_analysis: Dict, user_id: str) -> List[ProactiveInsight]:
        """Generate positive reinforcement for good learning patterns"""
        insights = []
        
        insight = ProactiveInsight(
            id=f"positive_reinforcement_{user_id}_{datetime.now().strftime('%Y%m%d')}",
            type=InsightType.CONVERSATION_STARTER,
            priority=InsightPriority.INFO,
            title="Strong Learning Engagement Detected",
            description="You've been actively learning about investment concepts",
            data=pattern_analysis,
            recommendations=[
                "Continue exploring advanced investment topics",
                "Consider expanding into new areas of interest",
                "Apply learned concepts to portfolio optimization"
            ],
            conversation_starters=[
                "I notice you've been asking great questions about investing lately. Ready to dive into some more advanced topics?",
                "Your engagement with investment concepts has been impressive. Would you like to explore applying these ideas to your portfolio?",
                "You've shown strong curiosity about investment strategies. Shall we discuss some sophisticated approaches you might find interesting?"
            ],
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=7),
            user_id=user_id
        )
        
        insights.append(insight)
        return insights
    
    def _is_recent(self, timestamp: datetime) -> bool:
        """Check if timestamp is within analysis window"""
        return datetime.now() - timestamp <= self.pattern_window

class MarketOpportunityDetector:
    """Detect market opportunities relevant to user's portfolio"""
    
    def __init__(self):
        self.opportunity_types = [
            'sector_rotation',
            'value_opportunities',
            'momentum_signals',
            'risk_arbitrage',
            'rebalancing_timing'
        ]
        
    async def identify_opportunities(self, portfolio_data: Dict, user_id: str) -> List[ProactiveInsight]:
        """Identify market opportunities relevant to portfolio"""
        insights = []
        
        try:
            # Portfolio-specific opportunity analysis
            holdings = portfolio_data.get('holdings', [])
            
            if holdings:
                # Sector analysis
                sector_insights = self._analyze_sector_opportunities(holdings, user_id)
                insights.extend(sector_insights)
                
                # Correlation opportunities
                correlation_insights = self._analyze_correlation_opportunities(holdings, user_id)
                insights.extend(correlation_insights)
                
                # Rebalancing timing
                timing_insights = self._analyze_rebalancing_timing(portfolio_data, user_id)
                insights.extend(timing_insights)
            
            # General market opportunities
            general_insights = self._identify_general_opportunities(user_id)
            insights.extend(general_insights)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error identifying market opportunities for user {user_id}: {e}")
            return []
    
    def _analyze_sector_opportunities(self, holdings: List[Dict], user_id: str) -> List[ProactiveInsight]:
        """Analyze sector-specific opportunities"""
        insights = []
        
        # Simple sector analysis based on holdings
        sectors = {}
        for holding in holdings:
            # This would need real sector classification
            sector = holding.get('sector', 'Unknown')
            sectors[sector] = sectors.get(sector, 0) + 1
        
        # Check for sector concentration or gaps
        if len(sectors) <= 2 and len(holdings) > 3:
            insight = ProactiveInsight(
                id=f"sector_diversification_{user_id}_{datetime.now().strftime('%Y%m%d')}",
                type=InsightType.MARKET_OPPORTUNITY,
                priority=InsightPriority.MEDIUM,
                title="Sector Diversification Opportunity",
                description="Portfolio shows limited sector exposure",
                data={'current_sectors': sectors, 'holdings_count': len(holdings)},
                recommendations=[
                    "Consider adding exposure to underrepresented sectors",
                    "Explore sector ETFs for broader diversification",
                    "Review correlation benefits of cross-sector holdings"
                ],
                conversation_starters=[
                    "Your portfolio is concentrated in few sectors. Would you like to explore diversification opportunities?",
                    "I see potential to enhance your portfolio through sector diversification. Interested in learning more?",
                    "There are some interesting sector opportunities that might complement your current holdings. Shall we discuss?"
                ],
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=14),
                user_id=user_id
            )
            insights.append(insight)
        
        return insights
    
    def _analyze_correlation_opportunities(self, holdings: List[Dict], user_id: str) -> List[ProactiveInsight]:
        """Analyze correlation and diversification opportunities"""
        insights = []
        
        # This would implement correlation analysis
        # For now, return basic diversification insight if portfolio is small
        
        if len(holdings) < 5:
            insight = ProactiveInsight(
                id=f"diversification_opportunity_{user_id}_{datetime.now().strftime('%Y%m%d')}",
                type=InsightType.MARKET_OPPORTUNITY,
                priority=InsightPriority.MEDIUM,
                title="Portfolio Diversification Opportunity",
                description="Additional positions could improve risk-adjusted returns",
                data={'current_holdings': len(holdings)},
                recommendations=[
                    "Consider adding 2-3 additional positions for better diversification",
                    "Explore low-correlation assets to reduce portfolio volatility",
                    "Review international exposure for geographic diversification"
                ],
                conversation_starters=[
                    f"With {len(holdings)} positions, there's room to enhance diversification. Would you like to explore some options?",
                    "Your portfolio could benefit from additional diversification. Shall we look at some complementary investments?",
                    "I've identified some diversification opportunities that could improve your risk-return profile. Interested?"
                ],
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=21),
                user_id=user_id
            )
            insights.append(insight)
        
        return insights
    
    def _analyze_rebalancing_timing(self, portfolio_data: Dict, user_id: str) -> List[ProactiveInsight]:
        """Analyze optimal rebalancing timing"""
        insights = []
        
        # This would implement timing analysis based on market conditions
        # For now, return basic timing insight
        
        return insights
    
    def _identify_general_opportunities(self, user_id: str) -> List[ProactiveInsight]:
        """Identify general market opportunities"""
        insights = []
        
        # This would implement market-wide opportunity detection
        # For now, return empty list
        
        return insights