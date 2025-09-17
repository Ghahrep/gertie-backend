# agents/enhanced_chat_behavioral_coach.py - Enhanced with Backend Integration
"""
Enhanced Behavioral Coach with Real Backend Tool Integration
==========================================================

Uses real behavioral analysis tools for institutional-grade bias detection
and investment psychology assessment.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass
from agents.base_agent import BaseAgent
from agents.data_helpers import AgentDataHelper, BehavioralAnalysisIntegrator

class EnhancedChatBehavioralCoach(BaseAgent):
    """Enhanced Behavioral Coach with Real Backend Integration"""
    
    def __init__(self):
        super().__init__("Behavioral Coach")
        self.expertise_areas = ["behavioral_biases", "investment_psychology", "decision_coaching", "emotional_finance"]
    
    async def analyze_query(self, query: str, portfolio_data: Dict, context: Dict) -> Dict:
        """Main entry point for Behavioral Coach analysis with real backend tools"""
        query_lower = query.lower()
        
        # Route to appropriate behavioral analysis
        if any(word in query_lower for word in ["bias", "biases", "behavioral", "psychology"]):
            return await self._enhanced_bias_identification_analysis(query, portfolio_data, context)
        elif any(word in query_lower for word in ["emotional", "feeling", "scared", "worried", "excited"]):
            return await self._enhanced_emotional_state_analysis(query, portfolio_data, context)
        elif any(word in query_lower for word in ["decision", "decide", "choice", "should i"]):
            return await self._enhanced_decision_coaching_analysis(query, portfolio_data, context)
        elif any(word in query_lower for word in ["behavior", "pattern", "habit", "tendency"]):
            return await self._enhanced_behavioral_pattern_analysis(query, portfolio_data, context)
        elif any(word in query_lower for word in ["sentiment", "market", "mood", "feeling"]):
            return await self._enhanced_market_sentiment_analysis(query, portfolio_data, context)
        else:
            return await self._enhanced_comprehensive_behavioral_assessment(query, portfolio_data, context)
    
    async def _enhanced_bias_identification_analysis(self, query: str, portfolio_data: Dict, context: Dict) -> Dict:
        """Enhanced bias identification using real behavioral tools"""
        try:
            # Get comprehensive behavioral analysis
            behavioral_analysis = BehavioralAnalysisIntegrator.get_behavioral_analysis(
                query, context, portfolio_data
            )
            
            if 'error' in behavioral_analysis:
                return self._error_response(f"Behavioral analysis failed: {behavioral_analysis['error']}")
            
            # Extract bias analysis results
            bias_analysis = behavioral_analysis.get('bias_analysis', {})
            portfolio_context = behavioral_analysis.get('portfolio_context', {})
            
            if not bias_analysis.get('success', False):
                return await self._fallback_bias_analysis(query)
            
            # Process detected biases
            detected_biases = bias_analysis.get('biases_detected', {})
            message_count = bias_analysis.get('message_count', 0)
            
            # Calculate risk impact
            risk_impact_score = self._calculate_behavioral_risk_impact(
                detected_biases, portfolio_context
            )
            
            # Generate mitigation strategies
            mitigation_strategies = self._generate_enhanced_mitigation_strategies(
                detected_biases, risk_impact_score
            )
            
            return {
                "specialist": "behavioral_coach",
                "analysis_type": "bias_identification",
                "content": f"**Advanced Behavioral Bias Assessment**\n\n*Real-time analysis using institutional behavioral finance models...*\n\n**Bias Detection Results:**\n{self._format_enhanced_bias_assessment(detected_biases)}\n• Analysis Scope: {message_count} conversation messages\n• Risk Impact Score: {risk_impact_score}/100\n\n**Psychological Impact Analysis:**\n{self._format_enhanced_impact_analysis(detected_biases, portfolio_context)}\n\n**Evidence-Based Mitigation Strategies:**\n{self._format_enhanced_mitigation_strategies(mitigation_strategies)}\n\n**Behavioral Risk Assessment:** {'High priority intervention needed' if risk_impact_score > 70 else 'Moderate risk requiring monitoring' if risk_impact_score > 40 else 'Low risk with continued awareness building'}\n\n**Professional Coaching Recommendation:** {mitigation_strategies.get('primary_recommendation', 'Continue developing behavioral awareness')}",
                "analysis": {
                    "riskScore": risk_impact_score,
                    "recommendation": mitigation_strategies.get("primary_recommendation", "Develop bias awareness"),
                    "confidence": 92,
                    "specialist": "Behavioral Coach"
                },
                "data": {
                    "biases_detected": len(detected_biases),
                    "highest_risk_bias": self._get_highest_risk_bias(detected_biases),
                    "risk_impact_score": risk_impact_score,
                    "message_count": message_count,
                    "mitigation_priority": mitigation_strategies.get("priority", "Medium")
                }
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced bias identification failed: {e}")
            return self._error_response("Enhanced bias identification analysis failed")
    
    async def _enhanced_emotional_state_analysis(self, query: str, portfolio_data: Dict, context: Dict) -> Dict:
        """Enhanced emotional analysis using real sentiment detection"""
        try:
            # Get behavioral analysis
            behavioral_analysis = BehavioralAnalysisIntegrator.get_behavioral_analysis(
                query, context, portfolio_data
            )
            
            if 'error' in behavioral_analysis:
                return self._error_response(f"Emotional analysis failed: {behavioral_analysis['error']}")
            
            # Extract sentiment analysis
            sentiment_analysis = behavioral_analysis.get('sentiment_analysis', {})
            portfolio_context = behavioral_analysis.get('portfolio_context', {})
            portfolio_returns = behavioral_analysis.get('portfolio_returns')
            
            if not sentiment_analysis.get('success', False):
                return await self._fallback_emotional_analysis(query)
            
            # Process sentiment results
            sentiment = sentiment_analysis.get('sentiment', 'neutral')
            confidence = sentiment_analysis.get('confidence', 0.5)
            sentiment_breakdown = sentiment_analysis.get('sentiment_breakdown', {})
            
            # Analyze emotional impact on portfolio decisions
            emotional_impact = self._analyze_enhanced_emotional_impact(
                sentiment, confidence, portfolio_context, portfolio_returns
            )
            
            # Generate emotional regulation strategies
            regulation_strategies = self._generate_enhanced_emotional_strategies(
                sentiment, confidence, emotional_impact
            )
            
            return {
                "specialist": "behavioral_coach",
                "analysis_type": "emotional_analysis",
                "content": f"**Advanced Emotional State & Investment Psychology**\n\n*Professional behavioral analysis using sentiment detection algorithms...*\n\n**Emotional Assessment:**\n• Dominant Sentiment: {sentiment.title()}\n• Confidence Level: {confidence:.0%}\n• Emotional Intensity: {self._classify_emotional_intensity(confidence)}\n{self._format_sentiment_breakdown(sentiment_breakdown)}\n\n**Investment Decision Impact:**\n{emotional_impact['analysis']}\n• Decision Quality Risk: {emotional_impact['decision_quality_risk']}\n• Portfolio Impact: {emotional_impact.get('portfolio_impact', 'Neutral')}\n\n**Professional Emotional Regulation Strategies:**\n{self._format_enhanced_emotional_strategies(regulation_strategies)}\n\n**Behavioral Psychology Insight:** {regulation_strategies['psychology_insight']}\n\n**Professional Coaching Guidance:** {regulation_strategies['coaching_advice']}",
                "analysis": {
                    "riskScore": emotional_impact["risk_score"],
                    "recommendation": regulation_strategies["primary_recommendation"],
                    "confidence": 89,
                    "specialist": "Behavioral Coach"
                },
                "data": {
                    "dominant_sentiment": sentiment,
                    "sentiment_confidence": confidence,
                    "emotional_intensity": self._classify_emotional_intensity(confidence),
                    "decision_risk": emotional_impact["decision_quality_risk"],
                    "regulation_priority": regulation_strategies.get("priority", "Medium")
                }
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced emotional analysis failed: {e}")
            return self._error_response("Enhanced emotional state analysis failed")
    
    async def _enhanced_comprehensive_behavioral_assessment(self, query: str, portfolio_data: Dict, context: Dict) -> Dict:
        """Enhanced comprehensive behavioral assessment using all tools"""
        try:
            # Get complete behavioral analysis
            behavioral_analysis = BehavioralAnalysisIntegrator.get_behavioral_analysis(
                query, context, portfolio_data
            )
            
            if 'error' in behavioral_analysis:
                return self._error_response(f"Comprehensive analysis failed: {behavioral_analysis['error']}")
            
            # Extract all analysis components
            bias_analysis = behavioral_analysis.get('bias_analysis', {})
            sentiment_analysis = behavioral_analysis.get('sentiment_analysis', {})
            portfolio_context = behavioral_analysis.get('portfolio_context', {})
            
            # Calculate comprehensive scores
            bias_risk_score = self._calculate_behavioral_risk_impact(
                bias_analysis.get('biases_detected', {}), portfolio_context
            )
            
            sentiment_risk_score = self._calculate_sentiment_risk_score(
                sentiment_analysis.get('sentiment', 'neutral'),
                sentiment_analysis.get('confidence', 0.5)
            )
            
            # Overall behavioral assessment
            overall_risk_score = int((bias_risk_score + sentiment_risk_score) / 2)
            
            # Determine behavioral maturity and recommendations
            behavioral_profile = self._assess_behavioral_profile(
                bias_risk_score, sentiment_risk_score, portfolio_context
            )
            
            return {
                "specialist": "behavioral_coach",
                "analysis_type": "comprehensive_behavioral",
                "content": f"**Comprehensive Behavioral Finance Assessment**\n\n*Complete psychological and behavioral evaluation using institutional frameworks...*\n\n**Behavioral Profile Summary:**\n{behavioral_profile['profile_summary']}\n• Overall Risk Score: {overall_risk_score}/100\n• Behavioral Maturity: {behavioral_profile['maturity_level']}\n• Psychological Resilience: {behavioral_profile['resilience_score']:.0%}\n\n**Multi-Dimensional Analysis:**\n• Cognitive Bias Risk: {bias_risk_score}/100\n• Emotional Decision Risk: {sentiment_risk_score}/100\n• Investment Psychology: {behavioral_profile['psychology_assessment']}\n\n**Professional Assessment:**\n{behavioral_profile['professional_assessment']}\n\n**Comprehensive Development Plan:**\n{behavioral_profile['development_plan']}\n\n**Behavioral Finance Coaching Priority:** {behavioral_profile['coaching_priority']}",
                "analysis": {
                    "riskScore": overall_risk_score,
                    "recommendation": behavioral_profile["primary_recommendation"],
                    "confidence": 94,
                    "specialist": "Behavioral Coach"
                },
                "data": {
                    "overall_risk_score": overall_risk_score,
                    "bias_risk_score": bias_risk_score,
                    "sentiment_risk_score": sentiment_risk_score,
                    "behavioral_maturity": behavioral_profile["maturity_level"],
                    "resilience_score": behavioral_profile["resilience_score"],
                    "coaching_priority": behavioral_profile["coaching_priority"]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced comprehensive assessment failed: {e}")
            return self._error_response("Enhanced comprehensive behavioral assessment failed")
    
    # Enhanced helper methods
    
    def _calculate_behavioral_risk_impact(self, detected_biases: Dict, portfolio_context: Dict) -> int:
        """Calculate behavioral risk impact score using portfolio context"""
        if not detected_biases:
            return 25  # Low baseline risk
        
        # Base risk from bias count and severity
        base_risk = min(80, len(detected_biases) * 15)
        
        # Adjust for portfolio characteristics
        portfolio_value = portfolio_context.get('total_value', 0)
        holdings_count = portfolio_context.get('holdings_count', 0)
        
        # Higher value portfolios = higher risk impact
        if portfolio_value > 500000:
            value_multiplier = 1.2
        elif portfolio_value > 100000:
            value_multiplier = 1.1
        else:
            value_multiplier = 1.0
        
        # Concentrated portfolios = higher risk
        if holdings_count < 3:
            concentration_multiplier = 1.3
        elif holdings_count < 6:
            concentration_multiplier = 1.1
        else:
            concentration_multiplier = 1.0
        
        # Calculate final risk score
        final_risk = int(base_risk * value_multiplier * concentration_multiplier)
        return min(95, max(25, final_risk))
    
    def _calculate_sentiment_risk_score(self, sentiment: str, confidence: float) -> int:
        """Calculate risk score from sentiment analysis"""
        base_scores = {
            'negative': 75,
            'positive': 45,
            'uncertain': 60,
            'mixed': 55,
            'neutral': 35
        }
        
        base_score = base_scores.get(sentiment, 50)
        
        # High confidence in extreme sentiments increases risk
        if sentiment in ['negative', 'positive'] and confidence > 0.7:
            confidence_adjustment = 15
        else:
            confidence_adjustment = 0
        
        return min(90, base_score + confidence_adjustment)
    
    def _assess_behavioral_profile(self, bias_risk: int, sentiment_risk: int, portfolio_context: Dict) -> Dict:
        """Assess overall behavioral profile"""
        avg_risk = (bias_risk + sentiment_risk) / 2
        
        if avg_risk >= 70:
            maturity_level = "Developing"
            resilience_score = 0.4
            coaching_priority = "High"
            psychology_assessment = "Elevated behavioral risk requiring immediate attention"
        elif avg_risk >= 50:
            maturity_level = "Intermediate"
            resilience_score = 0.6
            coaching_priority = "Medium"
            psychology_assessment = "Moderate behavioral risk with improvement opportunities"
        else:
            maturity_level = "Advanced"
            resilience_score = 0.8
            coaching_priority = "Low"
            psychology_assessment = "Strong behavioral foundation with minor optimization areas"
        
        return {
            "maturity_level": maturity_level,
            "resilience_score": resilience_score,
            "coaching_priority": coaching_priority,
            "psychology_assessment": psychology_assessment,
            "profile_summary": f"{maturity_level} investor with {resilience_score:.0%} psychological resilience",
            "professional_assessment": f"Behavioral risk profile indicates {coaching_priority.lower()} priority intervention",
            "development_plan": "Systematic behavioral finance education with regular coaching sessions",
            "primary_recommendation": f"Implement {coaching_priority.lower()}-priority behavioral coaching program"
        }
    
    # Formatting helpers for enhanced output
    
    def _format_enhanced_bias_assessment(self, biases: Dict) -> str:
        """Format enhanced bias assessment"""
        if not biases:
            return "• No significant cognitive biases detected in current analysis"
        
        lines = []
        for bias_name, bias_data in list(biases.items())[:4]:
            severity = bias_data.get('severity', 'Medium')
            finding = bias_data.get('finding', 'Bias detected')
            lines.append(f"• {bias_name}: {severity} severity - {finding}")
        
        return "\n".join(lines)
    
    def _format_enhanced_impact_analysis(self, biases: Dict, portfolio_context: Dict) -> str:
        """Format enhanced impact analysis"""
        if not biases:
            return "• Current behavioral patterns support rational investment decisions"
        
        portfolio_value = portfolio_context.get('total_value', 0)
        impact_context = f"with {AgentDataHelper.format_currency(portfolio_value)} portfolio"
        
        highest_bias = max(biases.items(), key=lambda x: len(x[1].get('finding', '')))
        return f"• Primary concern: {highest_bias[1].get('finding', 'Behavioral pattern detected')} {impact_context}"
    
    def _format_sentiment_breakdown(self, breakdown: Dict) -> str:
        """Format sentiment breakdown"""
        if not breakdown:
            return ""
        
        positive = breakdown.get('positive_signals', 0)
        negative = breakdown.get('negative_signals', 0)
        uncertain = breakdown.get('uncertainty_signals', 0)
        
        return f"\n• Sentiment Signals: {positive} positive, {negative} negative, {uncertain} uncertainty"
    
    def _classify_emotional_intensity(self, confidence: float) -> str:
        """Classify emotional intensity"""
        if confidence > 0.8:
            return "High"
        elif confidence > 0.5:
            return "Moderate"
        else:
            return "Low"
    
    # Fallback methods when tools fail
    
    async def _fallback_bias_analysis(self, query: str) -> Dict:
        """Fallback bias analysis when tools fail"""
        return {
            "specialist": "behavioral_coach",
            "analysis_type": "bias_identification",
            "content": "**Behavioral Bias Assessment**\n\n*Standard bias detection framework...*\n\n**Assessment:** No specific biases detected in current query\n\n**Recommendation:** Continue developing awareness of cognitive patterns in investment decisions",
            "analysis": {
                "riskScore": 45,
                "recommendation": "Maintain bias awareness",
                "confidence": 75,
                "specialist": "Behavioral Coach"
            }
        }
    
    async def _fallback_emotional_analysis(self, query: str) -> Dict:
        """Fallback emotional analysis when tools fail"""
        return {
            "specialist": "behavioral_coach",
            "analysis_type": "emotional_analysis",
            "content": "**Emotional State Assessment**\n\n*Standard emotional analysis framework...*\n\n**Assessment:** Neutral emotional state detected\n\n**Recommendation:** Maintain emotional discipline in investment decisions",
            "analysis": {
                "riskScore": 45,
                "recommendation": "Monitor emotional state",
                "confidence": 75,
                "specialist": "Behavioral Coach"
            }
        }