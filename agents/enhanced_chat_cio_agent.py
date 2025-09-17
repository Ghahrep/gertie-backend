# agents/enhanced_chat_cio_agent.py - Enhanced with Backend Integration
"""
Enhanced CIO Agent with Real Backend Tool Integration
===================================================

Replaces mock regime analysis with institutional-grade market regime detection
and strategic asset allocation tools.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass
from agents.base_agent import BaseAgent
from agents.data_helpers import AgentDataHelper, get_regime_analysis

class EnhancedChatCIOAgent(BaseAgent):
    """Enhanced Chief Investment Officer with Real Backend Integration"""
    
    def __init__(self):
        super().__init__("Chief Investment Officer")
        self.expertise_areas = ["strategic_allocation", "market_outlook", "investment_strategy", "portfolio_construction"]
    
    async def analyze_query(self, query: str, portfolio_data: Dict, context: Dict) -> Dict:
        """Main entry point for CIO analysis with real backend tools"""
        query_lower = query.lower()
        
        # Route to appropriate CIO analysis
        if any(word in query_lower for word in ["strategy", "strategic", "allocation", "allocate"]):
            return await self._enhanced_strategic_allocation_analysis(portfolio_data, context)
        elif any(word in query_lower for word in ["market", "outlook", "economic", "macro"]):
            return await self._enhanced_market_outlook_analysis(portfolio_data, context)
        elif any(word in query_lower for word in ["diversify", "diversification", "spread"]):
            return await self._enhanced_diversification_analysis(portfolio_data, context)
        elif any(word in query_lower for word in ["rebalance", "optimize", "improve"]):
            return await self._enhanced_portfolio_optimization_analysis(portfolio_data, context)
        elif any(word in query_lower for word in ["long term", "future", "goals", "planning"]):
            return await self._enhanced_long_term_planning_analysis(portfolio_data, context)
        else:
            return await self._enhanced_comprehensive_strategic_review(portfolio_data, context)
    
    async def _enhanced_market_outlook_analysis(self, portfolio_data: Dict, context: Dict) -> Dict:
        """Enhanced market outlook using real regime detection tools"""
        try:
            holdings = portfolio_data.get("holdings", [])
            total_value = portfolio_data.get("total_value", 100000)
            
            # Get real regime analysis from backend tools
            regime_analysis = get_regime_analysis(portfolio_data)
            
            if 'error' in regime_analysis:
                return self._fallback_market_outlook()
            
            # Extract regime information
            hmm_regimes = regime_analysis.get('hmm_regimes', {})
            vol_regimes = regime_analysis.get('volatility_regimes', {})
            
            # Process HMM regime results
            current_regime = hmm_regimes.get('current_regime', 0)
            regime_probs = hmm_regimes.get('regime_probabilities', [0.5, 0.5])
            transition_matrix = hmm_regimes.get('transition_matrix', [])
            
            # Process volatility regime results
            vol_state = vol_regimes.get('current_regime', 'Normal')
            vol_persistence = vol_regimes.get('regime_persistence', {})
            
            # Determine regime characteristics
            if current_regime == 0:
                regime_name = "Low Volatility Growth"
                regime_confidence = regime_probs[0] if len(regime_probs) > 0 else 0.7
                market_character = "Constructive"
                strategic_bias = "Growth-oriented"
            else:
                regime_name = "High Volatility Stress"
                regime_confidence = regime_probs[1] if len(regime_probs) > 1 else 0.7
                market_character = "Challenging"
                strategic_bias = "Defensive"
            
            # Generate market outlook
            outlook_summary = self._generate_regime_based_outlook(
                regime_name, regime_confidence, vol_state
            )
            
            # Analyze portfolio fit with regime
            portfolio_fit = self._analyze_portfolio_regime_fit(
                holdings, total_value, current_regime, vol_state
            )
            
            # Strategic recommendations based on regime
            strategic_recommendations = self._generate_regime_strategic_recommendations(
                current_regime, vol_state, portfolio_fit
            )
            
            return {
                "specialist": "cio",
                "analysis_type": "market_outlook",
                "content": f"**CIO Strategic Market Assessment**\n\n*Advanced regime detection and strategic positioning analysis...*\n\n**Market Regime Analysis:**\n• Current Regime: {regime_name}\n• Regime Confidence: {regime_confidence:.0%}\n• Volatility State: {vol_state}\n• Market Character: {market_character}\n\n**Strategic Market Outlook:**\n{outlook_summary}\n\n**Portfolio Regime Positioning:**\n{portfolio_fit['analysis']}\n• Strategic Alignment: {portfolio_fit['alignment']}\n• Regime Fitness Score: {portfolio_fit['fitness_score']}/10\n\n**Strategic Positioning Recommendations:**\n{self._format_strategic_positioning(strategic_recommendations)}\n\n**Risk Management Priority:** {portfolio_fit['risk_priority']}\n\n**Strategic Horizon:** Next regime transition probability: {(1-regime_confidence)*100:.0f}% within 3-6 months",
                "analysis": {
                    "riskScore": portfolio_fit["risk_score"],
                    "recommendation": strategic_recommendations["primary_recommendation"],
                    "confidence": int(regime_confidence * 100),
                    "specialist": "Chief Investment Officer"
                },
                "data": {
                    "current_regime": current_regime,
                    "regime_name": regime_name,
                    "regime_confidence": regime_confidence,
                    "volatility_state": vol_state,
                    "portfolio_fitness": portfolio_fit["fitness_score"],
                    "strategic_bias": strategic_bias
                }
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced market outlook analysis failed: {e}")
            return self._error_response("Enhanced market outlook analysis failed")
    
    async def _enhanced_strategic_allocation_analysis(self, portfolio_data: Dict, context: Dict) -> Dict:
        """Enhanced strategic allocation analysis using regime-aware frameworks"""
        try:
            total_value = portfolio_data.get("total_value", 100000)
            holdings = portfolio_data.get("holdings", [])
            
            # Get regime analysis for strategic context
            regime_analysis = get_regime_analysis(portfolio_data)
            
            # Calculate current allocation
            current_allocation = self._calculate_current_allocation(holdings, total_value)
            
            # Get regime-conditional target allocation
            if 'error' not in regime_analysis:
                hmm_regimes = regime_analysis.get('hmm_regimes', {})
                current_regime = hmm_regimes.get('current_regime', 0)
                target_allocation = self._get_regime_based_target_allocation(current_regime, total_value)
            else:
                target_allocation = self._get_standard_target_allocation(total_value)
                current_regime = 0
            
            # Analyze allocation quality
            allocation_analysis = self._analyze_strategic_allocation_quality(
                current_allocation, target_allocation, current_regime
            )
            
            # Generate strategic recommendations
            strategic_recommendations = self._generate_strategic_allocation_recommendations(
                current_allocation, target_allocation, allocation_analysis
            )
            
            return {
                "specialist": "cio",
                "analysis_type": "strategic_allocation",
                "content": f"**Strategic Asset Allocation Analysis**\n\n*Regime-aware strategic allocation with institutional frameworks...*\n\n**Current Strategic Position:**\n{self._format_allocation_comparison(current_allocation, target_allocation)}\n\n**Strategic Assessment:**\n{allocation_analysis['assessment']}\n• Allocation Score: {allocation_analysis['strategic_score']}/100\n• Regime Alignment: {allocation_analysis['regime_alignment']}\n• Strategic Drift: {allocation_analysis['total_drift']:.1%}\n\n**Strategic Recommendations:**\n{self._format_strategic_recommendations(strategic_recommendations)}\n\n**Implementation Framework:**\n{strategic_recommendations['implementation_framework']}\n\n**Strategic Priority:** {allocation_analysis['priority']}",
                "analysis": {
                    "riskScore": allocation_analysis["risk_score"],
                    "recommendation": strategic_recommendations["primary_recommendation"],
                    "confidence": allocation_analysis["confidence"],
                    "specialist": "Chief Investment Officer"
                },
                "data": {
                    "current_allocation": current_allocation,
                    "target_allocation": target_allocation,
                    "strategic_score": allocation_analysis["strategic_score"],
                    "allocation_drift": allocation_analysis["total_drift"],
                    "regime_alignment": allocation_analysis["regime_alignment"]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced strategic allocation analysis failed: {e}")
            return self._error_response("Enhanced strategic allocation analysis failed")
    
    # Helper methods for enhanced analysis
    
    def _generate_regime_based_outlook(self, regime_name: str, confidence: float, vol_state: str) -> str:
        """Generate market outlook based on regime analysis"""
        if "Low Volatility" in regime_name:
            base_outlook = "Favorable environment for growth-oriented strategies with controlled risk-taking"
            risk_factors = "Monitor for regime transition signals and emerging volatility"
        else:
            base_outlook = "Challenging environment requiring defensive positioning and risk management"
            risk_factors = "Focus on capital preservation and quality positioning"
        
        confidence_qualifier = "high confidence" if confidence > 0.8 else "moderate confidence" if confidence > 0.6 else "developing signals"
        
        return f"{base_outlook} ({confidence_qualifier}). {risk_factors}. Volatility state: {vol_state}."
    
    def _analyze_portfolio_regime_fit(self, holdings: List, total_value: float, 
                                    current_regime: int, vol_state: str) -> Dict:
        """Analyze how well portfolio fits current market regime"""
        holdings_count = len(holdings)
        
        # Regime fitness scoring
        if current_regime == 0:  # Low volatility regime
            # Growth-oriented portfolios fit better
            if holdings_count >= 6 and total_value > 50000:
                fitness_score = 8
                alignment = "Well-positioned"
                analysis = "Portfolio structure aligns with growth-oriented regime characteristics"
                risk_score = 45
                risk_priority = "Medium"
            else:
                fitness_score = 6
                alignment = "Adequately positioned"
                analysis = "Portfolio can participate in growth regime with some optimization"
                risk_score = 55
                risk_priority = "Medium"
        else:  # High volatility regime
            # Defensive characteristics preferred
            if holdings_count >= 8:  # More diversification is defensive
                fitness_score = 7
                alignment = "Defensively positioned"
                analysis = "Portfolio diversification provides good defense against volatility"
                risk_score = 60
                risk_priority = "Medium"
            else:
                fitness_score = 4
                alignment = "Exposed to regime stress"
                analysis = "Portfolio concentration creates vulnerability in stress regime"
                risk_score = 75
                risk_priority = "High"
        
        return {
            "fitness_score": fitness_score,
            "alignment": alignment,
            "analysis": analysis,
            "risk_score": risk_score,
            "risk_priority": risk_priority
        }
    
    def _generate_regime_strategic_recommendations(self, current_regime: int, 
                                                 vol_state: str, portfolio_fit: Dict) -> Dict:
        """Generate strategic recommendations based on regime analysis"""
        if current_regime == 0:  # Low volatility regime
            primary_rec = "Optimize for growth with systematic risk management"
            recommendations = [
                "Maintain growth-oriented allocation with quality focus",
                "Prepare defensive positions for regime transition",
                "Optimize position sizing for return enhancement"
            ]
            implementation = "Gradual implementation with regime monitoring"
        else:  # High volatility regime
            primary_rec = "Implement defensive positioning with opportunistic elements"
            recommendations = [
                "Increase defensive positioning and quality focus",
                "Reduce concentration risk through diversification",
                "Maintain liquidity for regime transition opportunities"
            ]
            implementation = "Immediate defensive measures with gradual optimization"
        
        return {
            "primary_recommendation": primary_rec,
            "recommendations": recommendations,
            "implementation_framework": implementation
        }
    
    def _get_regime_based_target_allocation(self, current_regime: int, total_value: float) -> Dict:
        """Get target allocation based on current market regime"""
        if current_regime == 0:  # Low volatility - growth allocation
            if total_value > 100000:
                return {"equities": 0.70, "bonds": 0.20, "alternatives": 0.05, "cash": 0.05}
            else:
                return {"equities": 0.65, "bonds": 0.25, "alternatives": 0.0, "cash": 0.10}
        else:  # High volatility - defensive allocation
            if total_value > 100000:
                return {"equities": 0.50, "bonds": 0.35, "alternatives": 0.10, "cash": 0.05}
            else:
                return {"equities": 0.45, "bonds": 0.40, "alternatives": 0.0, "cash": 0.15}
    
    def _get_standard_target_allocation(self, total_value: float) -> Dict:
        """Get standard target allocation when regime analysis unavailable"""
        if total_value > 100000:
            return {"equities": 0.60, "bonds": 0.30, "alternatives": 0.05, "cash": 0.05}
        else:
            return {"equities": 0.55, "bonds": 0.35, "alternatives": 0.0, "cash": 0.10}
    
    def _calculate_current_allocation(self, holdings: List, total_value: float) -> Dict:
        """Calculate current asset allocation with enhanced classification"""
        if not holdings or total_value == 0:
            return {"equities": 0.0, "bonds": 0.0, "cash": 1.0, "alternatives": 0.0}
        
        # Enhanced asset classification
        equity_value = 0
        bond_value = 0
        alt_value = 0
        
        for holding in holdings:
            value = holding.get("value", 0) or holding.get("market_value", 0)
            ticker = holding.get("ticker", holding.get("symbol", ""))
            
            # Simple classification (in production, use proper asset classification)
            if any(bond_indicator in ticker.upper() for bond_indicator in ['BND', 'AGG', 'TLT', 'IEF']):
                bond_value += value
            elif any(alt_indicator in ticker.upper() for alt_indicator in ['GLD', 'VNQ', 'REIT']):
                alt_value += value
            else:
                equity_value += value
        
        cash_value = max(0, total_value - equity_value - bond_value - alt_value)
        
        return {
            "equities": equity_value / total_value,
            "bonds": bond_value / total_value,
            "alternatives": alt_value / total_value,
            "cash": cash_value / total_value
        }
    
    def _analyze_strategic_allocation_quality(self, current: Dict, target: Dict, regime: int) -> Dict:
        """Analyze strategic allocation quality with regime context"""
        # Calculate allocation deviations
        total_deviation = 0
        max_deviation = 0
        deviations = {}
        
        for asset_class, target_weight in target.items():
            current_weight = current.get(asset_class, 0)
            deviation = abs(current_weight - target_weight)
            deviations[asset_class] = deviation
            total_deviation += deviation
            max_deviation = max(max_deviation, deviation)
        
        # Regime-aware assessment
        regime_context = "growth regime" if regime == 0 else "defensive regime"
        
        if total_deviation < 0.15:
            assessment = f"Well-aligned with strategic targets for {regime_context}"
            priority = "Low"
            risk_score = 40
            strategic_score = 85
            regime_alignment = "Strong"
        elif total_deviation < 0.30:
            assessment = f"Moderate deviation from optimal {regime_context} allocation"
            priority = "Medium"
            risk_score = 60
            strategic_score = 65
            regime_alignment = "Adequate"
        else:
            assessment = f"Significant misalignment with {regime_context} strategy"
            priority = "High"
            risk_score = 80
            strategic_score = 40
            regime_alignment = "Weak"
        
        return {
            "assessment": assessment,
            "priority": priority,
            "risk_score": risk_score,
            "strategic_score": strategic_score,
            "regime_alignment": regime_alignment,
            "total_drift": total_deviation,
            "max_deviation": max_deviation,
            "confidence": 0.90
        }
    
    def _generate_strategic_allocation_recommendations(self, current: Dict, target: Dict, analysis: Dict) -> Dict:
        """Generate strategic allocation recommendations"""
        recommendations = []
        implementation_steps = []
        
        # Generate specific recommendations based on deviations
        for asset_class, target_weight in target.items():
            current_weight = current.get(asset_class, 0)
            deviation = current_weight - target_weight
            
            if abs(deviation) > 0.10:  # 10% threshold
                if deviation > 0:
                    recommendations.append(f"Reduce {asset_class} allocation by {deviation:.1%}")
                    implementation_steps.append(f"Trim {asset_class} positions gradually")
                else:
                    recommendations.append(f"Increase {asset_class} allocation by {abs(deviation):.1%}")
                    implementation_steps.append(f"Build {asset_class} positions systematically")
        
        primary_recommendation = recommendations[0] if recommendations else "Maintain current strategic allocation"
        
        implementation_framework = "Phased implementation over 2-3 months with regime monitoring" if len(implementation_steps) > 2 else "Direct implementation with market timing consideration"
        
        return {
            "primary_recommendation": primary_recommendation,
            "recommendations": recommendations,
            "implementation_steps": implementation_steps,
            "implementation_framework": implementation_framework
        }
    
    # Formatting helpers
    
    def _format_allocation_comparison(self, current: Dict, target: Dict) -> str:
        """Format allocation comparison for display"""
        lines = []
        for asset_class in target.keys():
            current_pct = current.get(asset_class, 0) * 100
            target_pct = target.get(asset_class, 0) * 100
            deviation = current_pct - target_pct
            
            lines.append(f"• {asset_class.title()}: {current_pct:.1f}% (target: {target_pct:.1f}%, deviation: {deviation:+.1f}%)")
        
        return "\n".join(lines)
    
    def _format_strategic_recommendations(self, recommendations: Dict) -> str:
        """Format strategic recommendations for display"""
        recs = recommendations.get("recommendations", [])
        return "\n".join(f"• {rec}" for rec in recs[:4]) if recs else "• Current allocation is strategically optimal"
    
    def _format_strategic_positioning(self, recommendations: Dict) -> str:
        """Format strategic positioning recommendations"""
        recs = recommendations.get("recommendations", [])
        return "\n".join(f"• {rec}" for rec in recs[:3]) if recs else "• Maintain current strategic positioning"
    
    def _fallback_market_outlook(self) -> Dict:
        """Fallback market outlook when regime analysis fails"""
        return {
            "specialist": "cio",
            "analysis_type": "market_outlook",
            "content": "**Strategic Market Assessment**\n\n*Standard market outlook framework...*\n\n**Market Environment:**\n• Regime: Mixed signals\n• Outlook: Balanced approach recommended\n• Strategy: Maintain diversified positioning\n\n**Strategic Recommendation:** Continue systematic investment approach with regular portfolio reviews",
            "analysis": {
                "riskScore": 55,
                "recommendation": "Maintain balanced strategic approach",
                "confidence": 75,
                "specialist": "Chief Investment Officer"
            }
        }
    
    async def _enhanced_comprehensive_strategic_review(self, portfolio_data: Dict, context: Dict) -> Dict:
        """Enhanced comprehensive strategic review"""
        try:
            # Combine market outlook and strategic allocation analysis
            market_analysis = await self._enhanced_market_outlook_analysis(portfolio_data, context)
            allocation_analysis = await self._enhanced_strategic_allocation_analysis(portfolio_data, context)
            
            # Extract key metrics
            market_risk = market_analysis.get("analysis", {}).get("riskScore", 50)
            allocation_risk = allocation_analysis.get("analysis", {}).get("riskScore", 50)
            overall_risk = int((market_risk + allocation_risk) / 2)
            
            return {
                "specialist": "cio",
                "analysis_type": "comprehensive_strategic_review",
                "content": f"**Comprehensive Strategic Portfolio Review**\n\n*Integrated market outlook and strategic allocation analysis...*\n\n**Strategic Assessment Summary:**\n• Overall Strategic Risk: {overall_risk}/100\n• Market Environment Assessment: {market_analysis.get('data', {}).get('regime_name', 'Mixed signals')}\n• Strategic Alignment: {allocation_analysis.get('data', {}).get('regime_alignment', 'Under review')}\n\n**Integrated Recommendations:**\n• Market positioning: {market_analysis.get('analysis', {}).get('recommendation', 'Monitor conditions')}\n• Strategic allocation: {allocation_analysis.get('analysis', {}).get('recommendation', 'Review allocation')}\n\n**Strategic Priority:** Balanced approach with regime awareness",
                "analysis": {
                    "riskScore": overall_risk,
                    "recommendation": "Implement integrated strategic approach",
                    "confidence": 88,
                    "specialist": "Chief Investment Officer"
                }
            }
        except Exception as e:
            self.logger.error(f"Comprehensive strategic review failed: {e}")
            return self._error_response("Comprehensive strategic review failed")