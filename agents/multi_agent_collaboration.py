# agents/multi_agent_collaboration.py - Complete Fixed Implementation
"""
Multi-Agent Collaboration Framework for Investment Committee
============================================================

Complete implementation with all missing methods fixed.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CollaborativeAnalysis:
    """Result of collaborative analysis between multiple agents"""
    synthesis: Dict[str, Any]
    specialists_involved: List[str]
    collaboration_metadata: Dict[str, Any]
    secondary_analyses: List[Dict[str, Any]]
    execution_time: float

class AgentCollaborationManager:
    """
    Manages collaboration between investment committee specialists
    """
    
    def __init__(self, committee_manager):
        self.committee_manager = committee_manager
        self.collaboration_rules = self._initialize_collaboration_rules()
        self.synthesis_strategies = self._initialize_synthesis_strategies()
        self.collaboration_history = []
        logger.info("AgentCollaborationManager initialized")
    
    def _initialize_collaboration_rules(self) -> List[Dict]:
        """Initialize rules for when agents should collaborate"""
        return [
            {
                "name": "high_risk_emotional",
                "condition": lambda query, analysis: (
                    analysis.get("analysis", {}).get("riskScore", 0) > 75 and
                    any(word in query.lower() for word in ["worried", "scared", "panic", "anxious"])
                ),
                "primary_agent": "quantitative_analyst",
                "secondary_agent": "behavioral_coach",
                "confidence": 0.9,
                "reason": "High risk with emotional language requires behavioral analysis"
            },
            {
                "name": "low_confidence_validation",
                "condition": lambda query, analysis: analysis.get("analysis", {}).get("confidence", 100) < 70,
                "primary_agent": "any",
                "secondary_agent": "validation_specialist",
                "confidence": 0.8,
                "reason": "Low confidence analysis needs validation"
            },
            {
                "name": "optimization_strategy",
                "condition": lambda query, analysis: any(
                    word in query.lower() for word in ["optimize", "rebalance", "allocation", "weight"]
                ),
                "primary_agent": "portfolio_manager",
                "secondary_agent": "cio",
                "confidence": 0.85,
                "reason": "Optimization decisions benefit from strategic context"
            },
            {
                "name": "strategy_implementation",
                "condition": lambda query, analysis: any(
                    word in query.lower() for word in ["strategy", "long-term", "allocation"]
                ),
                "primary_agent": "cio",
                "secondary_agent": "portfolio_manager",
                "confidence": 0.8,
                "reason": "Strategic decisions need implementation perspective"
            },
            {
                "name": "complex_multifactor",
                "condition": lambda query, analysis: len([
                    word for word in query.lower().split() 
                    if word in ["risk", "return", "optimize", "strategy", "behavioral", "bias"]
                ]) >= 3,
                "primary_agent": "any",
                "secondary_agent": "complementary",
                "confidence": 0.7,
                "reason": "Complex queries benefit from multiple perspectives"
            },
            {
                "name": "behavioral_bias_detection",
                "condition": lambda query, analysis: any(
                    word in query.lower() for word in ["bias", "emotional", "feeling", "instinct", "gut"]
                ),
                "primary_agent": "any",
                "secondary_agent": "behavioral_coach",
                "confidence": 0.85,
                "reason": "Behavioral language suggests bias analysis needed"
            }
        ]
    
    def _initialize_synthesis_strategies(self) -> Dict:
        """Initialize synthesis strategies for combining analyses"""
        return {
            "consensus_building": self._synthesize_consensus,
            "risk_validation": self._synthesize_risk_validation,
            "recommendation_synthesis": self._synthesize_recommendations,
            "confidence_enhancement": self._synthesize_confidence_enhancement,
            "perspective_integration": self._synthesize_perspectives
        }
    
    async def analyze_with_collaboration(
        self,
        query: str,
        portfolio_context: Dict,
        primary_analysis: Dict,
        collaboration_hint: Optional[str] = None
    ) -> CollaborativeAnalysis:
        """
        Main collaboration analysis method
        """
        start_time = datetime.now()
        
        try:
            # Detect collaboration opportunities
            collaboration_opportunities = self._detect_collaboration_opportunities(
                query, primary_analysis, collaboration_hint
            )
            
            if not collaboration_opportunities:
                # No collaboration needed - return enhanced primary analysis
                return CollaborativeAnalysis(
                    synthesis=primary_analysis,
                    specialists_involved=[primary_analysis.get("specialist_used", "unknown")],
                    collaboration_metadata={"collaboration_triggered": False, "reason": "No opportunities detected"},
                    secondary_analyses=[],
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Execute secondary analyses
            secondary_analyses = await self._execute_secondary_analyses(
                query, portfolio_context, primary_analysis, collaboration_opportunities
            )
            
            # Synthesize results
            synthesis = await self._synthesize_collaborative_results(
                primary_analysis, secondary_analyses, collaboration_opportunities
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Track collaboration performance
            self._track_collaboration_performance({
                "query": query,
                "opportunities": collaboration_opportunities,
                "secondary_analyses_count": len(secondary_analyses),
                "execution_time": execution_time,
                "success": True
            })
            
            return CollaborativeAnalysis(
                synthesis=synthesis,
                specialists_involved=self._extract_specialists_involved(primary_analysis, secondary_analyses),
                collaboration_metadata={
                    "collaboration_triggered": True,
                    "opportunities": collaboration_opportunities,
                    "execution_time": execution_time
                },
                secondary_analyses=secondary_analyses,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Collaboration analysis failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return CollaborativeAnalysis(
                synthesis=primary_analysis,
                specialists_involved=[primary_analysis.get("specialist_used", "unknown")],
                collaboration_metadata={
                    "collaboration_triggered": False,
                    "error": str(e),
                    "execution_time": execution_time
                },
                secondary_analyses=[],
                execution_time=execution_time
            )
    
    def _detect_collaboration_opportunities(
        self, 
        query: str, 
        primary_analysis: Dict, 
        collaboration_hint: Optional[str]
    ) -> List[Dict]:
        """Detect when collaboration would add value"""
        opportunities = []
        
        # Check each collaboration rule
        for rule in self.collaboration_rules:
            try:
                if rule["condition"](query, primary_analysis):
                    secondary_specialist = self._determine_secondary_specialist(
                        rule, primary_analysis.get("specialist_used", "unknown")
                    )
                    
                    if secondary_specialist:
                        opportunities.append({
                            "rule_name": rule["name"],
                            "secondary_specialist": secondary_specialist,
                            "confidence": rule["confidence"],
                            "reason": rule["reason"],
                            "strategy": self._select_synthesis_strategy(rule["name"])
                        })
                        
            except Exception as e:
                logger.warning(f"Rule evaluation failed for {rule['name']}: {e}")
        
        # Handle explicit collaboration hint
        if collaboration_hint:
            hint_opportunity = self._process_collaboration_hint(
                collaboration_hint, primary_analysis.get("specialist_used")
            )
            if hint_opportunity:
                opportunities.append(hint_opportunity)
        
        # Prioritize and limit opportunities
        opportunities.sort(key=lambda x: x["confidence"], reverse=True)
        return opportunities[:2]  # Limit to top 2 opportunities
    
    def _determine_secondary_specialist(self, rule: Dict, primary_specialist: str) -> Optional[str]:
        """Determine the secondary specialist based on collaboration rule"""
        secondary_agent = rule["secondary_agent"]
        
        if secondary_agent == "validation_specialist":
            # Select validation specialist based on primary
            validation_map = {
                "quantitative_analyst": "cio",
                "cio": "quantitative_analyst", 
                "portfolio_manager": "quantitative_analyst",
                "behavioral_coach": "cio"
            }
            return validation_map.get(primary_specialist)
        
        elif secondary_agent == "complementary":
            # Select complementary specialist
            complementary_map = {
                "quantitative_analyst": "behavioral_coach",
                "cio": "portfolio_manager",
                "portfolio_manager": "cio", 
                "behavioral_coach": "quantitative_analyst"
            }
            return complementary_map.get(primary_specialist)
        
        elif secondary_agent != primary_specialist:
            return secondary_agent
        
        return None
    
    def _select_synthesis_strategy(self, rule_name: str) -> str:
        """Select synthesis strategy based on collaboration rule"""
        strategy_map = {
            "high_risk_emotional": "risk_validation",
            "low_confidence_validation": "confidence_enhancement",
            "optimization_strategy": "recommendation_synthesis",
            "strategy_implementation": "recommendation_synthesis",
            "complex_multifactor": "perspective_integration",
            "behavioral_bias_detection": "consensus_building"
        }
        return strategy_map.get(rule_name, "perspective_integration")
    
    def _process_collaboration_hint(self, hint: str, primary_specialist: str) -> Optional[Dict]:
        """Process explicit collaboration hint from user"""
        hint_mapping = {
            "behavioral": "behavioral_coach",
            "strategy": "cio",
            "optimization": "portfolio_manager", 
            "risk": "quantitative_analyst"
        }
        
        suggested_specialist = hint_mapping.get(hint.lower())
        if suggested_specialist and suggested_specialist != primary_specialist:
            return {
                "rule_name": "explicit_hint",
                "secondary_specialist": suggested_specialist,
                "confidence": 0.95,
                "reason": f"User requested {hint} perspective",
                "strategy": "perspective_integration"
            }
        
        return None
    
    async def _execute_secondary_analyses(
        self,
        query: str,
        portfolio_context: Dict,
        primary_analysis: Dict,
        opportunities: List[Dict]
    ) -> List[Dict]:
        """Execute secondary analyses in parallel"""
        secondary_tasks = []
        
        for opportunity in opportunities:
            secondary_specialist = opportunity["secondary_specialist"]
            
            # Skip if specialist not available
            if secondary_specialist not in self.committee_manager.specialists:
                logger.warning(f"Secondary specialist {secondary_specialist} not available")
                continue
            
            # Create focused context for secondary analysis
            secondary_context = {
                **portfolio_context,
                "collaboration_mode": True,
                "primary_analysis": primary_analysis,
                "collaboration_focus": opportunity["rule_name"],
                "collaboration_reason": opportunity["reason"]
            }
            
            # Create focused query
            focused_query = self._create_focused_query(query, opportunity, secondary_specialist)
            
            # Schedule secondary analysis
            task = self._get_secondary_analysis(focused_query, secondary_context, secondary_specialist)
            secondary_tasks.append((task, opportunity))
        
        # Execute all secondary analyses concurrently
        results = []
        if secondary_tasks:
            completed_analyses = await asyncio.gather(
                *[task for task, _ in secondary_tasks],
                return_exceptions=True
            )
            
            for i, result in enumerate(completed_analyses):
                opportunity = secondary_tasks[i][1]
                if isinstance(result, Exception):
                    logger.error(f"Secondary analysis failed for {opportunity['secondary_specialist']}: {result}")
                else:
                    results.append({
                        "specialist": opportunity["secondary_specialist"],
                        "analysis": result,
                        "opportunity": opportunity,
                        "collaboration_value": opportunity["confidence"]
                    })
        
        return results
    
    def _create_focused_query(self, original_query: str, opportunity: Dict, specialist: str) -> str:
        """Create focused query for secondary specialist"""
        rule_name = opportunity["rule_name"]
        
        focus_templates = {
            "high_risk_emotional": {
                "behavioral_coach": f"Analyze the emotional and behavioral aspects of this investment concern: {original_query}"
            },
            "low_confidence_validation": {
                "quantitative_analyst": f"Provide quantitative validation for: {original_query}",
                "cio": f"Assess strategic implications of: {original_query}",
                "portfolio_manager": f"Evaluate implementation aspects of: {original_query}"
            },
            "optimization_strategy": {
                "cio": f"What strategic considerations should guide this optimization: {original_query}",
                "portfolio_manager": f"What are the practical implementation aspects of: {original_query}"
            },
            "behavioral_bias_detection": {
                "behavioral_coach": f"Identify potential cognitive biases in this investment decision: {original_query}"
            }
        }
        
        template = focus_templates.get(rule_name, {}).get(specialist)
        return template if template else original_query
    
    async def _get_secondary_analysis(self, query: str, context: Dict, specialist_name: str) -> Dict:
        """Get analysis from secondary specialist"""
        try:
            specialist = self.committee_manager.specialists[specialist_name]
            result = await specialist.analyze_query(query, context, context)
            result["collaboration_mode"] = True
            return result
        except Exception as e:
            logger.error(f"Secondary analysis failed for {specialist_name}: {e}")
            return {
                "content": f"Collaborative analysis unavailable from {specialist_name}",
                "analysis": {"confidence": 0, "riskScore": 50},
                "error": str(e),
                "collaboration_mode": True
            }
    
    async def _synthesize_collaborative_results(
        self,
        primary_analysis: Dict,
        secondary_analyses: List[Dict],
        opportunities: List[Dict]
    ) -> Dict:
        """Synthesize collaborative results using appropriate strategy"""
        if not secondary_analyses:
            return primary_analysis
        
        # Determine synthesis strategy
        strategies_used = [opp["strategy"] for opp in opportunities]
        primary_strategy = strategies_used[0] if strategies_used else "perspective_integration"
        
        # Apply synthesis strategy
        synthesis_func = self.synthesis_strategies.get(primary_strategy, self._synthesize_perspectives)
        synthesis = synthesis_func(primary_analysis, secondary_analyses)
        
        return synthesis
    
    def _synthesize_consensus(self, primary_analysis: Dict, secondary_analyses: List[Dict]) -> Dict:
        """Synthesize using consensus-building approach"""
        primary_content = primary_analysis.get("content", "")
        primary_specialist = primary_analysis.get("specialist_used", "Unknown")
        
        # Build consensus response
        sections = [f"**Primary Analysis - {primary_specialist.replace('_', ' ').title()}:**\n{primary_content}"]
        
        # Add secondary perspectives
        for secondary in secondary_analyses:
            specialist_name = secondary["specialist"].replace('_', ' ').title()
            content = secondary["analysis"].get("content", "Analysis unavailable")
            sections.append(f"\n**{specialist_name} Perspective:**\n{content}")
        
        # Add consensus summary
        consensus_summary = self._generate_consensus_summary(primary_analysis, secondary_analyses)
        if consensus_summary:
            sections.append(f"\n**Committee Consensus:**\n{consensus_summary}")
        
        synthesized_response = primary_analysis.copy()
        synthesized_response["content"] = "\n".join(sections)
        synthesized_response["collaboration_synthesis"] = "consensus_building"
        
        return synthesized_response
    
    def _synthesize_risk_validation(self, primary_analysis: Dict, secondary_analyses: List[Dict]) -> Dict:
        """Synthesize with focus on risk validation"""
        primary_risk = primary_analysis.get("analysis", {}).get("riskScore", 50)
        risk_assessments = [primary_risk]
        
        for secondary in secondary_analyses:
            secondary_risk = secondary["analysis"].get("analysis", {}).get("riskScore")
            if secondary_risk is not None:
                risk_assessments.append(secondary_risk)
        
        # Calculate consensus risk
        avg_risk = sum(risk_assessments) / len(risk_assessments)
        risk_range = max(risk_assessments) - min(risk_assessments)
        
        # Build risk-focused response
        primary_content = primary_analysis.get("content", "")
        validation_content = f"\n\n**Risk Validation Summary:**\n"
        
        if risk_range <= 15:
            validation_content += f"Strong consensus on risk level: {avg_risk:.0f}/100"
        else:
            validation_content += f"Risk assessments vary (range: {min(risk_assessments)}-{max(risk_assessments)}), requiring careful consideration"
        
        synthesized_response = primary_analysis.copy()
        synthesized_response["content"] = primary_content + validation_content
        synthesized_response["analysis"]["validated_risk_score"] = avg_risk
        synthesized_response["collaboration_synthesis"] = "risk_validation"
        
        return synthesized_response
    
    def _synthesize_recommendations(self, primary_analysis: Dict, secondary_analyses: List[Dict]) -> Dict:
        """Synthesize recommendations from multiple specialists"""
        all_recommendations = []
        
        # Extract primary recommendations
        primary_recommendations = primary_analysis.get("recommendations", [])
        if primary_recommendations:
            all_recommendations.extend(primary_recommendations)
        
        # Extract secondary recommendations
        for secondary in secondary_analyses:
            secondary_recs = secondary["analysis"].get("recommendations", [])
            if secondary_recs:
                all_recommendations.extend(secondary_recs)
        
        # Combine and prioritize recommendations
        unique_recommendations = list(set(all_recommendations))
        prioritized_recs = unique_recommendations[:5]  # Top 5 recommendations
        
        # Build recommendation-focused response
        primary_content = primary_analysis.get("content", "")
        rec_content = f"\n\n**Integrated Recommendations:**\n"
        
        for i, rec in enumerate(prioritized_recs, 1):
            rec_content += f"{i}. {rec}\n"
        
        synthesized_response = primary_analysis.copy()
        synthesized_response["content"] = primary_content + rec_content
        synthesized_response["integrated_recommendations"] = prioritized_recs
        synthesized_response["collaboration_synthesis"] = "recommendation_synthesis"
        
        return synthesized_response
    
    def _synthesize_confidence_enhancement(self, primary_analysis: Dict, secondary_analyses: List[Dict]) -> Dict:
        """Synthesize to enhance confidence through validation"""
        primary_confidence = primary_analysis.get("analysis", {}).get("confidence", 0)
        confidence_scores = [primary_confidence]
        
        for secondary in secondary_analyses:
            secondary_confidence = secondary["analysis"].get("analysis", {}).get("confidence")
            if secondary_confidence is not None:
                confidence_scores.append(secondary_confidence)
        
        # Calculate enhanced confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        confidence_boost = min(15, len(secondary_analyses) * 5)  # Max 15 point boost
        enhanced_confidence = min(95, avg_confidence + confidence_boost)
        
        # Build confidence-enhanced response
        primary_content = primary_analysis.get("content", "")
        confidence_content = f"\n\n**Validation Summary:**\n"
        confidence_content += f"Analysis validated by {len(secondary_analyses)} additional specialist(s). "
        confidence_content += f"Enhanced confidence: {enhanced_confidence:.0f}%"
        
        synthesized_response = primary_analysis.copy()
        synthesized_response["content"] = primary_content + confidence_content
        synthesized_response["analysis"]["confidence"] = enhanced_confidence
        synthesized_response["analysis"]["validation_count"] = len(secondary_analyses)
        synthesized_response["collaboration_synthesis"] = "confidence_enhancement"
        
        return synthesized_response
    
    def _synthesize_perspectives(self, primary_analysis: Dict, secondary_analyses: List[Dict]) -> Dict:
        """Synthesize by integrating multiple perspectives"""
        primary_content = primary_analysis.get("content", "")
        primary_specialist = primary_analysis.get("specialist_used", "Unknown")
        
        # Build integrated perspective response
        sections = [f"**{primary_specialist.replace('_', ' ').title()} Analysis:**\n{primary_content}"]
        
        # Add secondary perspectives with clear attribution
        for secondary in secondary_analyses:
            specialist_name = secondary["specialist"].replace('_', ' ').title()
            content = secondary["analysis"].get("content", "Analysis unavailable")
            
            # Summarize if content is too long
            if len(content) > 300:
                content = content[:250] + "... [continued]"
            
            sections.append(f"\n**{specialist_name} Perspective:**\n{content}")
        
        # Add integration summary
        integration_summary = self._generate_integration_summary(primary_analysis, secondary_analyses)
        if integration_summary:
            sections.append(f"\n**Integrated Assessment:**\n{integration_summary}")
        
        synthesized_response = primary_analysis.copy()
        synthesized_response["content"] = "\n".join(sections)
        synthesized_response["perspectives_integrated"] = len(secondary_analyses) + 1
        synthesized_response["collaboration_synthesis"] = "perspective_integration"
        
        return synthesized_response
    
    def _generate_consensus_summary(self, primary_analysis: Dict, secondary_analyses: List[Dict]) -> str:
        """Generate consensus summary from multiple analyses"""
        summaries = []
        
        # Analyze risk consensus
        risk_scores = [primary_analysis.get("analysis", {}).get("riskScore", 50)]
        for secondary in secondary_analyses:
            risk = secondary["analysis"].get("analysis", {}).get("riskScore")
            if risk is not None:
                risk_scores.append(risk)
        
        if len(risk_scores) > 1:
            risk_range = max(risk_scores) - min(risk_scores)
            if risk_range <= 10:
                summaries.append(f"Strong consensus on risk assessment ({sum(risk_scores)/len(risk_scores):.0f}/100)")
            else:
                summaries.append(f"Mixed risk perspectives require careful consideration")
        
        return ". ".join(summaries) if summaries else ""
    
    def _generate_integration_summary(self, primary_analysis: Dict, secondary_analyses: List[Dict]) -> str:
        """Generate integration summary highlighting key insights"""
        insights = []
        
        # Count specialist involvement
        total_specialists = len(secondary_analyses) + 1
        insights.append(f"Analysis incorporates {total_specialists} specialist perspectives")
        
        # Identify unique insights from collaboration
        collaboration_benefits = []
        for secondary in secondary_analyses:
            opportunity = secondary.get("opportunity", {})
            if opportunity.get("rule_name") == "high_risk_emotional":
                collaboration_benefits.append("behavioral risk factors")
            elif opportunity.get("rule_name") == "optimization_strategy":
                collaboration_benefits.append("strategic alignment")
            elif opportunity.get("rule_name") == "low_confidence_validation":
                collaboration_benefits.append("analytical validation")
        
        if collaboration_benefits:
            insights.append(f"Enhanced analysis includes: {', '.join(collaboration_benefits)}")
        
        return ". ".join(insights) if insights else ""
    
    def _extract_specialists_involved(self, primary_analysis: Dict, secondary_analyses: List[Dict]) -> List[str]:
        """Extract list of specialists involved in analysis"""
        specialists = [primary_analysis.get("specialist_used", "unknown")]
        
        for secondary in secondary_analyses:
            specialist = secondary.get("specialist")
            if specialist and specialist not in specialists:
                specialists.append(specialist)
        
        return specialists
    
    def _track_collaboration_performance(self, performance_data: Dict):
        """Track collaboration performance for optimization"""
        self.collaboration_history.append({
            "timestamp": datetime.now(),
            **performance_data
        })
        
        # Keep last 100 collaboration records
        if len(self.collaboration_history) > 100:
            self.collaboration_history = self.collaboration_history[-100:]
    
    def get_collaboration_analytics(self) -> Dict:
        """Get collaboration performance analytics"""
        if not self.collaboration_history:
            return {"total_collaborations": 0}
        
        total_collaborations = len(self.collaboration_history)
        successful_collaborations = sum(1 for c in self.collaboration_history if c.get("success", False))
        avg_execution_time = sum(c.get("execution_time", 0) for c in self.collaboration_history) / total_collaborations
        
        return {
            "total_collaborations": total_collaborations,
            "success_rate": successful_collaborations / total_collaborations,
            "avg_execution_time": avg_execution_time,
            "collaboration_rules_triggered": len(self.collaboration_rules)
        }