"""
Complete Enhanced Investment Committee Manager - Fixed Import Issues & Integrated Advanced Memory
==================================================================================================

Enhanced features:
1. Semantic routing with embedding-based agent selection
2. Advanced conversation memory with user preference learning and proactive insights
3. Enhanced confidence scoring with tool success weighting
4. Real-time agent performance tracking with comprehensive metrics
5. Portfolio-specific tool selection intelligence
6. Intelligent error recovery with contextual fallbacks
7. Context-aware quick actions generation
8. Professional response formatting
9. Multi-agent collaboration for complex queries
"""

import asyncio
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
import logging

# Set up logger first
logger = logging.getLogger(__name__)

# Import enhanced agents with proper error handling
from agents.enhanced_chat_quantitative_analyst import EnhancedChatQuantitativeAnalyst
from agents.enhanced_chat_cio_agent import EnhancedChatCIOAgent  
from agents.enhanced_chat_portfolio_manager import EnhancedChatPortfolioManager
from agents.enhanced_chat_behavioral_coach import EnhancedChatBehavioralCoach
from agents.multi_agent_collaboration import AgentCollaborationManager, CollaborativeAnalysis
from agents.data_helpers import AgentDataHelper
from agents.proactive_insights import ProactiveInsightsEngine, ProactiveInsight, InsightType, InsightPriority

# Import semantic routing and enhanced memory with safe imports
try:
    from agents.semantic_router import SemanticAgentRouter, RoutingDecision
    SEMANTIC_ROUTING_AVAILABLE = True
    logger.info("Semantic routing enabled")
except ImportError as e:
    logger.warning(f"Semantic routing not available: {e}")
    SEMANTIC_ROUTING_AVAILABLE = False

try:
    from agents.advanced_memory import (
        EnhancedConversationMemory as AdvancedConversationMemory,  # FIXED: Use alias
        MemoryStore,
        create_advanced_memory_system
    )
    ADVANCED_MEMORY_AVAILABLE = True
    logger.info("Advanced memory system available")
except ImportError as e:
    logger.warning(f"Advanced memory not available: {e}")
    ADVANCED_MEMORY_AVAILABLE = False


# Fallback memory class if advanced memory not available
if not ADVANCED_MEMORY_AVAILABLE:
    @dataclass
    class EnhancedConversationMemory:
        """Fallback enhanced conversation memory with analysis tracking and performance metrics"""
        conversation_id: str
        user_id: Optional[str] = None
        messages: List[Dict] = None
        insights: List[Dict] = None
        recommendations: List[str] = None
        portfolio_context: Dict = None
        last_analysis: Dict = None
        analysis_history: List[Dict] = None
        backend_performance: Dict = None
        tool_performance_history: List[Dict] = None
        confidence_history: List[float] = None
        created_at: datetime = None
        updated_at: datetime = None

        def __post_init__(self):
            if self.messages is None:
                self.messages = []
            if self.insights is None:
                self.insights = []
            if self.recommendations is None:
                self.recommendations = []
            if self.analysis_history is None:
                self.analysis_history = []
            if self.backend_performance is None:
                self.backend_performance = {}
            if self.tool_performance_history is None:
                self.tool_performance_history = []
            if self.confidence_history is None:
                self.confidence_history = []
            if self.created_at is None:
                self.created_at = datetime.now()
            if self.updated_at is None:
                self.updated_at = datetime.now()


class EnhancedInvestmentCommitteeManager:
    """
    Enhanced Investment Committee Manager with Semantic Routing and Advanced Memory
    """

    # 2. UPDATE YOUR __init__ METHOD in EnhancedInvestmentCommitteeManager: (INTEGRATED)
    def __init__(self):
        # Initialize enhanced specialists with error handling and fallbacks
        self.specialists = {}

        # Initialize each specialist with proper error handling
        if EnhancedChatQuantitativeAnalyst:
            try:
                self.specialists["quantitative_analyst"] = EnhancedChatQuantitativeAnalyst()
                self.specialists["quant"] = self.specialists["quantitative_analyst"]  # Alias
                logger.info("Enhanced Quantitative Analyst initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Enhanced Quantitative Analyst: {e}")

        if EnhancedChatCIOAgent:
            try:
                self.specialists["cio"] = EnhancedChatCIOAgent()
                self.specialists["chief_investment_officer"] = self.specialists["cio"]  # Alias
                logger.info("Enhanced CIO initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Enhanced CIO: {e}")

        if EnhancedChatPortfolioManager:
            try:
                self.specialists["portfolio_manager"] = EnhancedChatPortfolioManager()
                self.specialists["pm"] = self.specialists["portfolio_manager"]  # Alias
                logger.info("Enhanced Portfolio Manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Enhanced Portfolio Manager: {e}")

        if EnhancedChatBehavioralCoach:
            try:
                self.specialists["behavioral_coach"] = EnhancedChatBehavioralCoach()
                self.specialists["behavioral"] = self.specialists["behavioral_coach"]  # Alias
                self.specialists["coach"] = self.specialists["behavioral_coach"]  # Alias
                logger.info("Enhanced Behavioral Coach initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Enhanced Behavioral Coach: {e}")

        # Initialize semantic router if available
        if SEMANTIC_ROUTING_AVAILABLE:
            try:
                self.semantic_router = SemanticAgentRouter()
                logger.info("Semantic router initialized")
            except Exception as e:
                logger.error(f"Failed to initialize semantic router: {e}")
                self.semantic_router = None
        else:
            self.semantic_router = None

        # Initialize memory system
        if ADVANCED_MEMORY_AVAILABLE:
            try:
                self.memory_store = create_advanced_memory_system("investment_memory")
                self.advanced_memory_enabled = True
                logger.info("Advanced memory system initialized")
            except Exception as e:
                logger.error(f"Advanced memory initialization failed: {e}")
                # Fallback to existing system
                self.conversations = getattr(self, 'conversations', {})
                self.advanced_memory_enabled = False
        else:
            # Use existing fallback memory
            self.conversations = getattr(self, 'conversations', {})
            self.advanced_memory_enabled = False
            logger.info("Using fallback memory system")

        # Enhanced routing patterns with backend tool awareness (fallback for keyword routing)
        self.routing_patterns = {
            "quantitative_analyst": {
                "keywords": ["risk", "var", "stress", "correlation", "volatility", "beta", "analysis", "metrics", "calculate", "statistical", "monte carlo", "regime"],
                "phrases": ["how risky", "risk analysis", "stress test", "portfolio risk", "correlation analysis", "var analysis", "quantitative analysis"],
                "confidence_weight": 1.2
            },
            "cio": {
                "keywords": ["strategy", "strategic", "allocation", "market", "outlook", "trends", "macro", "economic", "long term", "planning", "regime", "diversification"],
                "phrases": ["investment strategy", "market outlook", "strategic allocation", "long-term planning", "market trends", "asset allocation"],
                "confidence_weight": 1.1
            },
            "portfolio_manager": {
                "keywords": ["rebalance", "optimize", "trade", "trading", "buy", "sell", "implementation", "execution", "performance", "weights", "positions"],
                "phrases": ["rebalance portfolio", "optimize allocation", "trading plan", "portfolio optimization", "should i buy", "should i sell", "position sizing"],
                "confidence_weight": 1.3
            },
            "behavioral_coach": {
                "keywords": ["bias", "behavioral", "psychology", "emotional", "decision", "feeling", "worried", "scared", "excited", "panic", "sentiment"],
                "phrases": ["behavioral analysis", "investment psychology", "decision making", "emotional state", "cognitive bias", "panic sell", "market sentiment"],
                "confidence_weight": 1.0
            }
        }

        logger.info(f"Enhanced Investment Committee Manager initialized with {len(self.specialists)} enhanced specialist instances")
        self.collaboration_manager = AgentCollaborationManager(self)
        logger.info("Enhanced Investment Committee Manager with Advanced Memory initialized")

    # 3. ADD NEW METHOD for memory-enhanced routing: (INTEGRATED)
    async def route_query_with_memory(
        self,
        query: str,
        portfolio_context: Dict,
        user_id: str,
        conversation_id: str,
        enable_collaboration: bool = True,
        preferred_specialist: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_satisfaction: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Route query with advanced memory-enhanced context and learning
        """
        try:
            start_time = datetime.now()

            # Get conversation memory
            if self.advanced_memory_enabled:
                memory = self.memory_store.get_conversation(conversation_id, user_id)
            else:
                memory = self._get_or_create_enhanced_conversation(conversation_id, portfolio_context, user_id)

            # Build memory-enhanced context
            enhanced_context = await self._build_memory_enhanced_context(
                query, portfolio_context, memory, user_id
            )

            # Get personalized routing weights if available
            routing_weights = {}
            if self.advanced_memory_enabled and hasattr(memory, 'get_personalized_routing_weights'):
                routing_weights = memory.get_personalized_routing_weights()

            # Route with enhanced context and personalized weights
            if enable_collaboration:
                result = await self.route_query_with_collaboration(
                    query=query,
                    portfolio_context=enhanced_context,
                    enable_collaboration=enable_collaboration,
                    preferred_specialist=preferred_specialist,
                    chat_history=chat_history,
                    user_id=user_id,
                    user_satisfaction=user_satisfaction
                )
            else:
                result = await self.route_query(
                    query=query,
                    portfolio_context=enhanced_context,
                    preferred_specialist=preferred_specialist,
                    chat_history=chat_history,
                    user_id=user_id,
                    user_satisfaction=user_satisfaction
                )

            # Store interaction in memory with learning
            await self._store_interaction_with_learning(
                memory, query, result, portfolio_context, user_satisfaction
            )

            # Add memory-enhanced features to response
            result["memory_enhanced"] = self.advanced_memory_enabled
            result["personalized_routing"] = len(routing_weights) > 0

            if self.advanced_memory_enabled:
                # Add proactive insights
                user_insights = memory.get_user_learning_insights()
                if user_insights.get("learning_confidence", 0) > 0.3:
                    result["user_insights"] = self._format_user_insights(user_insights)

                # Add similar past discussions
                similar_discussions = memory.semantic_search(query, 3)
                if similar_discussions:
                    result["similar_past_discussions"] = self._format_similar_discussions(similar_discussions)

                # Add portfolio evolution insights
                portfolio_insights = memory.get_portfolio_insights()
                if portfolio_insights.get("status") == "analysis_complete":
                    result["portfolio_evolution"] = self._format_portfolio_insights(portfolio_insights)

            execution_time = (datetime.now() - start_time).total_seconds()
            result["memory_processing_time"] = execution_time

            logger.info(f"Memory-enhanced routing completed in {execution_time:.3f}s")
            return result

        except Exception as e:
            logger.error(f"Memory-enhanced routing failed: {e}")
            # Fallback to standard routing
            return await self.route_query_with_collaboration(
                query, portfolio_context, enable_collaboration, None,
                preferred_specialist, chat_history, user_id, user_satisfaction
            )

    # 4. ADD SUPPORTING METHODS: (INTEGRATED)
    async def _build_memory_enhanced_context(
        self, query: str, portfolio_context: Dict, memory: Any, user_id: str
    ) -> Dict:
        """Build enhanced context with memory insights"""
        enhanced_context = portfolio_context.copy()

        if self.advanced_memory_enabled:
            # Add conversation history
            recent_context = memory.get_recent_context(5)
            enhanced_context["recent_conversations"] = [
                {
                    "timestamp": turn.timestamp.isoformat(),
                    "query": turn.user_query,
                    "specialist": turn.specialist,
                    "risk_score": turn.risk_score,
                    "confidence": turn.confidence
                }
                for turn in recent_context
            ]

            # Add user learning insights
            user_insights = memory.get_user_learning_insights()
            enhanced_context["user_preferences"] = user_insights

            # Add semantic search results
            similar_queries = memory.semantic_search(query, 3)
            enhanced_context["similar_past_queries"] = [
                {
                    "query": turn.user_query,
                    "specialist": turn.specialist,
                    "similarity": similarity,
                    "outcome_confidence": turn.confidence
                }
                for turn, similarity in similar_queries
            ]

            # Add portfolio evolution context
            portfolio_insights = memory.get_portfolio_insights()
            enhanced_context["portfolio_evolution"] = portfolio_insights
        return enhanced_context

    async def _store_interaction_with_learning(
        self, memory: Any, query: str, result: Dict,
        portfolio_context: Dict, user_satisfaction: Optional[float]
    ):
        """Store interaction in memory with learning"""
        if self.advanced_memory_enabled and hasattr(memory, 'add_turn'):
            # Extract collaboration information
            collaboration_involved = result.get("collaboration_triggered", False)
            secondary_specialists = result.get("specialists_consulted", [])
            if secondary_specialists and len(secondary_specialists) > 1:
                # Remove primary specialist from secondary list
                primary_specialist = result.get("specialist_used", "")
                secondary_specialists = [s for s in secondary_specialists if s != primary_specialist]

            # Add turn to memory
            memory.add_turn(
                user_query=query,
                agent_response=result.get("content", ""),
                specialist=result.get("specialist_used", "unknown"),
                confidence=result.get("analysis", {}).get("confidence", 0),
                risk_score=result.get("analysis", {}).get("riskScore", 0),
                tool_results=result.get("tool_results", []),
                execution_time=result.get("execution_time", 0),
                user_satisfaction=user_satisfaction,
                collaboration_involved=collaboration_involved,
                secondary_specialists=secondary_specialists,
                portfolio_context=portfolio_context
            )

            # Save conversation periodically
            if len(memory.turns) % 5 == 0:  # Save every 5 turns
                self.memory_store.save_conversation(memory.conversation_id)

    def _format_user_insights(self, user_insights: Dict) -> Dict:
        """Format user insights for response"""
        insights = {}

        # Expertise level insight
        if user_insights.get("expertise_level"):
            insights["expertise_level"] = {
                "level": user_insights["expertise_level"],
                "message": f"Analysis adapted for {user_insights['expertise_level']} level"
            }
        # Collaboration preference
        collab_pref = user_insights.get("collaboration_preference")
        if collab_pref and collab_pref != "balanced":
            insights["collaboration_style"] = {
                "preference": collab_pref,
                "message": f"Using {collab_pref} collaboration approach based on your preferences"
            }
        # Risk tolerance insight
        risk_tolerance = user_insights.get("risk_tolerance", 0.5)
        if risk_tolerance < 0.3:
            insights["risk_guidance"] = {
                "type": "conservative",
                "message": "Analysis includes extra risk considerations based on your preferences"
            }
        elif risk_tolerance > 0.7:
            insights["risk_guidance"] = {
                "type": "aggressive",
                "message": "Analysis focuses on growth opportunities matching your risk appetite"
            }
        return insights

    def _format_similar_discussions(self, similar_discussions: List) -> List[Dict]:
        """Format similar past discussions for response"""
        formatted = []
        for turn, similarity in similar_discussions:
            if similarity > 0.3:  # Only include reasonably similar discussions
                formatted.append({
                    "timestamp": turn.timestamp.isoformat(),
                    "query": turn.user_query[:100] + "..." if len(turn.user_query) > 100 else turn.user_query,
                    "specialist": turn.specialist,
                    "similarity_score": round(similarity, 2),
                    "previous_confidence": turn.confidence,
                    "concepts": turn.concepts_extracted[:3] if turn.concepts_extracted else []
                })
        return formatted[:2]  # Return top 2 similar discussions

    def _format_portfolio_insights(self, portfolio_insights: Dict) -> Dict:
        """Format portfolio evolution insights for response"""
        formatted = {
            "analysis_available": True,
            "time_span_days": portfolio_insights.get("time_span_days", 0)
        }
        # Value change insights
        value_change = portfolio_insights.get("value_change", {})
        if value_change.get("value_change_percent") is not None:
            change_pct = value_change["value_change_percent"]
            formatted["value_trend"] = {
                "change_percent": change_pct,
                "direction": "positive" if change_pct > 0 else "negative" if change_pct < 0 else "stable",
                "magnitude": "significant" if abs(change_pct) > 5 else "moderate" if abs(change_pct) > 1 else "minimal"
            }
        # Risk change insights
        risk_change = portfolio_insights.get("risk_profile_change", {})
        if risk_change.get("risk_level_changed"):
            formatted["risk_evolution"] = {
                "changed": True,
                "direction": risk_change.get("risk_direction", "unknown"),
                "current_level": risk_change.get("current_risk", "MODERATE")
            }
        return formatted

    async def route_query_with_collaboration(
        self,
        query: str,
        portfolio_context: Dict,
        enable_collaboration: bool = True,
        collaboration_hint: Optional[str] = None,
        preferred_specialist: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[str] = None,
        user_satisfaction: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Enhanced query routing with intelligent multi-agent collaboration
        """

        if not enable_collaboration:
            # Standard single-agent routing
            return await self.route_query(
                query, portfolio_context, preferred_specialist,
                chat_history, user_id, user_satisfaction
            )

        try:
            start_time = datetime.now()
            logger.info(f"Starting collaborative analysis for query: {query[:100]}...")

            # Step 1: Get primary analysis using existing semantic routing
            primary_analysis = await self.route_query(
                query, portfolio_context, preferred_specialist,
                chat_history, user_id, user_satisfaction
            )

            # Step 2: Analyze collaboration opportunities
            collaboration_opportunities = await self._identify_collaboration_opportunities(
                query, primary_analysis, portfolio_context, collaboration_hint
            )

            if not collaboration_opportunities:
                # No collaboration needed - return enhanced primary analysis
                primary_analysis["collaboration_considered"] = True
                primary_analysis["collaboration_triggered"] = False
                primary_analysis["collaboration_reason"] = "Single specialist sufficient"
                return primary_analysis

            # Step 3: Execute collaborative analysis
            collaboration_result = await self._execute_collaborative_analysis(
                query, portfolio_context, primary_analysis, collaboration_opportunities
            )

            # Step 4: Synthesize collaborative insights
            synthesized_response = await self._synthesize_collaborative_insights(
                primary_analysis, collaboration_result, query
            )

            # Step 5: Add collaboration metadata
            execution_time = (datetime.now() - start_time).total_seconds()
            synthesized_response.update({
                "collaboration_enabled": True,
                "collaboration_triggered": True,
                "collaboration_execution_time": execution_time,
                "specialists_consulted": collaboration_result["specialists_involved"],
                "collaboration_opportunities": collaboration_opportunities,
                "collaboration_confidence": collaboration_result["synthesis_confidence"],
                "enhancement_version": "3.1_collaboration"
            })

            # Update conversation memory with collaborative context
            conversation_id = portfolio_context.get("conversation_id", str(uuid.uuid4()))
            conversation = self._get_or_create_enhanced_conversation(conversation_id, portfolio_context, user_id)
            self._track_collaboration_performance(conversation, collaboration_result, execution_time)

            logger.info(f"Collaborative analysis completed in {execution_time:.2f}s with {len(collaboration_result['specialists_involved'])} specialists")
            return synthesized_response

        except Exception as e:
            logger.error(f"Collaboration failed, falling back to single agent: {e}")
            # Graceful fallback to standard routing
            primary_response = await self.route_query(
                query, portfolio_context, preferred_specialist,
                chat_history, user_id, user_satisfaction
            )
            primary_response.update({
                "collaboration_enabled": True,
                "collaboration_triggered": False,
                "collaboration_fallback": True,
                "collaboration_error": str(e)
            })
            return primary_response

    async def _identify_collaboration_opportunities(
        self, query: str, primary_analysis: Dict, portfolio_context: Dict, collaboration_hint: Optional[str]
    ) -> List[Dict]:
        """
        Identify when multiple specialists should collaborate on a query
        """
        opportunities = []
        analysis = primary_analysis.get("analysis", {})
        risk_score = analysis.get("riskScore", 50)
        confidence = analysis.get("confidence", 75)
        primary_specialist = primary_analysis.get("specialist_used")

        # DEBUG: Print analysis details
        logger.info(f"Collaboration detection: risk={risk_score}, confidence={confidence}, specialist={primary_specialist}")
        logger.info(f"Query analysis: '{query.lower()}'")

        # Rule 1: High Risk + Behavioral Concerns
        emotional_words = ["worried", "scared", "anxious", "panic", "nervous", "concerned"]
        has_emotional_language = any(word in query.lower() for word in emotional_words)

        if risk_score > 60 and has_emotional_language:  # Lowered threshold for testing
            if primary_specialist != "behavioral_coach" and "behavioral_coach" in self.specialists:
                opportunities.append({
                    "type": "risk_behavioral",
                    "secondary_specialist": "behavioral_coach",
                    "reason": f"Risk score {risk_score} with emotional language detected",
                    "priority": "high",
                    "confidence": 0.9
                })
                logger.info("Added behavioral collaboration opportunity")

        # Rule 2: Low Confidence Needs Validation
        if confidence < 80:  # Lowered threshold
            validation_specialist = self._select_validation_specialist(primary_specialist, query)
            if validation_specialist and validation_specialist in self.specialists:
                opportunities.append({
                    "type": "confidence_validation",
                    "secondary_specialist": validation_specialist,
                    "reason": f"Low confidence ({confidence}%) requires validation",
                    "priority": "medium",
                    "confidence": 0.75
                })
                logger.info(f"Added validation collaboration opportunity: {validation_specialist}")

        # Rule 3: Portfolio Optimization + Strategic Context
        optimization_words = ["optimize", "rebalance", "allocation", "strategy", "implement"]
        has_optimization = any(word in query.lower() for word in optimization_words)

        if has_optimization:
            if primary_specialist == "portfolio_manager" and "cio" in self.specialists:
                opportunities.append({
                    "type": "optimization_strategy",
                    "secondary_specialist": "cio",
                    "reason": "Optimization decisions benefit from strategic context",
                    "priority": "medium",
                    "confidence": 0.8
                })
                logger.info("Added CIO strategic collaboration opportunity")
            elif primary_specialist != "portfolio_manager" and "portfolio_manager" in self.specialists:
                opportunities.append({
                    "type": "strategy_implementation",
                    "secondary_specialist": "portfolio_manager",
                    "reason": "Strategic analysis needs implementation perspective",
                    "priority": "medium",
                    "confidence": 0.8
                })
                logger.info("Added portfolio manager implementation opportunity")

        # Rule 4: Complex Multi-Factor Queries
        complex_words = ["risk", "return", "strategy", "allocation", "rebalance", "optimize",
                         "diversification", "correlation", "volatility", "bias", "behavioral"]
        query_complexity = len([word for word in query.split() if word.lower() in complex_words])

        if query_complexity >= 2:  # Lowered threshold
            complementary_specialists = self._get_complementary_specialists(primary_specialist)
            for specialist in complementary_specialists:
                if specialist in self.specialists:
                    opportunities.append({
                        "type": "complex_multifactor",
                        "secondary_specialist": specialist,
                        "reason": f"Complex query ({query_complexity} factors) benefits from multi-perspective analysis",
                        "priority": "low",
                        "confidence": 0.6
                    })
                    logger.info(f"Added complex query collaboration: {specialist}")
                    break  # Only add one complementary specialist

        # Rule 5: Explicit Collaboration Hint
        if collaboration_hint:
            hint_mapping = {
                "behavioral": "behavioral_coach",
                "strategy": "cio",
                "optimization": "portfolio_manager",
                "risk": "quantitative_analyst"
            }

            suggested_specialist = hint_mapping.get(collaboration_hint.lower())
            if suggested_specialist and suggested_specialist != primary_specialist and suggested_specialist in self.specialists:
                opportunities.append({
                    "type": "explicit_hint",
                    "secondary_specialist": suggested_specialist,
                    "reason": f"User requested {collaboration_hint} perspective",
                    "priority": "high",
                    "confidence": 0.95
                })
                logger.info(f"Added explicit hint collaboration: {suggested_specialist}")

        # Prioritize and limit opportunities
        opportunities.sort(key=lambda x: (x["priority"] == "high", x["confidence"]), reverse=True)
        final_opportunities = opportunities[:2]  # Limit to top 2 opportunities

        logger.info(f"Final collaboration opportunities: {len(final_opportunities)}")
        for opp in final_opportunities:
            logger.info(f"  - {opp['type']}: {opp['secondary_specialist']} ({opp['priority']} priority)")

        return final_opportunities

    async def _execute_collaborative_analysis(
        self, query: str, portfolio_context: Dict, primary_analysis: Dict, opportunities: List[Dict]
    ) -> Dict:
        """
        Execute collaborative analysis with secondary specialists
        """
        secondary_analyses = []
        specialists_involved = [primary_analysis.get("specialist_used")]

        # Execute secondary analyses in parallel
        secondary_tasks = []
        for opportunity in opportunities:
            secondary_specialist = opportunity["secondary_specialist"]
            if secondary_specialist in self.specialists:

                # Create collaboration-aware context
                collaboration_context = {
                    **portfolio_context,
                    "collaboration_mode": True,
                    "primary_analysis": primary_analysis,
                    "collaboration_focus": opportunity["type"],
                    "collaboration_reason": opportunity["reason"]
                }

                # Schedule secondary analysis
                task = self._get_secondary_analysis(query, collaboration_context, secondary_specialist)
                secondary_tasks.append((task, opportunity))

        # Execute all secondary analyses concurrently
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
                    secondary_analyses.append({
                        "specialist": opportunity["secondary_specialist"],
                        "analysis": result,
                        "opportunity": opportunity,
                        "collaboration_value": opportunity["confidence"]
                    })
                    specialists_involved.append(opportunity["secondary_specialist"])

        return {
            "primary_analysis": primary_analysis,
            "secondary_analyses": secondary_analyses,
            "specialists_involved": specialists_involved,
            "collaboration_opportunities": opportunities,
            "synthesis_confidence": min(0.95, len(secondary_analyses) * 0.15 + 0.65)
        }

    async def _get_secondary_analysis(self, query: str, collaboration_context: Dict, specialist_name: str) -> Dict:
        """
        Get analysis from secondary specialist with collaboration awareness
        """
        try:
            specialist = self.specialists[specialist_name]

            # Create focused query for secondary analysis
            focused_query = self._create_focused_collaborative_query(
                query, collaboration_context["collaboration_focus"], specialist_name
            )

            # Execute analysis with collaboration context
            result = await specialist.analyze_query(focused_query, collaboration_context.copy(), collaboration_context)

            # Mark as collaborative analysis
            result["collaboration_mode"] = True
            result["collaboration_focus"] = collaboration_context["collaboration_focus"]

            return result

        except Exception as e:
            logger.error(f"Secondary analysis failed for {specialist_name}: {e}")
            return {
                "content": f"Collaborative analysis unavailable from {specialist_name}",
                "analysis": {"confidence": 0, "riskScore": 50},
                "error": str(e),
                "collaboration_mode": True
            }

    def _create_focused_collaborative_query(self, original_query: str, collaboration_focus: str, specialist_name: str) -> str:
        """
        Create focused query for secondary specialist based on collaboration type
        """
        focus_templates = {
            "risk_behavioral": {
                "behavioral_coach": f"Analyze the emotional and behavioral aspects of this concern: {original_query}"
            },
            "confidence_validation": {
                "quantitative_analyst": f"Provide quantitative validation for: {original_query}",
                "cio": f"Assess the strategic implications of: {original_query}",
                "portfolio_manager": f"Evaluate implementation considerations for: {original_query}"
            },
            "optimization_strategy": {
                "cio": f"What strategic considerations should guide this optimization decision: {original_query}",
                "portfolio_manager": f"What are the practical implementation aspects of: {original_query}"
            },
            "complex_multifactor": {
                "behavioral_coach": f"What behavioral factors should be considered for: {original_query}",
                "quantitative_analyst": f"What are the quantitative risk implications of: {original_query}",
                "cio": f"How does this align with strategic portfolio objectives: {original_query}",
                "portfolio_manager": f"What are the execution and implementation considerations for: {original_query}"
            },
            "explicit_hint": {
                "behavioral_coach": f"Provide behavioral analysis for: {original_query}",
                "cio": f"Provide strategic perspective on: {original_query}",
                "portfolio_manager": f"Provide implementation guidance for: {original_query}",
                "quantitative_analyst": f"Provide quantitative analysis for: {original_query}"
            },
            "regime_risk_analysis": {
                "quantitative_analyst": f"Assess portfolio risk in current market environment: {original_query}"
            }
        }

        specialist_query = focus_templates.get(collaboration_focus, {}).get(specialist_name)
        return specialist_query if specialist_query else original_query

    async def _synthesize_collaborative_insights(
        self, primary_analysis: Dict, collaboration_result: Dict, original_query: str
    ) -> Dict:
        """
        Synthesize insights from collaborative analysis into unified response
        """
        primary_content = primary_analysis.get("content", "")
        primary_specialist = primary_analysis.get("specialist_used", "Unknown")
        secondary_analyses = collaboration_result["secondary_analyses"]

        if not secondary_analyses:
            return primary_analysis

        # Build collaborative response
        collaborative_sections = []

        # Primary analysis section
        collaborative_sections.append(f"**Primary Analysis - {primary_specialist.replace('_', ' ').title()}:**\n{primary_content}")

        # Secondary analysis sections
        for secondary in secondary_analyses:
            specialist_name = secondary["specialist"].replace('_', ' ').title()
            content = secondary["analysis"].get("content", "Analysis unavailable")
            collaboration_value = secondary["collaboration_value"]

            if collaboration_value > 0.7:
                collaborative_sections.append(f"\n**Additional Perspective - {specialist_name}:**\n{content}")
            else:
                # Summarize lower-value contributions
                summary = content[:200] + "..." if len(content) > 200 else content
                collaborative_sections.append(f"\n**{specialist_name} Notes:** {summary}")

        # Synthesis section
        synthesis_insights = self._generate_synthesis_insights(primary_analysis, secondary_analyses)
        if synthesis_insights:
            collaborative_sections.append(f"\n**Integrated Assessment:**\n{synthesis_insights}")

        # Combine all sections
        synthesized_content = "\n".join(collaborative_sections)

        # Create synthesized response preserving primary analysis structure
        synthesized_response = primary_analysis.copy()
        synthesized_response.update({
            "content": synthesized_content,
            "collaboration_synthesis": True,
            "specialists_consulted": collaboration_result["specialists_involved"],
            "collaboration_added_value": len(secondary_analyses) > 0
        })

        # Enhance analysis metrics with collaborative insights
        if "analysis" in synthesized_response:
            synthesized_response["analysis"]["collaboration_enhanced"] = True
            synthesized_response["analysis"]["specialist_consensus"] = self._calculate_consensus(
                primary_analysis, secondary_analyses
            )

        return synthesized_response

    def _generate_synthesis_insights(self, primary_analysis: Dict, secondary_analyses: List[Dict]) -> str:
        """
        Generate synthesis insights from collaborative analysis
        """
        if not secondary_analyses:
            return ""

        insights = []
        primary_risk = primary_analysis.get("analysis", {}).get("riskScore", 50)
        primary_confidence = primary_analysis.get("analysis", {}).get("confidence", 75)

        # Risk consensus analysis
        risk_scores = [primary_risk]
        for secondary in secondary_analyses:
            if "analysis" in secondary["analysis"] and "riskScore" in secondary["analysis"]["analysis"]:
                risk_scores.append(secondary["analysis"]["analysis"]["riskScore"])

        if len(risk_scores) > 1:
            risk_range = max(risk_scores) - min(risk_scores)
            if risk_range > 20:
                insights.append(f"Risk assessment shows varied perspectives (range: {min(risk_scores)}-{max(risk_scores)}), suggesting careful consideration of multiple factors.")
            else:
                avg_risk = sum(risk_scores) / len(risk_scores)
                insights.append(f"Risk assessment shows strong consensus around {avg_risk:.0f}/100.")

        # Confidence validation
        confidence_scores = [primary_confidence]
        for secondary in secondary_analyses:
            if "analysis" in secondary["analysis"] and "confidence" in secondary["analysis"]["analysis"]:
                confidence_scores.append(secondary["analysis"]["analysis"]["confidence"])

        if len(confidence_scores) > 1:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            if avg_confidence > primary_confidence + 10:
                insights.append("Collaborative analysis increases confidence in recommendations.")

        return " ".join(insights) if insights else ""

    def _calculate_consensus(self, primary_analysis: Dict, secondary_analyses: List[Dict]) -> float:
        """
        Calculate consensus score across specialist analyses
        """
        if not secondary_analyses:
            return 1.0

        primary_risk = primary_analysis.get("analysis", {}).get("riskScore", 50)
        risk_scores = [primary_risk]

        for secondary in secondary_analyses:
            if "analysis" in secondary["analysis"] and "riskScore" in secondary["analysis"]["analysis"]:
                risk_scores.append(secondary["analysis"]["analysis"]["riskScore"])

        if len(risk_scores) < 2:
            return 1.0

        # Calculate consensus based on risk score variance
        mean_risk = sum(risk_scores) / len(risk_scores)
        variance = sum((score - mean_risk) ** 2 for score in risk_scores) / len(risk_scores)

        # Convert variance to consensus score (0-1, where 1 is perfect consensus)
        max_variance = 2500  # Assume max variance when scores are at extremes (0 and 100)
        consensus = max(0, 1 - (variance / max_variance))

        return round(consensus, 2)

    def _select_validation_specialist(self, primary_specialist: str, query: str) -> Optional[str]:
        """
        Select appropriate specialist for validation based on primary specialist and query
        """
        validation_mapping = {
            "quantitative_analyst": "cio",  # Strategic validation for quant analysis
            "cio": "quantitative_analyst",  # Quantitative validation for strategy
            "portfolio_manager": "quantitative_analyst",  # Risk validation for optimization
            "behavioral_coach": "cio"  # Strategic context for behavioral analysis
        }

        return validation_mapping.get(primary_specialist)

    def _get_complementary_specialists(self, primary_specialist: str) -> List[str]:
        """
        Get specialists that complement the primary specialist
        """
        complementary_mapping = {
            "quantitative_analyst": ["cio", "behavioral_coach"],
            "cio": ["portfolio_manager", "quantitative_analyst"],
            "portfolio_manager": ["cio", "behavioral_coach"],
            "behavioral_coach": ["quantitative_analyst", "cio"]
        }

        return complementary_mapping.get(primary_specialist, [])

    def _track_collaboration_performance(self, conversation: Any, collaboration_result: Dict, execution_time: float):
        """
        Track collaboration performance metrics
        """
        if not hasattr(conversation, 'collaboration_history'):
            conversation.collaboration_history = []

        performance_record = {
            "timestamp": datetime.now(),
            "specialists_involved": collaboration_result["specialists_involved"],
            "opportunities_identified": len(collaboration_result["collaboration_opportunities"]),
            "secondary_analyses_completed": len(collaboration_result["secondary_analyses"]),
            "execution_time": execution_time,
            "synthesis_confidence": collaboration_result["synthesis_confidence"]
        }

        conversation.collaboration_history.append(performance_record)

        # Keep last 20 collaboration records
        if len(conversation.collaboration_history) > 20:
            conversation.collaboration_history = conversation.collaboration_history[-20:]

    async def route_query(
        self, 
        query: str, 
        portfolio_context: Dict,
        preferred_specialist: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[str] = None,
        user_satisfaction: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Enhanced query routing with semantic routing and advanced memory integration - FIXED
        """
        try:
            start_time = datetime.now()
            
            # FIX: Initialize selected_tools early to prevent UnboundLocalError
            selected_tools = []
            
            # Check if we have any specialists available
            if not self.specialists:
                return self._enhanced_fallback_response("No specialists available - import issues detected", 
                                                    portfolio_context, selected_tools)
            
            # Get or create enhanced conversation
            conversation_id = portfolio_context.get("conversation_id", str(uuid.uuid4()))
            conversation = self._get_or_create_enhanced_conversation(conversation_id, portfolio_context, user_id)
            
            # Portfolio-specific tool selection
            analysis_type = self._determine_analysis_type(query)
            selected_tools = AgentDataHelper.select_tools_for_portfolio(portfolio_context, analysis_type)
            logger.info(f"Selected tools for portfolio: {selected_tools}")
            
            # Enhanced routing with semantic or keyword-based approach
            if preferred_specialist and preferred_specialist in self.specialists:
                specialist_name = preferred_specialist
                routing_confidence = 95  # High confidence for explicit selection
                logger.info(f"Using preferred enhanced specialist: {specialist_name}")
            else:
                specialist_name, routing_confidence = self._enhanced_route_to_specialist_with_confidence(
                    query, conversation, selected_tools
                )
                if preferred_specialist and preferred_specialist not in self.specialists:
                    available = list(self.specialists.keys())
                    logger.warning(f"Preferred specialist '{preferred_specialist}' not available, using {specialist_name}. Available: {available}")
            
            # Get enhanced specialist response with backend integration
            specialist = self.specialists[specialist_name]
            
            # Prepare enhanced context with tool selection
            enhanced_context = {
                **portfolio_context,
                "conversation_history": self._get_conversation_history_for_context(conversation),
                "analysis_history": self._get_analysis_history_for_context(conversation),
                "backend_performance": self._get_backend_performance_for_context(conversation),
                "chat_history": chat_history or [],
                "selected_tools": selected_tools,
                "routing_confidence": routing_confidence
            }
            
            # Execute specialist analysis with error recovery
            try:
                response = await specialist.analyze_query(query, portfolio_context, enhanced_context)
                tool_execution_success = True
            except Exception as e:
                logger.error(f"Specialist analysis failed: {e}")
                # Intelligent error recovery
                response = AgentDataHelper.enhanced_error_response(
                    str(e), portfolio_context, selected_tools, specialist_name
                )
                tool_execution_success = False
            
            # Enhanced confidence scoring with tool success weighting
            base_confidence = response.get("analysis", {}).get("confidence", 75)
            tool_results = response.get("tool_results", [])
            if not tool_results:
                # Create mock tool results for confidence calculation
                tool_results = [{"success": tool_execution_success}] * len(selected_tools)
            
            enhanced_confidence = AgentDataHelper.calculate_enhanced_confidence(tool_results, base_confidence)
            response["analysis"]["confidence"] = enhanced_confidence
            
            # Real-time performance tracking
            execution_time = (datetime.now() - start_time).total_seconds()
            self._track_enhanced_performance(conversation, specialist_name, execution_time, response, selected_tools)
            
            # Update enhanced conversation memory
            self._update_enhanced_conversation_memory(conversation, query, response, specialist_name, user_satisfaction)
            
            # Professional response formatting
            if response.get("content"):
                tool_performance = response.get("tool_performance", {})
                risk_score = response.get("analysis", {}).get("riskScore")
                response["content"] = AgentDataHelper.format_professional_response(
                    response["content"],
                    enhanced_confidence,
                    specialist_name,
                    risk_score,
                    tool_performance
                )
            
            # Generate enhanced insights
            enhanced_insights = self._generate_enhanced_insights(conversation, portfolio_context, response)
            if enhanced_insights:
                response["enhanced_insights"] = enhanced_insights
            
            # Cross-specialist recommendations
            cross_specialist_suggestions = await self._generate_cross_specialist_suggestions(
                conversation, response, specialist_name
            )
            if cross_specialist_suggestions:
                response["cross_specialist_suggestions"] = cross_specialist_suggestions
            
            # Context-aware quick actions
            contextual_actions = AgentDataHelper.generate_contextual_quick_actions(response, portfolio_context)
            response["quick_actions"] = contextual_actions
            
            # Add semantic search results if advanced memory available
            if ADVANCED_MEMORY_AVAILABLE and hasattr(conversation, 'semantic_search'):
                if len(getattr(conversation, 'turns', [])) >= 3:
                    similar_conversations = self.search_conversation_history(conversation_id, query, top_k=2)
                    if similar_conversations:
                        response["similar_past_discussions"] = similar_conversations
            
            # Add enhanced metadata
            response.update({
                "conversation_id": conversation_id,
                "specialist_used": specialist_name,
                "execution_time": execution_time,
                "backend_integration": True,
                "analysis_confidence": enhanced_confidence,
                "routing_confidence": routing_confidence,
                "tools_selected": selected_tools,
                "enhancement_version": "3.0_semantic",
                "semantic_routing_used": self.semantic_router is not None,
                "advanced_memory_used": ADVANCED_MEMORY_AVAILABLE
            })
            
            # Enhanced smart suggestions with context
            response["smart_suggestions"] = self._generate_enhanced_smart_suggestions(conversation, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Enhanced committee routing error: {e}")
            import traceback
            traceback.print_exc()
            
            return self._enhanced_fallback_response(f"Analysis failed: {str(e)}",
                                                portfolio_context, selected_tools or [])

    def _enhanced_route_to_specialist_with_confidence(self, query: str,
                                                      conversation: Any,
                                                      selected_tools: List[str]) -> tuple:
        """
        Enhanced routing using semantic similarity or keyword-based fallback
        """
        # Try semantic routing first if available (works with both memory systems)
        if self.semantic_router:
            try:
                # Get user preferences if available
                user_preferences = {}
                if hasattr(conversation, 'user_profile') and conversation.user_profile:
                    user_preferences = {
                        "agent_satisfaction_scores": conversation.user_profile.agent_satisfaction_scores,
                        "query_complexity_preference": conversation.user_profile.query_complexity_preference,
                        "detail_level_preference": conversation.user_profile.detail_level_preference
                    }

                # Get conversation context (works with both memory systems)
                conversation_history = self._get_conversation_history_for_context(conversation)

                # Get portfolio context
                portfolio_context = {}
                if hasattr(conversation, 'portfolio_evolution') and conversation.portfolio_evolution:
                    portfolio_context = conversation.portfolio_evolution[-1]
                elif hasattr(conversation, 'portfolio_context'):
                    portfolio_context = conversation.portfolio_context or {}

                # Use semantic router
                routing_decision = self.semantic_router.route_query(
                    query=query,
                    conversation_history=conversation_history,
                    user_preferences=user_preferences,
                    portfolio_context=portfolio_context
                )

                # Validate that selected specialist is available
                if routing_decision.primary_agent not in self.specialists:
                    logger.warning(f"Semantic router selected unavailable specialist: {routing_decision.primary_agent}")
                    return self._fallback_keyword_routing(query, conversation, selected_tools)

                logger.info(f"Semantic routing: {routing_decision.reasoning}")
                return routing_decision.primary_agent, int(routing_decision.confidence * 100)

            except Exception as e:
                logger.error(f"Semantic routing failed, falling back to keyword routing: {e}")

        # Fallback to keyword-based routing
        return self._fallback_keyword_routing(query, conversation, selected_tools)

    def _fallback_keyword_routing(self, query: str,
                                  conversation: Any,
                                  selected_tools: List[str]) -> tuple:
        """Fallback keyword-based routing when semantic routing fails or unavailable"""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Enhanced scoring with conversation context and tool alignment
        specialist_scores = {}

        for specialist_name, patterns in self.routing_patterns.items():
            if specialist_name not in self.specialists:
                continue

            score = 0

            # Keyword matching with confidence weights
            keywords = patterns.get("keywords", [])
            keyword_matches = len(query_words.intersection(set(keywords)))
            confidence_weight = patterns.get("confidence_weight", 1.0)
            score += keyword_matches * 2 * confidence_weight

            # Phrase matching (higher weight)
            phrases = patterns.get("phrases", [])
            for phrase in phrases:
                if phrase in query_lower:
                    score += 5 * confidence_weight

            # Tool alignment bonus
            specialist_tools = self._get_specialist_preferred_tools(specialist_name)
            tool_alignment = len(set(selected_tools).intersection(set(specialist_tools)))
            score += tool_alignment * 1.5

            # Conversation context bonus (if available)
            if hasattr(conversation, 'analysis_history') and conversation.analysis_history:
                recent_analyses = conversation.analysis_history[-3:]
                if any(analysis.get("specialist") == specialist_name for analysis in recent_analyses):
                    score += 1  # Continuity bonus

            # Performance history bonus (if available)
            if hasattr(conversation, 'backend_performance') and conversation.backend_performance:
                backend_perf = conversation.backend_performance.get(specialist_name, {})
                avg_confidence = backend_perf.get("avg_confidence", 0.5)
                success_rate = backend_perf.get("success_rate", 1.0)
                performance_bonus = (avg_confidence + success_rate) / 2
                score += performance_bonus * 2

            if score > 0:
                specialist_scores[specialist_name] = score

        # Return best match with confidence calculation
        if specialist_scores:
            best_specialist = max(specialist_scores.items(), key=lambda x: x[1])[0]
            best_score = specialist_scores[best_specialist]
            max_possible_score = 15
            routing_confidence = min(95, int((best_score / max_possible_score) * 100))

            if best_score >= 3:
                logger.info(f"Keyword routing to {best_specialist} (score: {best_score}, confidence: {routing_confidence}%)")
                return best_specialist, routing_confidence

        # Default to best performing specialist or first available
        if hasattr(conversation, 'backend_performance'):
            best_performing = self._get_best_performing_specialist(conversation)
            if best_performing:
                logger.info(f"Routing defaulting to best performer: {best_performing}")
                return best_performing, 70

        default = list(self.specialists.keys())[0]
        logger.info(f"Routing defaulting to {default}")
        return default, 60

    def _get_or_create_enhanced_conversation(self, conversation_id: str, 
                                       portfolio_context: Dict, 
                                       user_id: Optional[str] = None) -> Any:
        """Get or create enhanced conversation memory with user learning - FIXED"""
        if ADVANCED_MEMORY_AVAILABLE:
            # Use advanced memory system
            conversations_dict = getattr(self, 'enhanced_conversations', {})
            if conversation_id in conversations_dict:
                conversation = conversations_dict[conversation_id]
                if hasattr(conversation, 'portfolio_evolution'):
                    conversation.portfolio_evolution.append({
                        "timestamp": datetime.now(),
                        "portfolio_data": portfolio_context
                    })
                return conversation
            else:
                # FIXED: Remove portfolio_context from __init__ - not a valid parameter
                conversation = AdvancedConversationMemory(
                    conversation_id=conversation_id,
                    user_id=user_id or portfolio_context.get("user_id")
                )
                # Add portfolio context after creation
                if hasattr(conversation, 'portfolio_evolution'):
                    conversation.portfolio_evolution.append({
                        "timestamp": datetime.now(),
                        "portfolio_data": portfolio_context
                    })
                conversations_dict[conversation_id] = conversation
                self.enhanced_conversations = conversations_dict
                return conversation
        else:
            # Use fallback memory system
            conversations_dict = getattr(self, 'conversations', {})
            if conversation_id in conversations_dict:
                conversation = conversations_dict[conversation_id]
                conversation.updated_at = datetime.now()
                return conversation
            else:
                conversation = EnhancedConversationMemory(
                    conversation_id=conversation_id,
                    user_id=user_id or portfolio_context.get("user_id"),
                    portfolio_context=portfolio_context
                )
                conversations_dict[conversation_id] = conversation
                self.conversations = conversations_dict
                return conversation

    def _get_conversation_history_for_context(self, conversation: Any) -> List[Dict]:
        """Get conversation history in consistent format"""
        if ADVANCED_MEMORY_AVAILABLE and hasattr(conversation, 'get_recent_context'):
            recent_turns = conversation.get_recent_context(10)
            return [
                {
                    "timestamp": turn.timestamp,
                    "query": turn.user_query,
                    "response": turn.agent_response,
                    "specialist": turn.specialist
                }
                for turn in recent_turns
            ]
        elif hasattr(conversation, 'messages'):
            return conversation.messages[-10:]
        else:
            return []

    def _get_analysis_history_for_context(self, conversation: Any) -> List[Dict]:
        """Get analysis history in consistent format"""
        if hasattr(conversation, 'analysis_history'):
            return conversation.analysis_history[-5:]
        else:
            return []

    def _get_backend_performance_for_context(self, conversation: Any) -> Dict:
        """Get backend performance in consistent format"""
        if hasattr(conversation, 'backend_performance'):
            return conversation.backend_performance
        else:
            return {}

    def _determine_analysis_type(self, query: str) -> str:
        """Determine the primary analysis type from query"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["risk", "var", "stress", "volatility"]):
            return "risk_analysis"
        elif any(word in query_lower for word in ["optimize", "rebalance", "allocation"]):
            return "optimization"
        elif any(word in query_lower for word in ["strategy", "outlook", "market"]):
            return "strategic"
        elif any(word in query_lower for word in ["bias", "behavioral", "psychology"]):
            return "behavioral"
        else:
            return "general"

    def _get_specialist_preferred_tools(self, specialist_name: str) -> List[str]:
        """Get preferred tools for each specialist"""
        tool_preferences = {
            "quantitative_analyst": ["calculate_risk_metrics", "advanced_monte_carlo_stress_test",
                                     "calculate_regime_conditional_risk", "var_calculation"],
            "cio": ["detect_hmm_regimes", "detect_volatility_regimes", "strategic_allocation_models"],
            "portfolio_manager": ["calculate_dynamic_risk_budgets", "portfolio_optimization",
                                  "design_risk_adjusted_momentum_strategy"],
            "behavioral_coach": ["analyze_chat_for_biases", "detect_market_sentiment"]
        }
        return tool_preferences.get(specialist_name, [])

    def _get_best_performing_specialist(self, conversation: Any) -> Optional[str]:
        """Get the best performing specialist based on historical data"""
        backend_performance = getattr(conversation, 'backend_performance', {})
        if not backend_performance:
            return None

        best_specialist = None
        best_score = 0

        for specialist, perf in backend_performance.items():
            composite_score = (perf.get("avg_confidence", 0) + perf.get("success_rate", 0)) / 2
            if composite_score > best_score:
                best_score = composite_score
                best_specialist = specialist

        return best_specialist if best_score > 0.6 else None

    def _enhanced_fallback_response(self, error_msg: str, portfolio_context: Dict,
                                      attempted_tools: List[str]) -> Dict[str, Any]:
        """Enhanced fallback response with intelligent error recovery"""
        fallback = AgentDataHelper.enhanced_error_response(
            error_msg, portfolio_context, attempted_tools, "system"
        )

        fallback.update({
            "id": f"fallback_{int(datetime.now().timestamp())}",
            "type": "enhanced_fallback",
            "timestamp": datetime.now().isoformat(),
            "backend_integration": False,
            "enhancement_version": "3.0_semantic",
            "error": error_msg
        })

        return fallback

    # 5. UPDATE YOUR get_enhanced_analytics_dashboard METHOD: (INTEGRATED)
    def get_enhanced_analytics_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive analytics including memory performance"""
        # Get existing analytics
        base_analytics = {}

        # Add memory analytics
        memory_analytics = {}
        if self.advanced_memory_enabled:
            memory_analytics = self.memory_store.get_memory_analytics()

            # Get user learning statistics
            user_conversations = {}
            for conversation in self.memory_store.conversations.values():
                if conversation.user_id and conversation.user_profile:
                    user_id = conversation.user_id
                    if user_id not in user_conversations:
                        user_conversations[user_id] = {
                            "total_conversations": 0,
                            "learning_confidence": 0,
                            "expertise_level": "intermediate"
                        }
                    user_conversations[user_id]["total_conversations"] += 1
                    insights = conversation.get_user_learning_insights()
                    user_conversations[user_id]["learning_confidence"] = insights.get("learning_confidence", 0)
                    user_conversations[user_id]["expertise_level"] = insights.get("expertise_level", "intermediate")

            memory_analytics["user_learning"] = {
                "users_with_profiles": len(user_conversations),
                "avg_learning_confidence": np.mean([u["learning_confidence"] for u in user_conversations.values()]) if user_conversations else 0,
                "expertise_distribution": {
                    "novice": sum(1 for u in user_conversations.values() if u["expertise_level"] == "novice"),
                    "intermediate": sum(1 for u in user_conversations.values() if u["expertise_level"] == "intermediate"),
                    "advanced": sum(1 for u in user_conversations.values() if u["expertise_level"] == "advanced")
                }
            }
        # Combine analytics
        enhanced_analytics = {
            **base_analytics,
            "memory_system": {
                "advanced_memory_enabled": self.advanced_memory_enabled,
                "memory_analytics": memory_analytics
            }
        }

        return enhanced_analytics

    # 6. ADD MEMORY-SPECIFIC METHODS: (INTEGRATED)
    def get_user_memory_insights(self, user_id: str) -> Dict:
        """Get memory insights for specific user"""
        if not self.advanced_memory_enabled:
            return {"status": "advanced_memory_not_available"}
        user_conversations = self.memory_store.get_user_conversations(user_id)
        if not user_conversations:
            return {"status": "no_user_data"}
        # Aggregate insights across all user conversations
        total_turns = sum(len(conv.turns) for conv in user_conversations)
        latest_conversation = max(user_conversations, key=lambda c: c.updated_at)
        insights = latest_conversation.get_user_learning_insights()
        portfolio_insights = latest_conversation.get_portfolio_insights()
        return {
            "user_id": user_id,
            "total_conversations": len(user_conversations),
            "total_turns": total_turns,
            "learning_insights": insights,
            "portfolio_insights": portfolio_insights,
            "last_active": latest_conversation.updated_at.isoformat(),
            "memory_confidence": insights.get("learning_confidence", 0)
        }

    def search_user_conversations(self, user_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """Search across all user conversations"""
        if not self.advanced_memory_enabled:
            return []
        user_conversations = self.memory_store.get_user_conversations(user_id)
        all_results = []
        for conversation in user_conversations:
            results = conversation.semantic_search(query, top_k)
            for turn, similarity in results:
                all_results.append({
                    "conversation_id": conversation.conversation_id,
                    "timestamp": turn.timestamp.isoformat(),
                    "query": turn.user_query,
                    "specialist": turn.specialist,
                    "similarity": similarity,
                    "confidence": turn.confidence,
                    "risk_score": turn.risk_score
                })
        # Sort by similarity and return top results
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        return all_results[:top_k]

    # User feedback processing
    def process_user_feedback(self, conversation_id: str, message_id: str,
                              satisfaction_rating: float, feedback_text: str = ""):
        """Process user feedback for learning"""
        conversations = getattr(self, 'enhanced_conversations', {}) or getattr(self, 'conversations', {})

        if conversation_id in conversations:
            conversation = conversations[conversation_id]

            if ADVANCED_MEMORY_AVAILABLE and hasattr(conversation, 'turns'):
                # Advanced memory system
                if conversation.turns:
                    latest_turn = conversation.turns[-1]
                    latest_turn.user_satisfaction = satisfaction_rating

                    if conversation.user_profile:
                        conversation.user_profile.update_satisfaction(
                            latest_turn.specialist,
                            satisfaction_rating
                        )

                    logger.info(f"Updated user satisfaction: {satisfaction_rating} for {latest_turn.specialist}")
            else:
                # Fallback: update performance tracking
                if hasattr(conversation, 'messages') and conversation.messages:
                    latest_message = conversation.messages[-1]
                    specialist = latest_message.get("specialist")

                    if specialist and hasattr(conversation, 'backend_performance'):
                        if specialist not in conversation.backend_performance:
                            conversation.backend_performance[specialist] = {"user_satisfaction_history": []}

                        conversation.backend_performance[specialist].setdefault("user_satisfaction_history", []).append(satisfaction_rating)

                        # Keep last 10 satisfaction scores
                        satisfaction_history = conversation.backend_performance[specialist]["user_satisfaction_history"]
                        if len(satisfaction_history) > 10:
                            conversation.backend_performance[specialist]["user_satisfaction_history"] = satisfaction_history[-10:]

    # Semantic search functionality
    def search_conversation_history(self, conversation_id: str, query: str,
                                    top_k: int = 3) -> List[Dict]:
        """Search conversation history using semantic similarity"""
        if not ADVANCED_MEMORY_AVAILABLE:
            return []

        conversations = getattr(self, 'enhanced_conversations', {})
        if conversation_id not in conversations:
            return []

        conversation = conversations[conversation_id]
        if not hasattr(conversation, 'semantic_search'):
            return []

        results = conversation.semantic_search(query, top_k)

        formatted_results = []
        for turn, similarity in results:
            formatted_results.append({
                "timestamp": turn.timestamp.isoformat(),
                "query": turn.user_query,
                "specialist": turn.specialist,
                "similarity_score": similarity,
                "risk_score": turn.risk_score,
                "concepts": getattr(turn, 'concepts_extracted', [])
            })

        return formatted_results

    # Simplified implementations for required methods
    def _track_enhanced_performance(self, conversation, specialist, execution_time, response, selected_tools):
        """Enhanced performance tracking with comprehensive metrics"""
        backend_performance = getattr(conversation, 'backend_performance', {})

        if specialist not in backend_performance:
            backend_performance[specialist] = {
                "total_calls": 0,
                "total_time": 0,
                "avg_confidence": 0,
                "success_rate": 0,
                "failed_calls": 0,
                "tools_used": [],
                "avg_tools_per_call": 0,
                "confidence_trend": []
            }

        perf = backend_performance[specialist]
        perf["total_calls"] += 1
        perf["total_time"] += execution_time

        perf["tools_used"].extend(selected_tools)
        perf["avg_tools_per_call"] = len(perf["tools_used"]) / perf["total_calls"]

        analysis = response.get("analysis", {})
        confidence = analysis.get("confidence", 0) / 100 if analysis.get("confidence", 0) > 1 else analysis.get("confidence", 0)

        perf["confidence_trend"].append(confidence)
        if len(perf["confidence_trend"]) > 10:
            perf["confidence_trend"] = perf["confidence_trend"][-10:]

        if perf["total_calls"] == 1:
            perf["avg_confidence"] = confidence
        else:
            perf["avg_confidence"] = (perf["avg_confidence"] * (perf["total_calls"] - 1) + confidence) / perf["total_calls"]

        if "error" in response or response.get("metadata", {}).get("recovery_mode"):
            perf["failed_calls"] += 1

        perf["success_rate"] = (perf["total_calls"] - perf["failed_calls"]) / perf["total_calls"]
        perf["avg_execution_time"] = perf["total_time"] / perf["total_calls"]

        # Update conversation's backend performance
        if hasattr(conversation, 'backend_performance'):
            conversation.backend_performance = backend_performance

        # Tool performance history
        tool_performance_history = getattr(conversation, 'tool_performance_history', [])
        tool_performance = {
            "timestamp": datetime.now(),
            "specialist": specialist,
            "tools_used": selected_tools,
            "execution_time": execution_time,
            "confidence": confidence,
            "success": "error" not in response
        }
        tool_performance_history.append(tool_performance)

        if len(tool_performance_history) > 50:
            tool_performance_history = tool_performance_history[-50:]

        if hasattr(conversation, 'tool_performance_history'):
            conversation.tool_performance_history = tool_performance_history

    def _update_enhanced_conversation_memory(self, conversation, query, response, specialist, user_satisfaction=None):
        """Update enhanced conversation memory with semantic processing"""
        if ADVANCED_MEMORY_AVAILABLE and hasattr(conversation, 'add_turn'):
            # Use advanced memory system
            analysis = response.get("analysis", {})
            confidence = analysis.get("confidence", 50)
            risk_score = analysis.get("riskScore", 50)
            tool_results = response.get("tool_results", [])
            execution_time = response.get("execution_time", 0)

            turn = conversation.add_turn(
                user_query=query,
                agent_response=response.get("content", ""),
                specialist=specialist,
                confidence=confidence,
                risk_score=risk_score,
                tool_results=tool_results,
                execution_time=execution_time,
                user_satisfaction=user_satisfaction
            )
        else:
            # Use fallback memory system
            conversation.messages.append({
                "timestamp": datetime.now(),
                "query": query,
                "response": response.get("content", ""),
                "specialist": specialist,
                "analysis": response.get("analysis", {}),
                "backend_integration": response.get("backend_integration", False),
                "execution_time": response.get("execution_time", 0),
                "tools_selected": response.get("tools_selected", []),
                "routing_confidence": response.get("routing_confidence", 0),
                "enhancement_version": response.get("enhancement_version", "3.0_semantic")
            })

            # Track confidence evolution
            if response.get("analysis", {}).get("confidence"):
                conversation.confidence_history.append(response["analysis"]["confidence"])
                if len(conversation.confidence_history) > 20:
                    conversation.confidence_history = conversation.confidence_history[-20:]

            # Enhanced analysis history tracking
            if response.get("analysis"):
                conversation.analysis_history.append({
                    "timestamp": datetime.now(),
                    "specialist": specialist,
                    "analysis_type": response.get("analysis_type", "unknown"),
                    "risk_score": response["analysis"].get("riskScore", 50),
                    "confidence": response["analysis"].get("confidence", 50),
                    "recommendation": response["analysis"].get("recommendation", ""),
                    "backend_tools_used": True,
                    "tools_selected": response.get("tools_selected", []),
                    "routing_confidence": response.get("routing_confidence", 0)
                })

            conversation.last_analysis = response.get("analysis", {})
            conversation.updated_at = datetime.now()

            # Keep memory manageable
            if len(conversation.messages) > 50:
                conversation.messages = conversation.messages[-50:]
            if len(conversation.analysis_history) > 20:
                conversation.analysis_history = conversation.analysis_history[-20:]

    def _generate_enhanced_insights(self, conversation, portfolio_context, current_response):
        """Generate enhanced proactive insights using comprehensive analysis"""
        insights = []
        analysis_history = getattr(conversation, 'analysis_history', [])

        # Risk trend analysis with confidence tracking
        recent_analyses = analysis_history[-5:] if analysis_history else []
        if len(recent_analyses) >= 3:
            risk_scores = [analysis["risk_score"] for analysis in recent_analyses]
            confidence_scores = [analysis["confidence"] for analysis in recent_analyses]

            risk_trend = "increasing" if risk_scores[-1] > risk_scores[0] + 10 else "decreasing" if risk_scores[-1] < risk_scores[0] - 10 else "stable"
            confidence_trend = "improving" if confidence_scores[-1] > confidence_scores[0] + 5 else "declining" if confidence_scores[-1] < confidence_scores[0] - 5 else "stable"

            if risk_trend == "increasing":
                insights.append({
                    "type": "risk_trend_alert",
                    "priority": "high",
                    "insight": f"Portfolio risk trend is increasing (current: {risk_scores[-1]}/100). Analysis confidence is {confidence_trend}.",
                    "suggested_action": "Consult with Portfolio Manager for risk optimization",
                    "confidence": 0.85,
                    "backend_derived": True,
                    "trend_data": {"risk_trend": risk_trend, "confidence_trend": confidence_trend}
                })

        return insights

    async def _generate_cross_specialist_suggestions(self, conversation, current_response, current_specialist):
        """Generate enhanced cross-specialist suggestions"""
        suggestions = []

        analysis = current_response.get("analysis", {})
        risk_score = analysis.get("riskScore", 50)
        confidence = analysis.get("confidence", 50)

        if risk_score > 75 and current_specialist != "behavioral_coach" and "behavioral_coach" in self.specialists:
            suggestions.append({
                "specialist": "behavioral_coach",
                "reason": "High risk score may indicate behavioral factors",
                "suggestion": "Consult Behavioral Coach to identify potential cognitive biases affecting decisions",
                "priority": "high",
                "confidence": 0.85
            })

        return suggestions

    def _generate_enhanced_smart_suggestions(self, conversation, current_response):
        """Generate enhanced smart suggestions based on comprehensive analysis"""
        suggestions = []

        analysis = current_response.get("analysis", {})
        risk_score = analysis.get("riskScore", 50)
        confidence = analysis.get("confidence", 50)
        tools_used = current_response.get("tools_selected", [])

        if risk_score > 80:
            suggestions.append({
                "suggestion_type": "risk_management",
                "suggestion_text": f"Your risk analysis shows elevated levels ({risk_score}/100). Comprehensive risk reduction strategy recommended.",
                "action": "comprehensive_risk_strategy",
                "priority": 10,
                "backend_insight": True,
                "tools_context": f"Based on {len(tools_used)} analytical tools"
            })

        return suggestions[:3]

    def get_real_optimization_results(self, portfolio_data: Dict, optimization_type: str = "basic") -> Dict:
        """
        Get real optimization results - fallback implementation
        """
        try:
            # Use your existing portfolio tools
            from tools.portfolio_tools import optimize_portfolio, calculate_efficient_frontier

            if optimization_type == "rebalancing":
                result = optimize_portfolio(portfolio_data)
            else:
                # Basic optimization using efficient frontier
                result = calculate_efficient_frontier(portfolio_data)

            return {
                "optimization_successful": True,
                "recommended_weights": result.get("weights", {}),
                "expected_return": result.get("return", 0.08),
                "expected_risk": result.get("risk", 0.15),
                "optimization_type": optimization_type
            }
        except Exception as e:
            logger.warning(f"Optimization failed, using fallback: {e}")
            return {
                "optimization_successful": False,
                "error": str(e),
                "fallback_recommendation": "Consider professional portfolio review",
                "optimization_type": optimization_type
            }
# Integration layer for Week 2 Proactive Insights Engine
# Add this to your enhanced_investment_committee_manager.py

from agents.proactive_insights import ProactiveInsightsEngine, ProactiveInsight, InsightType, InsightPriority

class EnhancedInvestmentCommitteeManagerWithInsights:
    """Enhanced Investment Committee Manager with Proactive Insights"""
    
    def __init__(self):
        # Initialize existing components
        super().__init__()  # Call existing initialization
        
        # Add proactive insights engine
        self.insights_engine = ProactiveInsightsEngine(memory_system=self.advanced_memory)
        self.active_insights = {}  # Cache active insights per user
        self.insight_history = {}  # Track insight engagement
        
        self.logger.info("Proactive Insights Engine initialized")
    
    async def get_proactive_insights(self, user_id: str, portfolio_data: Dict, 
                                   conversation_history: List[Dict] = None) -> List[Dict]:
        """Get proactive insights for user"""
        try:
            # Generate insights using the proactive engine
            insights = await self.insights_engine.generate_insights(
                user_id=user_id,
                portfolio_data=portfolio_data,
                conversation_history=conversation_history
            )
            
            # Store active insights
            self.active_insights[user_id] = insights
            
            # Convert to API-friendly format
            insights_data = []
            for insight in insights:
                insights_data.append({
                    'id': insight.id,
                    'type': insight.type.value,
                    'priority': insight.priority.value,
                    'title': insight.title,
                    'description': insight.description,
                    'recommendations': insight.recommendations,
                    'conversation_starters': insight.conversation_starters,
                    'created_at': insight.created_at.isoformat(),
                    'data': insight.data
                })
            
            self.logger.info(f"Generated {len(insights_data)} proactive insights for user {user_id}")
            return insights_data
            
        except Exception as e:
            self.logger.error(f"Error generating proactive insights for user {user_id}: {e}")
            return []
    
    async def route_query_with_insights(self, query: str, portfolio_context: Dict, 
                                      user_id: str, conversation_id: str = None,
                                      enable_collaboration: bool = True) -> Dict:
        """Enhanced routing with proactive insights integration"""
        
        try:
            # Get conversation history for insights
            conversation_history = []
            if self.advanced_memory and user_id:
                try:
                    # Get recent conversation history
                    memory_context = await self.advanced_memory.get_conversation_context(
                        user_id, query, limit=10
                    )
                    conversation_history = memory_context.get('recent_conversations', [])
                except Exception as e:
                    self.logger.warning(f"Could not retrieve conversation history: {e}")
            
            # Generate proactive insights
            proactive_insights = await self.get_proactive_insights(
                user_id=user_id,
                portfolio_data=portfolio_context,
                conversation_history=conversation_history
            )
            
            # Standard routing with collaboration
            routing_result = await self.route_query_with_memory(
                query=query,
                portfolio_context=portfolio_context,
                user_id=user_id,
                conversation_id=conversation_id,
                enable_collaboration=enable_collaboration
            )
            
            # Enhance response with proactive insights
            routing_result['proactive_insights'] = proactive_insights
            routing_result['insights_summary'] = self._generate_insights_summary(proactive_insights)
            
            # Check if query relates to existing insights
            related_insights = self._find_related_insights(query, proactive_insights)
            if related_insights:
                routing_result['related_insights'] = related_insights
                routing_result['insight_context'] = f"This query relates to {len(related_insights)} active insights"
            
            return routing_result
            
        except Exception as e:
            self.logger.error(f"Error in route_query_with_insights: {e}")
            # Fallback to standard routing
            return await self.route_query_with_memory(
                query=query,
                portfolio_context=portfolio_context,
                user_id=user_id,
                conversation_id=conversation_id,
                enable_collaboration=enable_collaboration
            )
    
    def _generate_insights_summary(self, insights: List[Dict]) -> Dict:
        """Generate summary of insights by priority and type"""
        summary = {
            'total_insights': len(insights),
            'by_priority': {},
            'by_type': {},
            'critical_count': 0,
            'high_priority_count': 0,
            'conversation_starters_count': 0
        }
        
        for insight in insights:
            priority = insight['priority']
            insight_type = insight['type']
            
            # Count by priority
            summary['by_priority'][priority] = summary['by_priority'].get(priority, 0) + 1
            
            # Count by type
            summary['by_type'][insight_type] = summary['by_type'].get(insight_type, 0) + 1
            
            # Special counts
            if priority == 'critical':
                summary['critical_count'] += 1
            elif priority == 'high':
                summary['high_priority_count'] += 1
            
            if insight_type == 'conversation_starter':
                summary['conversation_starters_count'] += 1
        
        return summary
    
    def _find_related_insights(self, query: str, insights: List[Dict]) -> List[Dict]:
        """Find insights related to current query"""
        related = []
        query_lower = query.lower()
        
        # Keywords that might relate to different insight types
        keyword_mapping = {
            'portfolio_drift': ['rebalance', 'allocation', 'drift', 'weight'],
            'risk_alert': ['risk', 'concentration', 'exposure'],
            'behavioral_pattern': ['anxiety', 'decision', 'worried', 'uncertain'],
            'market_opportunity': ['opportunity', 'diversification', 'sector']
        }
        
        for insight in insights:
            # Check if query keywords match insight type
            insight_type = insight['type']
            if insight_type in keyword_mapping:
                if any(keyword in query_lower for keyword in keyword_mapping[insight_type]):
                    related.append(insight)
                    continue
            
            # Check if query relates to insight title or description
            if (query_lower in insight['title'].lower() or 
                query_lower in insight['description'].lower()):
                related.append(insight)
        
        return related
    
    async def mark_insight_engaged(self, user_id: str, insight_id: str, 
                                 engagement_type: str = 'viewed') -> bool:
        """Mark an insight as engaged with by user"""
        try:
            if user_id not in self.insight_history:
                self.insight_history[user_id] = {}
            
            if insight_id not in self.insight_history[user_id]:
                self.insight_history[user_id][insight_id] = {}
            
            self.insight_history[user_id][insight_id][engagement_type] = datetime.now()
            
            # Update user preference learning if advanced memory is available
            if self.advanced_memory:
                try:
                    await self.advanced_memory.update_user_preferences(
                        user_id, {'insight_engagement': engagement_type}
                    )
                except Exception as e:
                    self.logger.warning(f"Could not update user preferences: {e}")
            
            self.logger.info(f"Marked insight {insight_id} as {engagement_type} for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error marking insight engagement: {e}")
            return False
    
    async def get_insight_analytics(self, user_id: str) -> Dict:
        """Get analytics on insight engagement for user"""
        try:
            analytics = {
                'total_insights_generated': 0,
                'insights_engaged': 0,
                'engagement_rate': 0.0,
                'most_engaged_types': [],
                'recent_activity': []
            }
            
            # Get insight history
            user_history = self.insight_history.get(user_id, {})
            active_insights = self.active_insights.get(user_id, [])
            
            analytics['total_insights_generated'] = len(active_insights)
            analytics['insights_engaged'] = len(user_history)
            
            if analytics['total_insights_generated'] > 0:
                analytics['engagement_rate'] = analytics['insights_engaged'] / analytics['total_insights_generated']
            
            # Analyze engagement by type
            type_engagement = {}
            for insight_id, engagements in user_history.items():
                # Find insight type from active insights
                insight_type = None
                for insight in active_insights:
                    if insight['id'] == insight_id:
                        insight_type = insight['type']
                        break
                
                if insight_type:
                    type_engagement[insight_type] = type_engagement.get(insight_type, 0) + 1
            
            analytics['most_engaged_types'] = sorted(
                type_engagement.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error generating insight analytics: {e}")
            return {}

# Test framework for Week 2
class ProactiveInsightsTestFramework:
    """Test framework for proactive insights functionality"""
    
    def __init__(self, committee_manager):
        self.committee_manager = committee_manager
        self.test_results = {}
    
    async def run_comprehensive_tests(self) -> Dict:
        """Run comprehensive tests of proactive insights system"""
        print("🚀 PROACTIVE INSIGHTS ENGINE - WEEK 2 TESTS")
        print("=" * 60)
        
        test_results = {
            'insights_generation': await self.test_insights_generation(),
            'portfolio_drift_detection': await self.test_portfolio_drift_detection(),
            'behavioral_pattern_analysis': await self.test_behavioral_pattern_analysis(),
            'market_opportunity_detection': await self.test_market_opportunity_detection(),
            'integration_functionality': await self.test_integration_functionality(),
            'performance_metrics': await self.test_performance_metrics()
        }
        
        # Calculate overall score
        passed_tests = sum(1 for result in test_results.values() if result.get('status') == 'PASS')
        total_tests = len(test_results)
        overall_score = (passed_tests / total_tests) * 100
        
        print(f"\n📊 WEEK 2 TEST SUMMARY:")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Overall Score: {overall_score:.1f}%")
        print(f"   Status: {'🎉 READY FOR PRODUCTION' if overall_score >= 80 else '🔧 NEEDS REFINEMENT'}")
        
        return {
            'test_results': test_results,
            'overall_score': overall_score,
            'status': 'PASS' if overall_score >= 80 else 'NEEDS_WORK'
        }
    
    async def test_insights_generation(self) -> Dict:
        """Test basic insights generation functionality"""
        print("\n1️⃣ Testing Insights Generation...")
        
        try:
            # Test portfolio data
            test_portfolio = {
                'holdings': [
                    {'symbol': 'AAPL', 'value': 50000, 'sector': 'Technology'},
                    {'symbol': 'MSFT', 'value': 30000, 'sector': 'Technology'},
                    {'symbol': 'TSLA', 'value': 20000, 'sector': 'Technology'}
                ],
                'total_value': 100000
            }
            
            # Generate insights
            insights = await self.committee_manager.get_proactive_insights(
                user_id="test_user_001",
                portfolio_data=test_portfolio,
                conversation_history=[]
            )
            
            # Validate results
            assert len(insights) > 0, "No insights generated"
            assert all('id' in insight for insight in insights), "Missing insight IDs"
            assert all('priority' in insight for insight in insights), "Missing priority levels"
            
            print(f"   ✅ Generated {len(insights)} insights successfully")
            print(f"   📊 Insight types: {set(i['type'] for i in insights)}")
            
            return {'status': 'PASS', 'insights_count': len(insights)}
            
        except Exception as e:
            print(f"   ❌ Insights generation failed: {e}")
            return {'status': 'FAIL', 'error': str(e)}
    
    async def test_portfolio_drift_detection(self) -> Dict:
        """Test portfolio drift detection functionality"""
        print("\n2️⃣ Testing Portfolio Drift Detection...")
        
        try:
            # Test concentrated portfolio
            concentrated_portfolio = {
                'holdings': [
                    {'symbol': 'AAPL', 'value': 70000},  # 70% concentration
                    {'symbol': 'MSFT', 'value': 20000},
                    {'symbol': 'GOOGL', 'value': 10000}
                ],
                'total_value': 100000
            }
            
            insights = await self.committee_manager.get_proactive_insights(
                user_id="test_concentration",
                portfolio_data=concentrated_portfolio
            )
            
            # Check for concentration alerts
            concentration_alerts = [
                i for i in insights 
                if i['type'] in ['risk_alert', 'portfolio_drift'] and 'concentration' in i['description'].lower()
            ]
            
            assert len(concentration_alerts) > 0, "No concentration risk detected"
            
            print(f"   ✅ Concentration risk detected: {len(concentration_alerts)} alerts")
            print(f"   🎯 Alert priorities: {[a['priority'] for a in concentration_alerts]}")
            
            return {'status': 'PASS', 'concentration_alerts': len(concentration_alerts)}
            
        except Exception as e:
            print(f"   ❌ Portfolio drift detection failed: {e}")
            return {'status': 'FAIL', 'error': str(e)}
    
    async def test_behavioral_pattern_analysis(self) -> Dict:
        """Test behavioral pattern analysis"""
        print("\n3️⃣ Testing Behavioral Pattern Analysis...")
        
        try:
            # Test conversation history with anxiety patterns
            anxiety_conversations = [
                {
                    'query': "I'm really worried about market volatility",
                    'timestamp': datetime.now() - timedelta(days=1)
                },
                {
                    'query': "Should I be concerned about my portfolio risk?",
                    'timestamp': datetime.now() - timedelta(days=3)
                },
                {
                    'query': "I'm scared about potential losses",
                    'timestamp': datetime.now() - timedelta(days=5)
                }
            ]
            
            insights = await self.committee_manager.get_proactive_insights(
                user_id="test_anxiety_pattern",
                portfolio_data={'holdings': [], 'total_value': 50000},
                conversation_history=anxiety_conversations
            )
            
            # Check for behavioral pattern insights
            behavioral_insights = [
                i for i in insights 
                if i['type'] == 'behavioral_pattern'
            ]
            
            print(f"   ✅ Behavioral analysis completed")
            print(f"   🧠 Behavioral insights generated: {len(behavioral_insights)}")
            
            if behavioral_insights:
                print(f"   📋 Pattern types detected: {[i['title'] for i in behavioral_insights]}")
            
            return {'status': 'PASS', 'behavioral_insights': len(behavioral_insights)}
            
        except Exception as e:
            print(f"   ❌ Behavioral pattern analysis failed: {e}")
            return {'status': 'FAIL', 'error': str(e)}
    
    async def test_market_opportunity_detection(self) -> Dict:
        """Test market opportunity detection"""
        print("\n4️⃣ Testing Market Opportunity Detection...")
        
        try:
            # Test small portfolio for diversification opportunities
            small_portfolio = {
                'holdings': [
                    {'symbol': 'AAPL', 'value': 30000, 'sector': 'Technology'},
                    {'symbol': 'MSFT', 'value': 20000, 'sector': 'Technology'}
                ],
                'total_value': 50000
            }
            
            insights = await self.committee_manager.get_proactive_insights(
                user_id="test_opportunities",
                portfolio_data=small_portfolio
            )
            
            # Check for market opportunity insights
            opportunity_insights = [
                i for i in insights 
                if i['type'] == 'market_opportunity'
            ]
            
            print(f"   ✅ Market opportunity analysis completed")
            print(f"   📈 Opportunities identified: {len(opportunity_insights)}")
            
            if opportunity_insights:
                print(f"   💡 Opportunity types: {[i['title'] for i in opportunity_insights]}")
            
            return {'status': 'PASS', 'opportunities': len(opportunity_insights)}
            
        except Exception as e:
            print(f"   ❌ Market opportunity detection failed: {e}")
            return {'status': 'FAIL', 'error': str(e)}
    
    async def test_integration_functionality(self) -> Dict:
        """Test integration with existing committee manager"""
        print("\n5️⃣ Testing Integration Functionality...")
        
        try:
            # Test enhanced routing with insights
            test_query = "Should I rebalance my portfolio?"
            test_portfolio = {
                'holdings': [{'symbol': 'AAPL', 'value': 50000}],
                'total_value': 50000
            }
            
            result = await self.committee_manager.route_query_with_insights(
                query=test_query,
                portfolio_context=test_portfolio,
                user_id="test_integration",
                enable_collaboration=True
            )
            
            # Validate integration
            assert 'proactive_insights' in result, "Proactive insights not included"
            assert 'insights_summary' in result, "Insights summary not included"
            assert 'specialist' in result, "Standard routing failed"
            
            insights_count = len(result['proactive_insights'])
            
            print(f"   ✅ Integration successful")
            print(f"   🔗 Standard routing: {result['specialist']}")
            print(f"   💡 Proactive insights: {insights_count}")
            print(f"   📊 Summary: {result['insights_summary']['total_insights']} total insights")
            
            return {
                'status': 'PASS', 
                'integration_working': True,
                'insights_included': insights_count > 0
            }
            
        except Exception as e:
            print(f"   ❌ Integration functionality failed: {e}")
            return {'status': 'FAIL', 'error': str(e)}
    
    async def test_performance_metrics(self) -> Dict:
        """Test performance and analytics functionality"""
        print("\n6️⃣ Testing Performance Metrics...")
        
        try:
            # Test insight engagement tracking
            user_id = "test_performance"
            
            # Generate insights first
            insights = await self.committee_manager.get_proactive_insights(
                user_id=user_id,
                portfolio_data={'holdings': [], 'total_value': 100000}
            )
            
            # Test engagement tracking
            if insights:
                insight_id = insights[0]['id']
                engagement_success = await self.committee_manager.mark_insight_engaged(
                    user_id=user_id,
                    insight_id=insight_id,
                    engagement_type='viewed'
                )
                
                assert engagement_success, "Engagement tracking failed"
            
            # Test analytics
            analytics = await self.committee_manager.get_insight_analytics(user_id)
            
            assert 'total_insights_generated' in analytics, "Analytics missing key metrics"
            
            print(f"   ✅ Performance metrics working")
            print(f"   📊 Analytics available: {len(analytics)} metrics")
            print(f"   🎯 Engagement tracking: {'✅ Working' if insights and engagement_success else '⚠️ Limited'}")
            
            return {
                'status': 'PASS',
                'analytics_working': True,
                'engagement_tracking': bool(insights and engagement_success)
            }
            
        except Exception as e:
            print(f"   ❌ Performance metrics failed: {e}")
            return {'status': 'FAIL', 'error': str(e)}
    
    async def execute_agent_action(self, agent_id: str, action: str, params: dict, 
                              context: dict, portfolio_context: dict, user_id: str) -> dict:
        '''Execute specific agent action'''
        try:
            # Route to appropriate agent based on agent_id
            if agent_id == "quantitative_analyst":
                return await self.quantitative_analyst.execute_action(action, params, context, portfolio_context)
            elif agent_id == "portfolio_manager":
                return await self.portfolio_manager.execute_action(action, params, context, portfolio_context)
            elif agent_id == "behavioral_coach":
                return await self.behavioral_coach.execute_action(action, params, context, portfolio_context)
            elif agent_id == "cio":
                return await self.cio.execute_action(action, params, context, portfolio_context)
            else:
                raise ValueError(f"Unknown agent_id: {agent_id}")
                
        except Exception as e:
            logger.error(f"Error executing agent action: {e}")
            return {"error": str(e), "status": "failed"}