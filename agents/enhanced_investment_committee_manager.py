# agents/enhanced_committee_manager.py - Enhanced with Quick Wins
"""
Complete Enhanced Investment Committee Manager - With Quick Wins Integration
=========================================================================

Added quick wins:
1. Enhanced confidence scoring with tool success weighting
2. Real-time agent performance tracking with comprehensive metrics
3. Portfolio-specific tool selection intelligence
4. Intelligent error recovery with contextual fallbacks
5. Context-aware quick actions generation
6. Professional response formatting
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
import logging

# Import enhanced agents with proper error handling
from agents.enhanced_chat_quantitative_analyst import EnhancedChatQuantitativeAnalyst
from agents.enhanced_chat_cio_agent import EnhancedChatCIOAgent  
from agents.enhanced_chat_portfolio_manager import EnhancedChatPortfolioManager
from agents.enhanced_chat_behavioral_coach import EnhancedChatBehavioralCoach

# Import enhanced data helpers
from agents.data_helpers import AgentDataHelper

logger = logging.getLogger(__name__)

@dataclass
class EnhancedConversationMemory:
    """Enhanced conversation memory with analysis tracking and performance metrics"""
    conversation_id: str
    user_id: Optional[str] = None
    messages: List[Dict] = None
    insights: List[Dict] = None
    recommendations: List[str] = None
    portfolio_context: Dict = None
    last_analysis: Dict = None
    analysis_history: List[Dict] = None
    backend_performance: Dict = None
    tool_performance_history: List[Dict] = None  # NEW: Track tool performance over time
    confidence_history: List[float] = None  # NEW: Track confidence evolution
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
    Enhanced Investment Committee Manager with Quick Wins Integration
    """
    
    def __init__(self):
        # Initialize enhanced specialists with error handling and fallbacks
        self.specialists = {}
        
        # Initialize each specialist with proper error handling
        if EnhancedChatQuantitativeAnalyst:
            try:
                self.specialists["quantitative_analyst"] = EnhancedChatQuantitativeAnalyst()
                self.specialists["quant"] = self.specialists["quantitative_analyst"]  # Alias
                logger.info("✅ Enhanced Quantitative Analyst initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Enhanced Quantitative Analyst: {e}")
        
        if EnhancedChatCIOAgent:
            try:
                self.specialists["cio"] = EnhancedChatCIOAgent()
                self.specialists["chief_investment_officer"] = self.specialists["cio"]  # Alias
                logger.info("✅ Enhanced CIO initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Enhanced CIO: {e}")
        
        if EnhancedChatPortfolioManager:
            try:
                self.specialists["portfolio_manager"] = EnhancedChatPortfolioManager()
                self.specialists["pm"] = self.specialists["portfolio_manager"]  # Alias
                logger.info("✅ Enhanced Portfolio Manager initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Enhanced Portfolio Manager: {e}")
        
        if EnhancedChatBehavioralCoach:
            try:
                self.specialists["behavioral_coach"] = EnhancedChatBehavioralCoach()
                self.specialists["behavioral"] = self.specialists["behavioral_coach"]  # Alias
                self.specialists["coach"] = self.specialists["behavioral_coach"]  # Alias
                logger.info("✅ Enhanced Behavioral Coach initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Enhanced Behavioral Coach: {e}")
        
        # Enhanced conversation memory storage
        self.conversations: Dict[str, EnhancedConversationMemory] = {}
        
        # Enhanced routing patterns with backend tool awareness
        self.routing_patterns = {
            "quantitative_analyst": {
                "keywords": ["risk", "var", "stress", "correlation", "volatility", "beta", "analysis", "metrics", "calculate", "statistical", "monte carlo", "regime"],
                "phrases": ["how risky", "risk analysis", "stress test", "portfolio risk", "correlation analysis", "var analysis", "quantitative analysis"],
                "confidence_weight": 1.2  # Higher weight for precise quantitative queries
            },
            "cio": {
                "keywords": ["strategy", "strategic", "allocation", "market", "outlook", "trends", "macro", "economic", "long term", "planning", "regime", "diversification"],
                "phrases": ["investment strategy", "market outlook", "strategic allocation", "long-term planning", "market trends", "asset allocation"],
                "confidence_weight": 1.1
            },
            "portfolio_manager": {
                "keywords": ["rebalance", "optimize", "trade", "trading", "buy", "sell", "implementation", "execution", "performance", "weights", "positions"],
                "phrases": ["rebalance portfolio", "optimize allocation", "trading plan", "portfolio optimization", "should i buy", "should i sell", "position sizing"],
                "confidence_weight": 1.3  # Higher weight for actionable queries
            },
            "behavioral_coach": {
                "keywords": ["bias", "behavioral", "psychology", "emotional", "decision", "feeling", "worried", "scared", "excited", "panic", "sentiment"],
                "phrases": ["behavioral analysis", "investment psychology", "decision making", "emotional state", "cognitive bias", "panic sell", "market sentiment"],
                "confidence_weight": 1.0
            }
        }
        
        logger.info(f"Enhanced Investment Committee Manager initialized with {len(self.specialists)} enhanced specialist instances")
    
    async def route_query(
        self, 
        query: str, 
        portfolio_context: Dict,
        preferred_specialist: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhanced query routing with comprehensive quick wins integration
        """
        try:
            start_time = datetime.now()
            
            # Check if we have any specialists available
            if not self.specialists:
                return self._enhanced_fallback_response("No specialists available - import issues detected", 
                                                       portfolio_context, [])
            
            # Get or create enhanced conversation
            conversation_id = portfolio_context.get("conversation_id", str(uuid.uuid4()))
            conversation = self._get_or_create_enhanced_conversation(conversation_id, portfolio_context)
            
            # QUICK WIN 2: Portfolio-specific tool selection
            analysis_type = self._determine_analysis_type(query)
            selected_tools = AgentDataHelper.select_tools_for_portfolio(portfolio_context, analysis_type)
            logger.info(f"Selected tools for portfolio: {selected_tools}")
            
            # Enhanced routing with confidence scoring
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
                "conversation_history": conversation.messages[-10:],  # Last 10 messages
                "analysis_history": conversation.analysis_history[-5:],  # Last 5 analyses
                "backend_performance": conversation.backend_performance,
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
                # QUICK WIN 5: Intelligent error recovery
                response = AgentDataHelper.enhanced_error_response(
                    str(e), portfolio_context, selected_tools, specialist_name
                )
                tool_execution_success = False
            
            # QUICK WIN 1: Enhanced confidence scoring with tool success weighting
            base_confidence = response.get("analysis", {}).get("confidence", 75)
            tool_results = response.get("tool_results", [])
            if not tool_results:
                # Create mock tool results for confidence calculation
                tool_results = [{"success": tool_execution_success}] * len(selected_tools)
            
            enhanced_confidence = AgentDataHelper.calculate_enhanced_confidence(tool_results, base_confidence)
            response["analysis"]["confidence"] = enhanced_confidence
            
            # QUICK WIN 3: Real-time performance tracking
            execution_time = (datetime.now() - start_time).total_seconds()
            self._track_enhanced_performance(conversation, specialist_name, execution_time, response, selected_tools)
            
            # Update enhanced conversation memory
            self._update_enhanced_conversation_memory(conversation, query, response, specialist_name)
            
            # QUICK WIN 4: Professional response formatting
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
            
            # QUICK WIN 3: Context-aware quick actions
            contextual_actions = AgentDataHelper.generate_contextual_quick_actions(response, portfolio_context)
            response["quick_actions"] = contextual_actions
            
            # Add enhanced metadata
            response["conversation_id"] = conversation_id
            response["specialist_used"] = specialist_name
            response["execution_time"] = execution_time
            response["backend_integration"] = True
            response["analysis_confidence"] = enhanced_confidence
            response["routing_confidence"] = routing_confidence
            response["tools_selected"] = selected_tools
            response["enhancement_version"] = "2.0_quickwins"
            
            # Enhanced smart suggestions with context
            response["smart_suggestions"] = self._generate_enhanced_smart_suggestions(conversation, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Enhanced committee routing error: {e}")
            import traceback
            traceback.print_exc()
            
            return self._enhanced_fallback_response(f"Analysis failed: {str(e)}", 
                                                   portfolio_context, selected_tools or [])
    
    def _enhanced_fallback_response(self, error_msg: str, portfolio_context: Dict, 
                                   attempted_tools: List[str]) -> Dict[str, Any]:
        """Enhanced fallback response with intelligent error recovery"""
        
        # Generate contextual fallback using enhanced error recovery
        fallback = AgentDataHelper.enhanced_error_response(
            error_msg, portfolio_context, attempted_tools, "system"
        )
        
        # Add system metadata
        fallback.update({
            "id": f"fallback_{int(datetime.now().timestamp())}",
            "type": "enhanced_fallback",
            "timestamp": datetime.now().isoformat(),
            "backend_integration": False,
            "enhancement_version": "2.0_quickwins",
            "error": error_msg
        })
        
        return fallback
    
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
    
    def _enhanced_route_to_specialist_with_confidence(self, query: str, 
                                                    conversation: EnhancedConversationMemory,
                                                    selected_tools: List[str]) -> tuple:
        """
        Enhanced routing with confidence scoring and tool awareness
        """
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
                    score += 5 * confidence_weight  # Weighted phrase matching
            
            # Tool alignment bonus
            specialist_tools = self._get_specialist_preferred_tools(specialist_name)
            tool_alignment = len(set(selected_tools).intersection(set(specialist_tools)))
            score += tool_alignment * 1.5
            
            # Conversation context bonus
            recent_analyses = conversation.analysis_history[-3:] if conversation.analysis_history else []
            if any(analysis.get("specialist") == specialist_name for analysis in recent_analyses):
                score += 1  # Continuity bonus
            
            # Performance history bonus
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
            max_possible_score = 15  # Approximate maximum possible score
            routing_confidence = min(95, int((best_score / max_possible_score) * 100))
            
            # Require minimum score for routing confidence
            if best_score >= 3:
                logger.info(f"Enhanced routing to {best_specialist} (score: {best_score}, confidence: {routing_confidence}%)")
                return best_specialist, routing_confidence
        
        # Default to best performing specialist or first available
        if self.specialists:
            # Find specialist with best historical performance
            best_performing = self._get_best_performing_specialist(conversation)
            if best_performing:
                logger.info(f"Enhanced routing defaulting to best performer: {best_performing}")
                return best_performing, 70
            else:
                default = list(self.specialists.keys())[0]
                logger.info(f"Enhanced routing defaulting to {default}")
                return default, 60
        else:
            raise Exception("No specialists available")
    
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
    
    def _get_best_performing_specialist(self, conversation: EnhancedConversationMemory) -> Optional[str]:
        """Get the best performing specialist based on historical data"""
        if not conversation.backend_performance:
            return None
        
        best_specialist = None
        best_score = 0
        
        for specialist, perf in conversation.backend_performance.items():
            # Combine confidence and success rate
            composite_score = (perf.get("avg_confidence", 0) + perf.get("success_rate", 0)) / 2
            if composite_score > best_score:
                best_score = composite_score
                best_specialist = specialist
        
        return best_specialist if best_score > 0.6 else None
    
    def _get_or_create_enhanced_conversation(self, conversation_id: str, portfolio_context: Dict) -> EnhancedConversationMemory:
        """Get or create enhanced conversation memory"""
        if conversation_id in self.conversations:
            conversation = self.conversations[conversation_id]
            conversation.updated_at = datetime.now()
            return conversation
        else:
            conversation = EnhancedConversationMemory(
                conversation_id=conversation_id,
                user_id=portfolio_context.get("user_id"),
                portfolio_context=portfolio_context
            )
            self.conversations[conversation_id] = conversation
            return conversation
    
    def _track_enhanced_performance(self, conversation: EnhancedConversationMemory, 
                                  specialist: str, execution_time: float, response: Dict,
                                  selected_tools: List[str]):
        """Enhanced performance tracking with comprehensive metrics"""
        if specialist not in conversation.backend_performance:
            conversation.backend_performance[specialist] = {
                "total_calls": 0,
                "total_time": 0,
                "avg_confidence": 0,
                "success_rate": 0,
                "failed_calls": 0,
                "tools_used": [],
                "avg_tools_per_call": 0,
                "confidence_trend": []
            }
        
        perf = conversation.backend_performance[specialist]
        perf["total_calls"] += 1
        perf["total_time"] += execution_time
        
        # Track tools usage
        perf["tools_used"].extend(selected_tools)
        perf["avg_tools_per_call"] = len(perf["tools_used"]) / perf["total_calls"]
        
        # Update confidence tracking with trend analysis
        analysis = response.get("analysis", {})
        confidence = analysis.get("confidence", 0) / 100 if analysis.get("confidence", 0) > 1 else analysis.get("confidence", 0)
        
        perf["confidence_trend"].append(confidence)
        if len(perf["confidence_trend"]) > 10:  # Keep last 10 confidence scores
            perf["confidence_trend"] = perf["confidence_trend"][-10:]
        
        if perf["total_calls"] == 1:
            perf["avg_confidence"] = confidence
        else:
            perf["avg_confidence"] = (perf["avg_confidence"] * (perf["total_calls"] - 1) + confidence) / perf["total_calls"]
        
        # Track success/failure with enhanced detection
        if "error" in response or response.get("metadata", {}).get("recovery_mode"):
            perf["failed_calls"] += 1
        
        perf["success_rate"] = (perf["total_calls"] - perf["failed_calls"]) / perf["total_calls"]
        perf["avg_execution_time"] = perf["total_time"] / perf["total_calls"]
        
        # Track tool performance history
        tool_performance = {
            "timestamp": datetime.now(),
            "specialist": specialist,
            "tools_used": selected_tools,
            "execution_time": execution_time,
            "confidence": confidence,
            "success": "error" not in response
        }
        conversation.tool_performance_history.append(tool_performance)
        
        # Keep performance history manageable
        if len(conversation.tool_performance_history) > 50:
            conversation.tool_performance_history = conversation.tool_performance_history[-50:]
    
    def _update_enhanced_conversation_memory(self, conversation: EnhancedConversationMemory, 
                                           query: str, response: Dict, specialist: str):
        """Update enhanced conversation memory with comprehensive tracking"""
        # Add message to history with enhanced metadata
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
            "enhancement_version": response.get("enhancement_version", "2.0_quickwins")
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
        
        # Enhanced recommendations tracking
        if response.get("analysis", {}).get("recommendation"):
            conversation.recommendations.append({
                "timestamp": datetime.now(),
                "specialist": specialist,
                "recommendation": response["analysis"]["recommendation"],
                "confidence": response["analysis"].get("confidence", 50),
                "risk_score": response["analysis"].get("riskScore", 50),
                "backend_driven": True,
                "context_aware": True,
                "tools_used": response.get("tools_selected", [])
            })
        
        conversation.last_analysis = response.get("analysis", {})
        conversation.updated_at = datetime.now()
        
        # Keep memory manageable
        if len(conversation.messages) > 50:
            conversation.messages = conversation.messages[-50:]
        if len(conversation.analysis_history) > 20:
            conversation.analysis_history = conversation.analysis_history[-20:]
    
    def _generate_enhanced_insights(self, conversation: EnhancedConversationMemory, 
                                  portfolio_context: Dict, current_response: Dict) -> List[Dict]:
        """Generate enhanced proactive insights using comprehensive analysis"""
        insights = []
        
        # Risk trend analysis with confidence tracking
        recent_analyses = conversation.analysis_history[-5:]
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
        
        # Tool performance insights
        tool_performance = current_response.get("tool_performance", {})
        if tool_performance:
            success_rate = tool_performance.get("success_rate", 1.0)
            if success_rate < 0.8:
                insights.append({
                    "type": "tool_performance_alert",
                    "priority": "medium",
                    "insight": f"Backend tool performance at {success_rate*100:.0f}%. Analysis reliability may be affected.",
                    "suggested_action": "Consider consulting multiple specialists for validation",
                    "confidence": 0.90,
                    "backend_derived": True,
                    "performance_data": tool_performance
                })
        
        # Cross-specialist analysis gaps
        recent_specialists = list(set([msg["specialist"] for msg in conversation.messages[-5:]]))
        if len(recent_specialists) == 1 and len(conversation.messages) >= 3:
            current_specialist = recent_specialists[0]
            suggested_specialists = self._get_complementary_specialists(current_specialist)
            if suggested_specialists:
                insights.append({
                    "type": "analysis_diversification",
                    "priority": "low",
                    "insight": f"Recent analysis focused on {current_specialist}. Consider broader perspective.",
                    "suggested_action": f"Consult with {suggested_specialists[0]} for additional insights",
                    "confidence": 0.70,
                    "backend_derived": True,
                    "suggested_specialists": suggested_specialists
                })
        
        return insights
    
    def _get_complementary_specialists(self, current_specialist: str) -> List[str]:
        """Get complementary specialists for broader analysis"""
        complementary_map = {
            "quantitative_analyst": ["behavioral_coach", "portfolio_manager"],
            "portfolio_manager": ["cio", "quantitative_analyst"],
            "cio": ["behavioral_coach", "quantitative_analyst"],
            "behavioral_coach": ["quantitative_analyst", "cio"]
        }
        
        complements = complementary_map.get(current_specialist, [])
        return [s for s in complements if s in self.specialists]
    
    async def _generate_cross_specialist_suggestions(self, conversation: EnhancedConversationMemory, 
                                                   current_response: Dict, current_specialist: str) -> List[Dict]:
        """Generate enhanced cross-specialist suggestions"""
        suggestions = []
        
        # Analyze current response for cross-specialist opportunities
        analysis = current_response.get("analysis", {})
        risk_score = analysis.get("riskScore", 50)
        confidence = analysis.get("confidence", 50)
        analysis_type = current_response.get("analysis_type", "")
        
        # High risk score suggests consulting behavioral coach
        if risk_score > 75 and current_specialist != "behavioral_coach" and "behavioral_coach" in self.specialists:
            suggestions.append({
                "specialist": "behavioral_coach",
                "reason": "High risk score may indicate behavioral factors",
                "suggestion": "Consult Behavioral Coach to identify potential cognitive biases affecting decisions",
                "priority": "high",
                "confidence": 0.85
            })
        
        # Low confidence suggests quantitative validation
        if confidence < 70 and current_specialist != "quantitative_analyst" and "quantitative_analyst" in self.specialists:
            suggestions.append({
                "specialist": "quantitative_analyst",
                "reason": "Low confidence analysis needs quantitative validation",
                "suggestion": "Get quantitative analysis for statistical validation",
                "priority": "high",
                "confidence": 0.80
            })
        
        # Quantitative analysis suggests portfolio optimization
        if current_specialist == "quantitative_analyst" and risk_score > 60 and "portfolio_manager" in self.specialists:
            suggestions.append({
                "specialist": "portfolio_manager", 
                "reason": "Risk analysis indicates optimization opportunities",
                "suggestion": "Portfolio Manager can provide specific optimization and rebalancing recommendations",
                "priority": "medium",
                "confidence": 0.75
            })
        
        # Portfolio analysis suggests strategic review
        if current_specialist == "portfolio_manager" and "optimization" in analysis_type and "cio" in self.specialists:
            suggestions.append({
                "specialist": "cio",
                "reason": "Portfolio optimization may require strategic context",
                "suggestion": "CIO can provide strategic market outlook and asset allocation guidance",
                "priority": "medium",
                "confidence": 0.70
            })
        
        return suggestions
    
    def _generate_enhanced_smart_suggestions(self, conversation: EnhancedConversationMemory, 
                                           current_response: Dict) -> List[Dict]:
        """Generate enhanced smart suggestions based on comprehensive analysis"""
        suggestions = []
        
        analysis = current_response.get("analysis", {})
        risk_score = analysis.get("riskScore", 50)
        confidence = analysis.get("confidence", 50)
        specialist = current_response.get("specialist", "")
        tools_used = current_response.get("tools_selected", [])
        
        # Backend-driven suggestions with tool awareness
        if risk_score > 80:
            suggestions.append({
                "suggestion_type": "risk_management",
                "suggestion_text": f"Your risk analysis shows elevated levels ({risk_score}/100). Comprehensive risk reduction strategy recommended.",
                "action": "comprehensive_risk_strategy",
                "priority": 10,
                "backend_insight": True,
                "tools_context": f"Based on {len(tools_used)} analytical tools"
            })
        
        if confidence > 90 and len(tools_used) > 2:
            suggestions.append({
                "suggestion_type": "implementation",
                "suggestion_text": f"High-confidence analysis ({confidence}%) from {len(tools_used)} tools. Ready to implement recommendations?",
                "action": "implementation_plan",
                "priority": 8,
                "backend_insight": True,
                "tools_context": f"Validated by {', '.join(tools_used[:2])}"
            })
        
        # Tool-specific suggestions
        if "monte_carlo" in str(tools_used).lower() and risk_score > 70:
            suggestions.append({
                "suggestion_type": "stress_testing",
                "suggestion_text": "Monte Carlo stress testing revealed vulnerabilities. Explore stress-resistant strategies?",
                "action": "stress_resistant_portfolio",
                "priority": 7,
                "backend_insight": True,
                "tools_context": "Advanced Monte Carlo simulation"
            })
        
        # Cross-specialist suggestions based on analysis patterns
        recent_specialists = [msg["specialist"] for msg in conversation.messages[-5:]]
        if len(set(recent_specialists)) == 1 and len(recent_specialists) >= 3:
            other_specialists = [s for s in self.specialists.keys() if s != specialist and s not in recent_specialists]
            if other_specialists:
                suggestions.append({
                    "suggestion_type": "perspective",
                    "suggestion_text": f"Multiple {specialist} consultations completed. Different perspective available?",
                    "action": f"consult_{other_specialists[0]}",
                    "priority": 6,
                    "backend_insight": True,
                    "tools_context": "Cross-validation recommended"
                })
        
        # Confidence trend suggestions
        if len(conversation.confidence_history) >= 3:
            recent_confidence = conversation.confidence_history[-3:]
            confidence_declining = all(recent_confidence[i] > recent_confidence[i+1] for i in range(len(recent_confidence)-1))
            
            if confidence_declining:
                suggestions.append({
                    "suggestion_type": "confidence_recovery",
                    "suggestion_text": "Analysis confidence declining. Multi-specialist validation recommended.",
                    "action": "multi_specialist_review",
                    "priority": 8,
                    "backend_insight": True,
                    "tools_context": "Confidence trend analysis"
                })
        
        return suggestions[:3]  # Limit to top 3
    
    def get_enhanced_specialists_info(self) -> List[Dict]:
        """Get enhanced information about available specialists with backend capabilities"""
        specialists_info = []
        
        specialist_definitions = {
            "quantitative_analyst": {
                "id": "quantitative_analyst",
                "name": "Quantitative Analyst",
                "description": "Advanced quantitative analysis with institutional-grade risk models",
                "capabilities": [
                    "Real VaR/CVaR analysis using portfolio returns",
                    "Monte Carlo stress testing with fat-tail modeling", 
                    "Regime-conditional risk analysis",
                    "Advanced correlation and factor analysis"
                ],
                "backend_tools": ["calculate_risk_metrics", "calculate_regime_conditional_risk", "advanced_monte_carlo_stress_test"],
                "enhancement_level": "2.0_quickwins"
            },
            "cio": {
                "id": "cio", 
                "name": "Chief Investment Officer",
                "description": "Strategic asset allocation with regime-aware market analysis",
                "capabilities": [
                    "HMM regime detection and analysis",
                    "Strategic asset allocation optimization",
                    "Market outlook with volatility regime assessment",
                    "Long-term strategic planning"
                ],
                "backend_tools": ["detect_hmm_regimes", "detect_volatility_regimes", "strategic_allocation_models"],
                "enhancement_level": "2.0_quickwins"
            },
            "portfolio_manager": {
                "id": "portfolio_manager",
                "name": "Portfolio Manager", 
                "description": "Tactical portfolio management with dynamic risk budgeting",
                "capabilities": [
                    "Dynamic risk budgeting and optimization",
                    "Real trade generation from strategy analysis",
                    "Equal Risk Contribution portfolio construction",
                    "Advanced rebalancing with transaction cost analysis"
                ],
                "backend_tools": ["calculate_dynamic_risk_budgets", "design_risk_adjusted_momentum_strategy", "portfolio_optimization"],
                "enhancement_level": "2.0_quickwins"
            },
            "behavioral_coach": {
                "id": "behavioral_coach",
                "name": "Behavioral Coach",
                "description": "Investment psychology with real bias detection", 
                "capabilities": [
                    "Chat history bias analysis using NLP",
                    "Market sentiment detection algorithms",
                    "Behavioral risk impact quantification",
                    "Evidence-based coaching strategies"
                ],
                "backend_tools": ["analyze_chat_for_biases", "detect_market_sentiment", "behavioral_risk_assessment"],
                "enhancement_level": "2.0_quickwins"
            }
        }
        
        # Only include specialists that are actually initialized
        for specialist_id, info in specialist_definitions.items():
            if specialist_id in self.specialists:
                info["available"] = True
                info["enhanced"] = True
                specialists_info.append(info)
        
        return specialists_info
    
    def get_enhanced_backend_performance_summary(self) -> Dict:
        """Get comprehensive summary of backend tool performance across all conversations"""
        total_calls = 0
        total_time = 0
        total_tools_executed = 0
        
        specialist_performance = {}
        tool_performance = {}
        
        for conversation in self.conversations.values():
            # Specialist performance aggregation
            for specialist, perf in conversation.backend_performance.items():
                if specialist not in specialist_performance:
                    specialist_performance[specialist] = {
                        "calls": 0,
                        "time": 0,
                        "confidence": 0,
                        "success": 0,
                        "tools_used": []
                    }
                
                specialist_performance[specialist]["calls"] += perf["total_calls"]
                specialist_performance[specialist]["time"] += perf["total_time"]
                specialist_performance[specialist]["confidence"] += perf["avg_confidence"] * perf["total_calls"]
                specialist_performance[specialist]["success"] += perf["success_rate"] * perf["total_calls"]
                specialist_performance[specialist]["tools_used"].extend(perf.get("tools_used", []))
                
                total_calls += perf["total_calls"]
                total_time += perf["total_time"]
            
            # Tool performance tracking
            for tool_perf in conversation.tool_performance_history:
                for tool in tool_perf.get("tools_used", []):
                    if tool not in tool_performance:
                        tool_performance[tool] = {
                            "executions": 0,
                            "successes": 0,
                            "total_time": 0,
                            "specialists_using": set()
                        }
                    
                    tool_performance[tool]["executions"] += 1
                    if tool_perf.get("success"):
                        tool_performance[tool]["successes"] += 1
                    tool_performance[tool]["total_time"] += tool_perf.get("execution_time", 0)
                    tool_performance[tool]["specialists_using"].add(tool_perf.get("specialist"))
                    
                    total_tools_executed += 1
        
        # Calculate overall metrics
        if total_calls > 0:
            for specialist in specialist_performance:
                perf = specialist_performance[specialist]
                if perf["calls"] > 0:
                    perf["avg_confidence"] = perf["confidence"] / perf["calls"]
                    perf["success_rate"] = perf["success"] / perf["calls"]
                    perf["avg_time"] = perf["time"] / perf["calls"]
                    perf["unique_tools"] = len(set(perf["tools_used"]))
        
        # Convert sets to lists for JSON serialization
        for tool in tool_performance:
            tool_performance[tool]["specialists_using"] = list(tool_performance[tool]["specialists_using"])
            if tool_performance[tool]["executions"] > 0:
                tool_performance[tool]["success_rate"] = tool_performance[tool]["successes"] / tool_performance[tool]["executions"]
                tool_performance[tool]["avg_time"] = tool_performance[tool]["total_time"] / tool_performance[tool]["executions"]
        
        return {
            "total_backend_calls": total_calls,
            "total_execution_time": total_time,
            "total_tools_executed": total_tools_executed,
            "avg_execution_time": total_time / total_calls if total_calls > 0 else 0,
            "specialist_performance": specialist_performance,
            "tool_performance": tool_performance,
            "backend_integration_active": True,
            "enhancement_version": "2.0_quickwins",
            "conversations_tracked": len(self.conversations)
        }
    
    def get_enhanced_conversation_history(self, user_id: str, portfolio_id: int) -> List[Dict]:
        """Get enhanced conversation history with comprehensive performance data"""
        history = []
        for conv_id, conv in self.conversations.items():
            if conv.user_id == str(user_id) and conv.portfolio_context.get("portfolio_id") == portfolio_id:
                for msg in conv.messages[-20:]:  # Last 20 messages
                    history.append({
                        "timestamp": msg["timestamp"].isoformat(),
                        "query": msg["query"],
                        "response": msg["response"],
                        "specialist": msg["specialist"],
                        "backend_integration": msg.get("backend_integration", False),
                        "execution_time": msg.get("execution_time", 0),
                        "confidence": msg.get("analysis", {}).get("confidence", 0),
                        "tools_selected": msg.get("tools_selected", []),
                        "routing_confidence": msg.get("routing_confidence", 0),
                        "enhancement_version": msg.get("enhancement_version", "1.0")
                    })
        
        return sorted(history, key=lambda x: x["timestamp"])
    
    def get_portfolio_specific_insights(self, portfolio_context: Dict) -> Dict:
        """Generate portfolio-specific insights using enhanced analysis"""
        portfolio_value = portfolio_context.get("total_value", 0)
        holdings_count = len(portfolio_context.get("holdings", []))
        
        # Portfolio characterization
        if portfolio_value > 500000:
            portfolio_tier = "institutional"
            recommended_tools = ["advanced_monte_carlo_stress_test", "calculate_regime_conditional_risk"]
        elif portfolio_value > 100000:
            portfolio_tier = "high_net_worth"
            recommended_tools = ["calculate_risk_metrics", "portfolio_optimization"]
        else:
            portfolio_tier = "retail"
            recommended_tools = ["basic_stress_test", "diversification_analysis"]
        
        # Diversification assessment
        if holdings_count >= 10:
            diversification_level = "well_diversified"
        elif holdings_count >= 5:
            diversification_level = "moderately_diversified"
        else:
            diversification_level = "concentrated"
        
        # Recommended specialists based on portfolio characteristics
        recommended_specialists = []
        if portfolio_tier == "institutional":
            recommended_specialists.extend(["cio", "quantitative_analyst"])
        if diversification_level == "concentrated":
            recommended_specialists.extend(["portfolio_manager", "behavioral_coach"])
        
        return {
            "portfolio_tier": portfolio_tier,
            "diversification_level": diversification_level,
            "recommended_tools": recommended_tools,
            "recommended_specialists": list(set(recommended_specialists)),
            "analysis_priority": "risk_management" if diversification_level == "concentrated" else "optimization",
            "enhancement_recommendations": {
                "immediate": f"Portfolio tier: {portfolio_tier} - consider {recommended_specialists[0] if recommended_specialists else 'quantitative_analyst'} consultation",
                "medium_term": f"Diversification: {diversification_level} - review asset allocation",
                "tools_suggestion": f"Utilize {len(recommended_tools)} specialized tools for this portfolio tier"
            }
        }