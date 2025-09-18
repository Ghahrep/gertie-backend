"""
Tuned semantic-based routing system for investment committee agents
Improved agent descriptions for better routing accuracy
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class RoutingDecision:
    """Enhanced routing decision with confidence and reasoning"""
    primary_agent: str
    confidence: float
    reasoning: str
    semantic_score: float
    context_bonus: float
    user_preference_bonus: float
    backup_agents: List[str]

class SemanticAgentRouter:
    """Enhanced semantic routing system for investment committee with tuned descriptions"""
    
    def __init__(self):
        # Initialize embedding model (lightweight but effective)
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
        
        # TUNED: More distinct agent profiles with balanced descriptions
        self.agent_profiles = {
            "quantitative_analyst": {
                "description": "Risk measurement VaR CVaR stress testing Monte Carlo simulation statistical analysis portfolio risk metrics volatility analysis correlation matrix mathematical modeling quantitative finance risk assessment tail risk downside risk maximum drawdown Sharpe ratio beta calculation variance covariance analysis GARCH models factor analysis principal components eigenvalues statistical significance confidence intervals",
                "expertise_keywords": ["risk", "var", "cvar", "stress", "volatility", "correlation", "statistical", "quantitative", "mathematical", "modeling", "variance", "covariance", "tail risk", "monte carlo", "simulation"],
                "confidence_weight": 1.0  # REDUCED from 1.2
            },
            "portfolio_manager": {
                "description": "Portfolio rebalancing asset allocation optimization trade execution buy sell decisions position sizing tactical allocation portfolio construction equal risk contribution momentum strategies value investing growth strategies factor tilts smart beta active management portfolio optimization efficient frontier rebalancing frequency transaction costs implementation shortfall trading strategies liquidity management",
                "expertise_keywords": ["rebalance", "rebalancing", "optimize", "optimization", "allocation", "trade", "trading", "buy", "sell", "tactical", "momentum", "value", "growth", "factor", "beta", "implementation", "position sizing", "portfolio construction"],
                "confidence_weight": 1.3  # INCREASED emphasis
            },
            "cio": {
                "description": "Strategic asset allocation investment strategy long-term planning market outlook economic analysis macroeconomic trends interest rate environment inflation expectations market cycles asset class selection geographic diversification sector allocation currency hedging strategic portfolio policy investment philosophy asset liability matching endowment model pension fund strategy sovereign wealth management institutional investing",
                "expertise_keywords": ["strategy", "strategic", "long-term", "market outlook", "economic", "macro", "macroeconomic", "interest rates", "inflation", "cycles", "asset class", "geographic", "sector", "currency", "policy", "philosophy", "institutional"],
                "confidence_weight": 1.2  # INCREASED emphasis
            },
            "behavioral_coach": {
                "description": "Investment psychology behavioral finance cognitive biases emotional investing decision-making process loss aversion prospect theory anchoring bias confirmation bias overconfidence herding behavior mental accounting framing effects availability heuristic representativeness bias emotional regulation fear greed market psychology investor sentiment behavioral risk management investment discipline mindfulness meditation coaching guidance",
                "expertise_keywords": ["behavioral", "psychology", "bias", "biases", "emotional", "emotions", "sentiment", "fear", "greed", "panic", "anxiety", "worried", "scared", "feeling", "cognitive", "loss aversion", "overconfidence", "anchoring", "herding", "discipline", "coaching"],
                "confidence_weight": 1.3  # INCREASED emphasis
            }
        }
        
        # Pre-compute embeddings for efficiency
        self.agent_embeddings = {}
        if self.model:
            for agent, profile in self.agent_profiles.items():
                try:
                    self.agent_embeddings[agent] = self.model.encode(profile["description"])
                except Exception as e:
                    logger.error(f"Failed to encode embeddings for {agent}: {e}")
        
        logger.info(f"Tuned semantic router initialized with {len(self.agent_embeddings)} agent embeddings")
    
    def route_query(self, 
                   query: str, 
                   conversation_history: List[Dict] = None,
                   user_preferences: Dict = None,
                   portfolio_context: Dict = None) -> RoutingDecision:
        """
        Route query using semantic similarity with improved scoring
        """
        if not self.model or not self.agent_embeddings:
            # Fallback to keyword routing
            return self._fallback_keyword_routing(query)
        
        try:
            # Encode query with context for better matching
            enhanced_query = self._enhance_query_with_context(query, conversation_history)
            query_embedding = self.model.encode(enhanced_query)
            
            # Calculate semantic similarities
            semantic_scores = {}
            for agent, agent_embedding in self.agent_embeddings.items():
                similarity = cosine_similarity([query_embedding], [agent_embedding])[0][0]
                semantic_scores[agent] = float(similarity)
            
            # Apply context bonuses
            context_bonuses = self._calculate_context_bonuses(
                query, conversation_history, portfolio_context
            )
            
            # Apply user preference bonuses
            preference_bonuses = self._calculate_user_preference_bonuses(
                user_preferences, semantic_scores
            )
            
            # IMPROVED: More balanced scoring with threshold handling
            final_scores = {}
            for agent in semantic_scores:
                confidence_weight = self.agent_profiles[agent]["confidence_weight"]
                
                # Apply confidence weight more conservatively
                weighted_semantic = semantic_scores[agent] * confidence_weight
                
                final_score = (
                    weighted_semantic +
                    context_bonuses.get(agent, 0) * 0.5 +  # Reduced context weight
                    preference_bonuses.get(agent, 0) * 0.3  # Reduced preference weight
                )
                final_scores[agent] = final_score
            
            # IMPROVED: Better winner selection with minimum threshold
            sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            best_agent, best_score = sorted_scores[0]
            second_best_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
            
            # If top two scores are very close, prefer non-quantitative agent for diversity
            score_difference = best_score - second_best_score
            if score_difference < 0.02 and best_agent == "quantitative_analyst" and len(sorted_scores) > 1:
                logger.info(f"Close scores detected ({score_difference:.3f}), considering diversity")
                second_agent, second_score = sorted_scores[1]
                if second_agent != "quantitative_analyst":
                    best_agent = second_agent
                    best_score = second_score
                    logger.info(f"Selected {best_agent} for diversity")
            
            # Calculate confidence with better scaling
            confidence = min(0.95, max(0.60, best_score * 1.5))  # Improved scaling
            
            # Get backup agents
            backup_agents = [agent for agent, _ in sorted_scores[1:3]]
            
            # Generate reasoning
            reasoning = self._generate_routing_reasoning(
                best_agent, semantic_scores[best_agent], 
                context_bonuses.get(best_agent, 0),
                preference_bonuses.get(best_agent, 0)
            )
            
            return RoutingDecision(
                primary_agent=best_agent,
                confidence=confidence,
                reasoning=reasoning,
                semantic_score=semantic_scores[best_agent],
                context_bonus=context_bonuses.get(best_agent, 0),
                user_preference_bonus=preference_bonuses.get(best_agent, 0),
                backup_agents=backup_agents
            )
            
        except Exception as e:
            logger.error(f"Semantic routing failed: {e}")
            return self._fallback_keyword_routing(query)
    
    def _enhance_query_with_context(self, query: str, conversation_history: List[Dict] = None) -> str:
        """Enhance query with conversation context for better semantic matching"""
        enhanced_query = query
        
        if conversation_history:
            # Add context from recent conversation
            recent_context = []
            for msg in conversation_history[-2:]:  # Last 2 messages
                if msg.get("query"):
                    recent_context.append(msg["query"])
            
            if recent_context:
                enhanced_query = f"{query} Context: {' '.join(recent_context)}"
        
        return enhanced_query
    
    def _calculate_context_bonuses(self, 
                                  query: str, 
                                  conversation_history: List[Dict] = None,
                                  portfolio_context: Dict = None) -> Dict[str, float]:
        """Calculate context-based routing bonuses with improved logic"""
        bonuses = {}
        
        # Query-specific bonuses based on key phrases
        query_lower = query.lower()
        
        # IMPROVED: More specific phrase detection
        if any(phrase in query_lower for phrase in ["rebalance", "optimize allocation", "should i buy", "should i sell", "trade", "position"]):
            bonuses["portfolio_manager"] = 0.08
        
        if any(phrase in query_lower for phrase in ["strategy", "long-term", "market outlook", "asset class", "strategic"]):
            bonuses["cio"] = 0.08
        
        if any(phrase in query_lower for phrase in ["anxious", "worried", "scared", "panic", "emotional", "feeling", "bias"]):
            bonuses["behavioral_coach"] = 0.08
        
        if any(phrase in query_lower for phrase in ["var", "stress test", "monte carlo", "correlation", "volatility"]):
            bonuses["quantitative_analyst"] = 0.05  # Reduced from 0.08
        
        # Portfolio characteristics bonus
        if portfolio_context:
            portfolio_value = portfolio_context.get("total_value", 0)
            risk_level = portfolio_context.get("riskLevel", "MODERATE")
            holdings_count = len(portfolio_context.get("holdings", []))
            
            # High-value portfolios get strategic focus
            if portfolio_value > 500000:
                bonuses["cio"] = bonuses.get("cio", 0) + 0.03
            
            # High risk triggers behavioral analysis more than quantitative
            if risk_level == "HIGH":
                bonuses["behavioral_coach"] = bonuses.get("behavioral_coach", 0) + 0.04
                bonuses["quantitative_analyst"] = bonuses.get("quantitative_analyst", 0) + 0.02  # Reduced
            
            # Complex portfolios need management
            if holdings_count > 8:
                bonuses["portfolio_manager"] = bonuses.get("portfolio_manager", 0) + 0.04
        
        return bonuses
    
    def _calculate_user_preference_bonuses(self, 
                                         user_preferences: Dict = None,
                                         semantic_scores: Dict = None) -> Dict[str, float]:
        """Calculate user preference bonuses based on historical interactions"""
        bonuses = {}
        
        if not user_preferences:
            return bonuses
        
        # Agent satisfaction bonuses (reduced impact)
        agent_satisfaction = user_preferences.get("agent_satisfaction_scores", {})
        for agent, satisfaction in agent_satisfaction.items():
            if satisfaction > 0.8:  # Very high satisfaction
                bonuses[agent] = 0.03  # Reduced from 0.05
            elif satisfaction < 0.3:  # Low satisfaction
                bonuses[agent] = -0.02  # Reduced penalty
        
        return bonuses
    
    def _generate_routing_reasoning(self, 
                                  primary_agent: str, 
                                  semantic_score: float,
                                  context_bonus: float,
                                  preference_bonus: float) -> str:
        """Generate human-readable routing reasoning"""
        components = []
        
        # Semantic component
        if semantic_score > 0.6:
            components.append(f"strong semantic match ({semantic_score:.3f})")
        elif semantic_score > 0.4:
            components.append(f"good semantic match ({semantic_score:.3f})")
        else:
            components.append(f"moderate semantic match ({semantic_score:.3f})")
        
        # Context component
        if context_bonus > 0.05:
            components.append("strong context alignment")
        elif context_bonus > 0.02:
            components.append("context alignment")
        
        # Preference component
        if preference_bonus > 0.02:
            components.append("user preference match")
        
        agent_name = primary_agent.replace("_", " ").title()
        return f"{agent_name} selected based on: {', '.join(components)}"
    
    def _fallback_keyword_routing(self, query: str) -> RoutingDecision:
        """Fallback keyword-based routing when semantic fails"""
        query_lower = query.lower()
        
        # Improved keyword scoring
        scores = {}
        for agent, profile in self.agent_profiles.items():
            score = 0
            keywords = profile["expertise_keywords"]
            for keyword in keywords:
                if keyword in query_lower:
                    # Give more weight to exact matches
                    if keyword == query_lower.strip():
                        score += 3
                    else:
                        score += 1
            scores[agent] = score * profile["confidence_weight"]
        
        if scores and max(scores.values()) > 0:
            primary_agent = max(scores.items(), key=lambda x: x[1])[0]
            confidence = min(0.85, max(0.60, scores[primary_agent] / 8))  # Improved scaling
        else:
            # Better fallback logic
            primary_agent = "quantitative_analyst"  # Safe default
            confidence = 0.60
        
        return RoutingDecision(
            primary_agent=primary_agent,
            confidence=confidence,
            reasoning="Keyword-based fallback routing",
            semantic_score=0,
            context_bonus=0,
            user_preference_bonus=0,
            backup_agents=[]
        )
    
    def get_routing_analytics(self) -> Dict:
        """Get analytics about routing performance"""
        return {
            "semantic_model_loaded": self.model is not None,
            "agents_configured": len(self.agent_embeddings),
            "fallback_available": True,
            "version": "tuned_v1.1",
            "routing_features": [
                "enhanced_semantic_similarity",
                "conversation_context",
                "portfolio_context", 
                "user_preferences",
                "balanced_confidence_weighting",
                "diversity_preference"
            ]
        }