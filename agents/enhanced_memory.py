# agents/semantic_router.py - Enhanced Semantic Routing
"""
Semantic-based routing system for investment committee agents
Replaces keyword matching with embedding-based similarity
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
    """Enhanced semantic routing system for investment committee"""
    
    def __init__(self):
        # Initialize embedding model (lightweight but effective)
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
        
        # Pre-computed agent embeddings with expanded descriptions
        self.agent_profiles = {
            "quantitative_analyst": {
                "description": "Risk analysis value at risk VaR stress testing portfolio metrics quantitative modeling statistical analysis volatility correlation variance covariance Monte Carlo simulation tail risk conditional value at risk CVaR portfolio optimization mathematical models regression analysis time series analysis GARCH modeling factor models principal component analysis eigen values eigen vectors portfolio theory modern portfolio theory efficient frontier risk budgeting",
                "expertise_keywords": ["risk", "var", "cvar", "stress", "volatility", "correlation", "regression", "monte carlo", "statistical", "quantitative", "mathematical", "modeling", "variance", "covariance", "tail risk", "eigen", "pca"],
                "confidence_weight": 1.2
            },
            "portfolio_manager": {
                "description": "Portfolio optimization asset allocation rebalancing trading recommendations tactical allocation strategic allocation position sizing trade execution implementation risk management portfolio construction equal risk contribution risk parity momentum strategies value strategies growth strategies factor investing smart beta active management passive management benchmark tracking tracking error information ratio alpha beta portfolio performance attribution transaction costs liquidity optimization",
                "expertise_keywords": ["portfolio", "allocation", "rebalancing", "trading", "optimization", "tactical", "strategic", "momentum", "value", "growth", "factor", "beta", "alpha", "tracking", "performance", "attribution"],
                "confidence_weight": 1.3
            },
            "cio": {
                "description": "Strategic asset allocation market outlook economic analysis macroeconomic factors interest rates inflation economic cycles market regimes asset class performance sector rotation geographic allocation currency exposure geopolitical risks market timing strategic planning long term investment strategy policy portfolio investment philosophy asset liability matching endowment management pension fund management sovereign wealth fund management",
                "expertise_keywords": ["strategic", "market", "economic", "macro", "interest rates", "inflation", "cycles", "regimes", "sector", "geographic", "currency", "geopolitical", "policy", "philosophy", "endowment", "pension"],
                "confidence_weight": 1.1
            },
            "behavioral_coach": {
                "description": "Investment psychology behavioral finance cognitive biases emotional decision making loss aversion prospect theory anchoring bias confirmation bias overconfidence bias herding behavior mental accounting framing effects availability bias representativeness bias behavioral risk management investor psychology market psychology sentiment analysis fear greed market bubbles panic selling emotional investing discipline mindfulness meditation",
                "expertise_keywords": ["behavioral", "psychology", "bias", "emotional", "sentiment", "fear", "greed", "panic", "discipline", "mindfulness", "cognitive", "loss aversion", "overconfidence", "anchoring", "herding"],
                "confidence_weight": 1.0
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
        
        logger.info(f"Semantic router initialized with {len(self.agent_embeddings)} agent embeddings")
    
    def route_query(self, 
                   query: str, 
                   conversation_history: List[Dict] = None,
                   user_preferences: Dict = None,
                   portfolio_context: Dict = None) -> RoutingDecision:
        """
        Route query using semantic similarity with context awareness
        """
        if not self.model or not self.agent_embeddings:
            # Fallback to keyword routing
            return self._fallback_keyword_routing(query)
        
        try:
            # Encode query
            query_embedding = self.model.encode(query)
            
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
            
            # Combine scores
            final_scores = {}
            for agent in semantic_scores:
                confidence_weight = self.agent_profiles[agent]["confidence_weight"]
                
                final_score = (
                    semantic_scores[agent] * confidence_weight +
                    context_bonuses.get(agent, 0) +
                    preference_bonuses.get(agent, 0)
                )
                final_scores[agent] = final_score
            
            # Select best agent
            primary_agent = max(final_scores.items(), key=lambda x: x[1])[0]
            confidence = min(0.95, max(0.60, final_scores[primary_agent]))
            
            # Get backup agents
            sorted_agents = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            backup_agents = [agent for agent, _ in sorted_agents[1:3]]
            
            # Generate reasoning
            reasoning = self._generate_routing_reasoning(
                primary_agent, semantic_scores[primary_agent], 
                context_bonuses.get(primary_agent, 0),
                preference_bonuses.get(primary_agent, 0)
            )
            
            return RoutingDecision(
                primary_agent=primary_agent,
                confidence=confidence,
                reasoning=reasoning,
                semantic_score=semantic_scores[primary_agent],
                context_bonus=context_bonuses.get(primary_agent, 0),
                user_preference_bonus=preference_bonuses.get(primary_agent, 0),
                backup_agents=backup_agents
            )
            
        except Exception as e:
            logger.error(f"Semantic routing failed: {e}")
            return self._fallback_keyword_routing(query)
    
    def _calculate_context_bonuses(self, 
                                  query: str, 
                                  conversation_history: List[Dict] = None,
                                  portfolio_context: Dict = None) -> Dict[str, float]:
        """Calculate context-based routing bonuses"""
        bonuses = {}
        
        # Conversation continuity bonus
        if conversation_history:
            recent_agents = []
            for msg in conversation_history[-3:]:  # Last 3 messages
                if isinstance(msg, dict) and msg.get("specialist"):
                    recent_agents.append(msg["specialist"])
            
            # Moderate continuity bonus (not too strong to avoid lock-in)
            if recent_agents:
                most_recent = recent_agents[-1]
                if recent_agents.count(most_recent) >= 2:
                    bonuses[most_recent] = 0.05  # 5% bonus for continuity
        
        # Portfolio characteristics bonus
        if portfolio_context:
            portfolio_value = portfolio_context.get("total_value", 0)
            risk_level = portfolio_context.get("riskLevel", "MODERATE")
            holdings_count = len(portfolio_context.get("holdings", []))
            
            # High-value portfolios get CIO bonus
            if portfolio_value > 500000:
                bonuses["cio"] = bonuses.get("cio", 0) + 0.03
            
            # High risk portfolios get quantitative analyst bonus
            if risk_level == "HIGH":
                bonuses["quantitative_analyst"] = bonuses.get("quantitative_analyst", 0) + 0.04
            
            # Complex portfolios get portfolio manager bonus
            if holdings_count > 8:
                bonuses["portfolio_manager"] = bonuses.get("portfolio_manager", 0) + 0.03
        
        # Query urgency bonus
        urgency_words = ["urgent", "immediately", "asap", "emergency", "crisis", "crash"]
        if any(word in query.lower() for word in urgency_words):
            bonuses["quantitative_analyst"] = bonuses.get("quantitative_analyst", 0) + 0.04
        
        return bonuses
    
    def _calculate_user_preference_bonuses(self, 
                                         user_preferences: Dict = None,
                                         semantic_scores: Dict = None) -> Dict[str, float]:
        """Calculate user preference bonuses based on historical interactions"""
        bonuses = {}
        
        if not user_preferences:
            return bonuses
        
        # Agent satisfaction bonuses
        agent_satisfaction = user_preferences.get("agent_satisfaction_scores", {})
        for agent, satisfaction in agent_satisfaction.items():
            if satisfaction > 0.7:  # High satisfaction
                bonuses[agent] = 0.05
            elif satisfaction < 0.4:  # Low satisfaction
                bonuses[agent] = -0.03
        
        # Complexity preference bonus
        complexity_pref = user_preferences.get("query_complexity_preference", 0.5)
        if complexity_pref > 0.7:  # Prefers detailed analysis
            bonuses["quantitative_analyst"] = bonuses.get("quantitative_analyst", 0) + 0.03
        
        # Detail level preference
        detail_pref = user_preferences.get("detail_level_preference", 0.5)
        if detail_pref > 0.8:  # Prefers high detail
            bonuses["quantitative_analyst"] = bonuses.get("quantitative_analyst", 0) + 0.02
            bonuses["cio"] = bonuses.get("cio", 0) + 0.02
        
        return bonuses
    
    def _generate_routing_reasoning(self, 
                                  primary_agent: str, 
                                  semantic_score: float,
                                  context_bonus: float,
                                  preference_bonus: float) -> str:
        """Generate human-readable routing reasoning"""
        components = []
        
        # Semantic component
        if semantic_score > 0.7:
            components.append(f"Strong semantic match ({semantic_score:.3f})")
        elif semantic_score > 0.5:
            components.append(f"Good semantic match ({semantic_score:.3f})")
        else:
            components.append(f"Moderate semantic match ({semantic_score:.3f})")
        
        # Context component
        if context_bonus > 0.03:
            components.append("strong context alignment")
        elif context_bonus > 0:
            components.append("context alignment")
        
        # Preference component
        if preference_bonus > 0.03:
            components.append("user preference match")
        elif preference_bonus > 0:
            components.append("slight preference match")
        
        agent_name = primary_agent.replace("_", " ").title()
        return f"{agent_name} selected based on: {', '.join(components)}"
    
    def _fallback_keyword_routing(self, query: str) -> RoutingDecision:
        """Fallback keyword-based routing when semantic fails"""
        query_lower = query.lower()
        
        # Simple keyword scoring
        scores = {}
        for agent, profile in self.agent_profiles.items():
            score = 0
            keywords = profile["expertise_keywords"]
            for keyword in keywords:
                if keyword in query_lower:
                    score += 1
            scores[agent] = score * profile["confidence_weight"]
        
        if scores and max(scores.values()) > 0:
            primary_agent = max(scores.items(), key=lambda x: x[1])[0]
            confidence = min(0.85, max(0.60, scores[primary_agent] / 10))
        else:
            # Ultimate fallback
            primary_agent = "quantitative_analyst"
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
    
    def update_user_preferences(self, 
                              user_id: str,
                              agent_used: str, 
                              satisfaction_rating: float,
                              interaction_data: Dict):
        """Update user preferences based on interaction feedback"""
        # This would integrate with your conversation memory system
        # Implementation depends on your user preference storage
        pass
    
    def get_routing_analytics(self) -> Dict:
        """Get analytics about routing performance"""
        return {
            "semantic_model_loaded": self.model is not None,
            "agents_configured": len(self.agent_embeddings),
            "fallback_available": True,
            "routing_features": [
                "semantic_similarity",
                "conversation_context",
                "portfolio_context", 
                "user_preferences",
                "confidence_weighting"
            ]
        }