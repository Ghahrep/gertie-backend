# agents/advanced_memory.py - Advanced Memory System Core Implementation
"""
Advanced Memory System for Investment Committee
==============================================

Week 1 Implementation: Core memory foundation with semantic understanding,
user preference learning, and conversation pattern analysis.
"""

import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path

# Safe imports with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("sentence-transformers not available, using fallback similarity")

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Enhanced conversation turn with semantic understanding"""
    turn_id: str
    timestamp: datetime
    user_query: str
    agent_response: str
    specialist: str
    confidence: float
    risk_score: float
    tool_results: List[Dict] = field(default_factory=list)
    execution_time: float = 0.0
    user_satisfaction: Optional[float] = None
    concepts_extracted: List[str] = field(default_factory=list)
    query_embedding: Optional[np.ndarray] = None
    response_embedding: Optional[np.ndarray] = None
    collaboration_involved: bool = False
    secondary_specialists: List[str] = field(default_factory=list)
    portfolio_context: Dict = field(default_factory=dict)

@dataclass  
class UserProfile:
    """User learning profile for personalization"""
    user_id: str
    agent_satisfaction_scores: Dict[str, float] = field(default_factory=dict)
    query_complexity_preference: float = 0.5  # 0-1 scale
    detail_level_preference: float = 0.5  # 0-1 scale
    risk_tolerance_indicated: float = 0.5  # 0-1 scale
    behavioral_patterns: Dict[str, float] = field(default_factory=dict)
    portfolio_evolution: List[Dict] = field(default_factory=list)
    conversation_frequency: float = 0.0
    last_active: datetime = field(default_factory=datetime.now)
    expertise_level: str = "intermediate"  # novice, intermediate, advanced
    preferred_collaboration_style: str = "balanced"  # minimal, balanced, comprehensive
    total_conversations: int = 0
    total_collaborations: int = 0

class EnhancedConversationMemory:
    """
    Advanced conversation memory with semantic understanding and learning
    """
    
    def __init__(self, conversation_id: str, user_id: Optional[str] = None):
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.turns: List[ConversationTurn] = []
        self.user_profile: Optional[UserProfile] = None
        self.portfolio_evolution: List[Dict] = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Initialize semantic understanding components
        self._initialize_semantic_components()
        
        # Initialize learning components
        self.concept_extractor = ConceptExtractor()
        self.pattern_analyzer = ConversationPatternAnalyzer()
        self.preference_learner = UserPreferenceLearner()
        self.portfolio_tracker = PortfolioEvolutionTracker()
        
        logger.info(f"Advanced conversation memory initialized for {conversation_id}")
        
    def _initialize_semantic_components(self):
        """Initialize semantic understanding with fallback"""
        if EMBEDDINGS_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embeddings_enabled = True
                logger.info("Semantic embeddings enabled")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.embeddings_enabled = False
        else:
            self.embeddings_enabled = False
            logger.info("Using fallback similarity without embeddings")
    
    def add_turn(self, 
                 user_query: str,
                 agent_response: str, 
                 specialist: str,
                 confidence: float,
                 risk_score: float,
                 tool_results: List[Dict] = None,
                 execution_time: float = 0.0,
                 user_satisfaction: Optional[float] = None,
                 collaboration_involved: bool = False,
                 secondary_specialists: List[str] = None,
                 portfolio_context: Dict = None) -> ConversationTurn:
        """Add conversation turn with semantic processing"""
        
        if tool_results is None:
            tool_results = []
        if secondary_specialists is None:
            secondary_specialists = []
        if portfolio_context is None:
            portfolio_context = {}
        
        # Generate embeddings for semantic search if available
        query_embedding = None
        response_embedding = None
        
        if self.embeddings_enabled:
            try:
                query_embedding = self.semantic_model.encode(user_query)
                response_embedding = self.semantic_model.encode(agent_response)
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")
        
        # Extract concepts for semantic understanding
        concepts = self.concept_extractor.extract_financial_concepts(
            user_query, agent_response, tool_results
        )
        
        turn = ConversationTurn(
            turn_id=f"{self.conversation_id}_{len(self.turns)}",
            timestamp=datetime.now(),
            user_query=user_query,
            agent_response=agent_response,
            specialist=specialist,
            confidence=confidence,
            risk_score=risk_score,
            tool_results=tool_results,
            execution_time=execution_time,
            user_satisfaction=user_satisfaction,
            concepts_extracted=concepts,
            query_embedding=query_embedding,
            response_embedding=response_embedding,
            collaboration_involved=collaboration_involved,
            secondary_specialists=secondary_specialists,
            portfolio_context=portfolio_context
        )
        
        self.turns.append(turn)
        self.updated_at = datetime.now()
        
        # Update user profile with learning
        if self.user_id:
            self._update_user_profile(turn)
        
        # Track portfolio evolution
        if portfolio_context:
            self.portfolio_tracker.add_portfolio_snapshot(
                self.portfolio_evolution, portfolio_context
            )
        
        logger.info(f"Added conversation turn: {specialist} analysis with {confidence}% confidence")
        return turn
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[ConversationTurn, float]]:
        """Search conversation history using semantic similarity"""
        if not self.turns:
            return []
        
        if self.embeddings_enabled:
            return self._semantic_search_with_embeddings(query, top_k)
        else:
            return self._semantic_search_fallback(query, top_k)
    
    def _semantic_search_with_embeddings(self, query: str, top_k: int) -> List[Tuple[ConversationTurn, float]]:
        """Semantic search using embeddings"""
        try:
            query_embedding = self.semantic_model.encode(query)
            similarities = []
            
            for turn in self.turns:
                if turn.query_embedding is not None and turn.response_embedding is not None:
                    # Calculate similarity with both query and response
                    query_sim = self._cosine_similarity(query_embedding, turn.query_embedding)
                    response_sim = self._cosine_similarity(query_embedding, turn.response_embedding)
                    
                    # Weighted combination (query similarity weighted higher)
                    combined_sim = 0.7 * query_sim + 0.3 * response_sim
                    similarities.append((turn, float(combined_sim)))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Embedding search failed: {e}")
            return self._semantic_search_fallback(query, top_k)
    
    def _semantic_search_fallback(self, query: str, top_k: int) -> List[Tuple[ConversationTurn, float]]:
        """Fallback search using keyword similarity"""
        query_words = set(query.lower().split())
        similarities = []
        
        for turn in self.turns:
            # Simple keyword overlap similarity
            turn_words = set(turn.user_query.lower().split())
            overlap = len(query_words.intersection(turn_words))
            similarity = overlap / max(len(query_words), len(turn_words), 1)
            similarities.append((turn, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def get_recent_context(self, turns: int = 5) -> List[ConversationTurn]:
        """Get recent conversation context"""
        return self.turns[-turns:] if self.turns else []
    
    def get_portfolio_insights(self) -> Dict:
        """Get insights from portfolio evolution"""
        if not self.portfolio_evolution:
            return {"status": "no_portfolio_data"}
        
        return self.portfolio_tracker.analyze_evolution(self.portfolio_evolution)
    
    def get_user_learning_insights(self) -> Dict:
        """Get insights about user preferences and patterns"""
        if not self.user_profile:
            return {"status": "no_user_profile"}
        
        conversation_patterns = self.pattern_analyzer.analyze_patterns(self.turns)
        
        return {
            "agent_preferences": self.user_profile.agent_satisfaction_scores,
            "complexity_preference": self.user_profile.query_complexity_preference,
            "detail_preference": self.user_profile.detail_level_preference,
            "risk_tolerance": self.user_profile.risk_tolerance_indicated,
            "expertise_level": self.user_profile.expertise_level,
            "collaboration_preference": self.user_profile.preferred_collaboration_style,
            "conversation_patterns": conversation_patterns,
            "total_conversations": self.user_profile.total_conversations,
            "learning_confidence": self._calculate_learning_confidence()
        }
    
    def get_personalized_routing_weights(self) -> Dict[str, float]:
        """Get personalized agent routing weights based on user preferences"""
        if not self.user_profile or not self.user_profile.agent_satisfaction_scores:
            return {}  # Use default routing
        
        # Base weights
        weights = {
            'quantitative_analyst': 1.0,
            'portfolio_manager': 1.0,
            'behavioral_coach': 1.0,
            'cio': 1.0
        }
        
        # Adjust based on user satisfaction scores
        for agent, satisfaction in self.user_profile.agent_satisfaction_scores.items():
            if agent in weights:
                # Convert satisfaction (0-1) to weight multiplier (0.5-1.5)
                weights[agent] = 0.5 + satisfaction
        
        # Adjust based on user preferences
        if self.user_profile.detail_level_preference > 0.7:
            weights['quantitative_analyst'] *= 1.2  # Prefer detailed analysis
        
        if self.user_profile.risk_tolerance_indicated < 0.3:
            weights['behavioral_coach'] *= 1.3  # Prefer behavioral guidance for risk-averse users
        
        return weights
    
    def _update_user_profile(self, turn: ConversationTurn):
        """Update user profile based on conversation turn"""
        if not self.user_profile:
            self.user_profile = UserProfile(user_id=self.user_id)
        
        # Update agent satisfaction if provided
        if turn.user_satisfaction is not None:
            self._update_agent_satisfaction(turn.specialist, turn.user_satisfaction)
        
        # Learn preferences from behavior
        self.preference_learner.update_preferences(self.user_profile, turn)
        
        # Update activity tracking
        self.user_profile.last_active = datetime.now()
        self.user_profile.total_conversations += 1
        
        if turn.collaboration_involved:
            self.user_profile.total_collaborations += 1
        
        # Update collaboration preference
        collaboration_rate = self.user_profile.total_collaborations / max(1, self.user_profile.total_conversations)
        if collaboration_rate > 0.7:
            self.user_profile.preferred_collaboration_style = "comprehensive"
        elif collaboration_rate < 0.3:
            self.user_profile.preferred_collaboration_style = "minimal"
        else:
            self.user_profile.preferred_collaboration_style = "balanced"
    
    def _update_agent_satisfaction(self, specialist: str, satisfaction: float):
        """Update agent satisfaction using exponential moving average"""
        current_score = self.user_profile.agent_satisfaction_scores.get(specialist, 0.5)
        alpha = 0.3  # Learning rate
        new_score = alpha * satisfaction + (1 - alpha) * current_score
        self.user_profile.agent_satisfaction_scores[specialist] = new_score
    
    def _calculate_learning_confidence(self) -> float:
        """Calculate confidence in learned preferences"""
        if not self.user_profile:
            return 0.0
        
        # Confidence increases with more interactions
        interaction_confidence = min(1.0, self.user_profile.total_conversations / 10)
        
        # Confidence increases with more feedback
        feedback_count = len([s for s in self.user_profile.agent_satisfaction_scores.values() if s != 0.5])
        feedback_confidence = min(1.0, feedback_count / 4)
        
        return (interaction_confidence + feedback_confidence) / 2

class ConceptExtractor:
    """Extract financial concepts from conversations"""
    
    def __init__(self):
        self.financial_concepts = {
            "risk_concepts": ["volatility", "var", "risk", "correlation", "beta", "stress test", "downside", "upside"],
            "portfolio_concepts": ["allocation", "diversification", "rebalancing", "optimization", "weights", "holdings"],
            "market_concepts": ["market", "regime", "trend", "sentiment", "outlook", "economic", "macro"],
            "behavioral_concepts": ["bias", "emotional", "fear", "greed", "confidence", "worried", "scared"],
            "strategy_concepts": ["strategy", "long-term", "tactical", "strategic", "planning", "goals"],
            "performance_concepts": ["return", "performance", "gains", "losses", "profit", "yield"]
        }
    
    def extract_financial_concepts(self, query: str, response: str, tool_results: List[Dict]) -> List[str]:
        """Extract relevant financial concepts from conversation"""
        concepts = []
        text = f"{query} {response}".lower()
        
        # Extract concept categories
        for category, concept_list in self.financial_concepts.items():
            category_found = False
            for concept in concept_list:
                if concept in text:
                    if not category_found:
                        concepts.append(category)
                        category_found = True
                    concepts.append(f"{category}:{concept}")
        
        # Extract concepts from tool results
        for tool_result in tool_results:
            if isinstance(tool_result, dict):
                if "analysis" in tool_result:
                    analysis = tool_result["analysis"]
                    if isinstance(analysis, dict):
                        for key in analysis.keys():
                            if key in ["riskScore", "confidence", "recommendation"]:
                                concepts.append(f"tool_analysis:{key}")
        
        return list(set(concepts))

class ConversationPatternAnalyzer:
    """Analyze patterns in conversation behavior"""
    
    def analyze_patterns(self, turns: List[ConversationTurn]) -> Dict:
        """Analyze user conversation patterns"""
        if not turns:
            return {"status": "no_conversations"}
        
        patterns = {
            "avg_session_length": len(turns),
            "preferred_specialists": self._get_specialist_preferences(turns),
            "query_complexity": self._analyze_query_complexity(turns),
            "risk_discussion_frequency": self._count_risk_discussions(turns),
            "collaboration_preference": self._analyze_collaboration_preference(turns),
            "time_of_day_patterns": self._analyze_timing_patterns(turns),
            "concept_interests": self._analyze_concept_interests(turns),
            "confidence_patterns": self._analyze_confidence_patterns(turns)
        }
        
        return patterns
    
    def _get_specialist_preferences(self, turns: List[ConversationTurn]) -> Dict:
        """Analyze which specialists user interacts with most"""
        specialist_counts = {}
        for turn in turns:
            specialist_counts[turn.specialist] = specialist_counts.get(turn.specialist, 0) + 1
        
        total = len(turns)
        return {spec: count/total for spec, count in specialist_counts.items()}
    
    def _analyze_query_complexity(self, turns: List[ConversationTurn]) -> float:
        """Analyze average query complexity"""
        complexity_scores = []
        for turn in turns:
            # Complexity based on query length and concepts
            query_length_score = min(1.0, len(turn.user_query.split()) / 20)
            concept_score = min(1.0, len(turn.concepts_extracted) / 5) if turn.concepts_extracted else 0
            collaboration_bonus = 0.2 if turn.collaboration_involved else 0
            
            complexity = (query_length_score + concept_score + collaboration_bonus) / 2
            complexity_scores.append(complexity)
        
        return sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0.5
    
    def _count_risk_discussions(self, turns: List[ConversationTurn]) -> float:
        """Count frequency of risk-related discussions"""
        risk_turns = sum(1 for turn in turns if turn.risk_score > 60)
        return risk_turns / len(turns) if turns else 0
    
    def _analyze_collaboration_preference(self, turns: List[ConversationTurn]) -> float:
        """Analyze preference for collaborative analysis"""
        collab_turns = sum(1 for turn in turns if turn.collaboration_involved)
        return collab_turns / len(turns) if turns else 0
    
    def _analyze_timing_patterns(self, turns: List[ConversationTurn]) -> Dict:
        """Analyze when user typically engages"""
        hours = [turn.timestamp.hour for turn in turns]
        if not hours:
            return {"status": "no_timing_data"}
        
        return {
            "most_active_hour": max(set(hours), key=hours.count),
            "business_hours_preference": sum(1 for h in hours if 9 <= h <= 17) / len(hours),
            "evening_preference": sum(1 for h in hours if 17 <= h <= 21) / len(hours)
        }
    
    def _analyze_concept_interests(self, turns: List[ConversationTurn]) -> Dict:
        """Analyze user's concept interests"""
        concept_counts = {}
        for turn in turns:
            for concept in turn.concepts_extracted:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        total_concepts = sum(concept_counts.values())
        if total_concepts == 0:
            return {"status": "no_concepts"}
        
        # Return top 5 concept interests
        sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
        return {concept: count/total_concepts for concept, count in sorted_concepts[:5]}
    
    def _analyze_confidence_patterns(self, turns: List[ConversationTurn]) -> Dict:
        """Analyze confidence patterns in responses"""
        confidences = [turn.confidence for turn in turns if turn.confidence > 0]
        if not confidences:
            return {"status": "no_confidence_data"}
        
        return {
            "avg_confidence": sum(confidences) / len(confidences),
            "confidence_trend": "improving" if len(confidences) > 1 and confidences[-1] > confidences[0] else "stable",
            "low_confidence_frequency": sum(1 for c in confidences if c < 70) / len(confidences)
        }

class UserPreferenceLearner:
    """Learn user preferences from interaction patterns"""
    
    def update_preferences(self, profile: UserProfile, turn: ConversationTurn):
        """Update user preferences based on conversation turn"""
        
        # Update complexity preference based on query characteristics
        query_complexity = self._assess_query_complexity(turn)
        alpha = 0.2  # Learning rate
        profile.query_complexity_preference = (
            alpha * query_complexity + (1 - alpha) * profile.query_complexity_preference
        )
        
        # Update detail preference based on response engagement
        detail_preference = self._assess_detail_preference(turn)
        profile.detail_level_preference = (
            alpha * detail_preference + (1 - alpha) * profile.detail_level_preference
        )
        
        # Update risk tolerance based on risk scores and reactions
        if turn.risk_score > 0:
            risk_tolerance = self._assess_risk_tolerance(turn)
            profile.risk_tolerance_indicated = (
                alpha * risk_tolerance + (1 - alpha) * profile.risk_tolerance_indicated
            )
        
        # Update expertise level based on patterns
        self._update_expertise_level(profile, turn)
    
    def _assess_query_complexity(self, turn: ConversationTurn) -> float:
        """Assess complexity of user's query"""
        query_length = len(turn.user_query.split())
        concept_count = len(turn.concepts_extracted) if turn.concepts_extracted else 0
        
        # Normalize to 0-1 scale
        length_score = min(1.0, query_length / 25)
        concept_score = min(1.0, concept_count / 8)
        
        return (length_score + concept_score) / 2
    
    def _assess_detail_preference(self, turn: ConversationTurn) -> float:
        """Assess user's preference for detailed analysis"""
        # Longer execution time suggests more detailed analysis was requested/valued
        execution_score = min(1.0, turn.execution_time * 5)
        
        # Collaboration suggests preference for comprehensive analysis
        collaboration_score = 0.8 if turn.collaboration_involved else 0.2
        
        return (execution_score + collaboration_score) / 2
    
    def _assess_risk_tolerance(self, turn: ConversationTurn) -> float:
        """Assess user's risk tolerance from conversation"""
        emotional_words = ["worried", "scared", "anxious", "concerned", "nervous"]
        confidence_words = ["confident", "comfortable", "optimistic", "bullish"]
        
        query_lower = turn.user_query.lower()
        
        has_emotional_language = any(word in query_lower for word in emotional_words)
        has_confidence_language = any(word in query_lower for word in confidence_words)
        
        if turn.risk_score > 70 and not has_emotional_language:
            return 0.8  # High tolerance - comfortable with high risk
        elif turn.risk_score > 70 and has_emotional_language:
            return 0.2  # Low tolerance - worried about high risk
        elif has_confidence_language:
            return 0.7  # Confidence suggests higher risk tolerance
        else:
            return 0.5  # Neutral
    
    def _update_expertise_level(self, profile: UserProfile, turn: ConversationTurn):
        """Update user expertise level based on conversation patterns"""
        # Simple heuristic based on query complexity and concepts used
        if turn.concepts_extracted:
            advanced_concepts = sum(1 for concept in turn.concepts_extracted 
                                  if any(advanced in concept for advanced in 
                                        ["var", "beta", "correlation", "regime", "optimization"]))
            
            if advanced_concepts >= 3 and profile.total_conversations >= 5:
                profile.expertise_level = "advanced"
            elif advanced_concepts >= 1 and profile.total_conversations >= 3:
                profile.expertise_level = "intermediate"
            elif profile.total_conversations >= 10:
                profile.expertise_level = "intermediate"  # Experience over time

class PortfolioEvolutionTracker:
    """Track and analyze portfolio evolution over time"""
    
    def add_portfolio_snapshot(self, portfolio_history: List[Dict], portfolio_context: Dict):
        """Add portfolio snapshot to evolution history"""
        snapshot = {
            "timestamp": datetime.now(),
            "portfolio_data": portfolio_context.copy(),
            "snapshot_id": str(uuid.uuid4())
        }
        
        portfolio_history.append(snapshot)
        
        # Keep last 50 snapshots
        if len(portfolio_history) > 50:
            portfolio_history[:] = portfolio_history[-50:]
    
    def analyze_evolution(self, portfolio_history: List[Dict]) -> Dict:
        """Analyze portfolio changes over time"""
        if len(portfolio_history) < 2:
            return {"status": "insufficient_data", "snapshots": len(portfolio_history)}
        
        latest = portfolio_history[-1]["portfolio_data"]
        previous = portfolio_history[-2]["portfolio_data"]
        
        insights = {
            "status": "analysis_complete",
            "snapshots_analyzed": len(portfolio_history),
            "time_span_days": self._calculate_time_span(portfolio_history),
            "value_change": self._calculate_value_change(latest, previous),
            "risk_profile_change": self._calculate_risk_change(latest, previous),
            "allocation_analysis": self._analyze_allocation_changes(latest, previous),
            "trends": self._identify_trends(portfolio_history)
        }
        
        return insights
    
    def _calculate_time_span(self, portfolio_history: List[Dict]) -> int:
        """Calculate time span of portfolio history"""
        if len(portfolio_history) < 2:
            return 0
        
        earliest = portfolio_history[0]["timestamp"]
        latest = portfolio_history[-1]["timestamp"]
        
        if isinstance(earliest, str):
            earliest = datetime.fromisoformat(earliest)
        if isinstance(latest, str):
            latest = datetime.fromisoformat(latest)
        
        return (latest - earliest).days
    
    def _calculate_value_change(self, latest: Dict, previous: Dict) -> Dict:
        """Calculate portfolio value change"""
        latest_value = self._extract_numeric_value(latest.get("total_value", 0))
        previous_value = self._extract_numeric_value(previous.get("total_value", 0))
        
        if previous_value > 0:
            change_percent = (latest_value - previous_value) / previous_value * 100
            return {
                "value_change_percent": round(change_percent, 2),
                "value_change_absolute": latest_value - previous_value,
                "current_value": latest_value,
                "previous_value": previous_value
            }
        return {"status": "insufficient_value_data"}
    
    def _extract_numeric_value(self, value) -> float:
        """Extract numeric value from various formats"""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            # Remove $ and commas, convert to float
            numeric_str = value.replace("$", "").replace(",", "")
            try:
                return float(numeric_str)
            except ValueError:
                return 0.0
        return 0.0
    
    def _calculate_risk_change(self, latest: Dict, previous: Dict) -> Dict:
        """Calculate risk profile changes"""
        latest_risk = latest.get("riskLevel", "MODERATE")
        previous_risk = previous.get("riskLevel", "MODERATE")
        
        risk_levels = {"LOW": 1, "MODERATE": 2, "HIGH": 3}
        
        return {
            "risk_level_changed": latest_risk != previous_risk,
            "current_risk": latest_risk,
            "previous_risk": previous_risk,
            "risk_direction": "increased" if risk_levels.get(latest_risk, 2) > risk_levels.get(previous_risk, 2) else "decreased" if risk_levels.get(latest_risk, 2) < risk_levels.get(previous_risk, 2) else "unchanged"
        }
    
    def _analyze_allocation_changes(self, latest: Dict, previous: Dict) -> Dict:
        """Analyze allocation changes between snapshots"""
        # Simplified allocation analysis
        latest_holdings = latest.get("holdings", [])
        previous_holdings = previous.get("holdings", [])
        
        if not latest_holdings or not previous_holdings:
            return {"status": "no_holdings_data"}
        
        return {
            "holdings_count_change": len(latest_holdings) - len(previous_holdings),
            "new_positions": len(latest_holdings) - len(previous_holdings) if len(latest_holdings) > len(previous_holdings) else 0,
            "closed_positions": len(previous_holdings) - len(latest_holdings) if len(previous_holdings) > len(latest_holdings) else 0
        }
    
    def _identify_trends(self, portfolio_history: List[Dict]) -> Dict:
        """Identify trends in portfolio evolution"""
        if len(portfolio_history) < 3:
            return {"status": "insufficient_data_for_trends"}
        
        # Analyze value trend over time
        values = []
        for snapshot in portfolio_history[-5:]:  # Last 5 snapshots
            value = self._extract_numeric_value(snapshot["portfolio_data"].get("total_value", 0))
            if value > 0:
                values.append(value)
        
        if len(values) < 2:
            return {"status": "insufficient_value_data"}
        
        trend = "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "stable"
        
        return {
            "value_trend": trend,
            "values_analyzed": len(values),
            "trend_strength": abs(values[-1] - values[0]) / values[0] if values[0] > 0 else 0
        }

# Memory Store for centralized memory management
class MemoryStore:
    """Centralized memory storage and retrieval"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.conversations: Dict[str, EnhancedConversationMemory] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        self.storage_path = Path(storage_path) if storage_path else Path("memory_store")
        self.storage_path.mkdir(exist_ok=True)
        
        logger.info(f"Memory store initialized with storage at: {self.storage_path}")
    
    def get_conversation(self, conversation_id: str, user_id: str) -> EnhancedConversationMemory:
        """Get or create conversation memory"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = EnhancedConversationMemory(
                conversation_id, user_id
            )
        return self.conversations[conversation_id]
    
    def get_user_conversations(self, user_id: str) -> List[EnhancedConversationMemory]:
        """Get all conversations for a user"""
        return [conv for conv in self.conversations.values() if conv.user_id == user_id]
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile across all conversations"""
        user_conversations = self.get_user_conversations(user_id)
        if not user_conversations:
            return None
        
        # Aggregate profile from all conversations
        for conv in user_conversations:
            if conv.user_profile:
                return conv.user_profile
        
        return None
    
    def save_conversation(self, conversation_id: str):
        """Save conversation to storage"""
        if conversation_id in self.conversations:
            try:
                conv = self.conversations[conversation_id]
                filepath = self.storage_path / f"{conversation_id}.json"
                
                # Convert to serializable format (simplified)
                data = {
                    "conversation_id": conv.conversation_id,
                    "user_id": conv.user_id,
                    "created_at": conv.created_at.isoformat(),
                    "updated_at": conv.updated_at.isoformat(),
                    "turn_count": len(conv.turns),
                    "user_profile": self._serialize_user_profile(conv.user_profile) if conv.user_profile else None
                }
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"Saved conversation {conversation_id} to storage")
                
            except Exception as e:
                logger.error(f"Failed to save conversation {conversation_id}: {e}")
    
    def _serialize_user_profile(self, profile: UserProfile) -> Dict:
        """Serialize user profile for storage"""
        return {
            "user_id": profile.user_id,
            "agent_satisfaction_scores": profile.agent_satisfaction_scores,
            "query_complexity_preference": profile.query_complexity_preference,
            "detail_level_preference": profile.detail_level_preference,
            "risk_tolerance_indicated": profile.risk_tolerance_indicated,
            "expertise_level": profile.expertise_level,
            "preferred_collaboration_style": profile.preferred_collaboration_style,
            "total_conversations": profile.total_conversations,
            "total_collaborations": profile.total_collaborations,
            "last_active": profile.last_active.isoformat()
        }
    
    def get_memory_analytics(self) -> Dict:
        """Get analytics about memory usage and performance"""
        total_conversations = len(self.conversations)
        total_turns = sum(len(conv.turns) for conv in self.conversations.values())
        users_with_profiles = sum(1 for conv in self.conversations.values() if conv.user_profile)
        
        return {
            "total_conversations": total_conversations,
            "total_turns": total_turns,
            "avg_turns_per_conversation": total_turns / total_conversations if total_conversations else 0,
            "users_with_profiles": users_with_profiles,
            "embeddings_enabled": EMBEDDINGS_AVAILABLE,
            "storage_path": str(self.storage_path)
        }

# Factory function for easy integration
def create_advanced_memory_system(storage_path: Optional[str] = None) -> MemoryStore:
    """Create and return an advanced memory system"""
    return MemoryStore(storage_path)