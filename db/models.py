# Enhanced Database Models - Complete Schema
# File: db/models.py

from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Boolean, JSON, Text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from typing import Dict, Any, List

Base = declarative_base()

class User(Base):
    """Core user model - authentication and preferences"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Authentication and authorization fields
    is_active = Column(Boolean, default=True, nullable=False)
    role = Column(String, default="user", nullable=False)  # "user", "admin", "manager"
    last_login = Column(DateTime, nullable=True)
    
    # Simple JSON preferences instead of complex separate tables
    preferences = Column(JSON, default=lambda: {
        "risk_tolerance": "moderate",
        "default_analysis_depth": "standard", 
        "notification_settings": {"email": True, "webapp": True},
        "ui_preferences": {
            "theme": "light",
            "default_currency": "USD",
            "decimal_places": 2
        }
    })
    
    # Relationships
    portfolios = relationship("Portfolio", back_populates="owner", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="user", cascade="all, delete-orphan")
    conversation_turns = relationship("ConversationTurn", back_populates="user", cascade="all, delete-orphan")
    user_profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    proactive_insights = relationship("ProactiveInsight", back_populates="user", cascade="all, delete-orphan")
    enhanced_conversations = relationship("EnhancedConversation", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', role='{self.role}', active={self.is_active})>"
    
    @property
    def is_admin(self) -> bool:
        """Check if user has admin privileges"""
        return self.role == "admin"
    
    @property
    def display_name(self) -> str:
        """Get display name for user (email prefix)"""
        return self.email.split('@')[0] if self.email else f"User {self.id}"
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
    
    def get_preference(self, key: str, default=None):
        """Get user preference by key"""
        if not self.preferences:
            return default
        return self.preferences.get(key, default)
    
    def set_preference(self, key: str, value):
        """Set user preference"""
        if not self.preferences:
            self.preferences = {}
        self.preferences[key] = value
    
    def to_dict(self) -> dict:
        """Convert user to dictionary for API responses"""
        return {
            "id": self.id,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "display_name": self.display_name,
            "preferences": self.preferences or {}
        }

class Asset(Base):
    """Asset master data - stocks, ETFs, bonds, crypto"""
    __tablename__ = "assets"
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(20), unique=True, index=True, nullable=False)
    name = Column(String(200))
    asset_type = Column(String(50), default="stock")  # stock, etf, bond, crypto
    sector = Column(String(100))
    
    # Simple market data cache (updated periodically)
    current_price = Column(Float)
    last_updated = Column(DateTime)
    
    # Relationships
    holdings = relationship("Holding", back_populates="asset")

class Portfolio(Base):
    """User portfolios - can have multiple per user"""
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Portfolio settings
    is_active = Column(Boolean, default=True)
    currency = Column(String(3), default="USD")
    
    # Relationships
    owner = relationship("User", back_populates="portfolios")
    holdings = relationship("Holding", back_populates="portfolio", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="portfolio")
    transactions = relationship("Transaction", back_populates="portfolio")
    snapshots = relationship("PortfolioSnapshot", back_populates="portfolio", cascade="all, delete-orphan")
    enhanced_conversations = relationship("EnhancedConversation", back_populates="portfolio", cascade="all, delete-orphan")
    proactive_insights = relationship("ProactiveInsight", back_populates="portfolio")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize portfolio for API responses"""
        return {
            'id': self.id,
            'name': self.name,
            'user_id': self.user_id,
            'description': self.description,
            'is_active': self.is_active,
            'currency': self.currency,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'holdings_count': len(self.holdings) if self.holdings else 0
        }

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    asset_id = Column(Integer, ForeignKey("assets.id"))
    transaction_type = Column(String)  # "buy" or "sell"
    shares = Column(Float)
    price_per_share = Column(Float)
    total_amount = Column(Float)
    transaction_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    portfolio = relationship("Portfolio", back_populates="transactions")
    asset = relationship("Asset")

class Holding(Base):
    """Individual positions within portfolios"""
    __tablename__ = "holdings"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    
    # Position data
    shares = Column(Float, nullable=False)
    purchase_price = Column(Float, nullable=False)
    purchase_date = Column(DateTime, default=datetime.utcnow)
    
    # Optional: cost basis tracking for tax purposes
    cost_basis = Column(Float)  # Total cost including fees
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="holdings")
    asset = relationship("Asset", back_populates="holdings")
    
    def current_value(self) -> float:
        """Calculate current market value"""
        if self.asset and self.asset.current_price:
            return self.shares * self.asset.current_price
        return self.shares * self.purchase_price  # Fallback to purchase price
    
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L"""
        current_val = self.current_value()
        cost = self.cost_basis or (self.shares * self.purchase_price)
        return current_val - cost
    
    def unrealized_pnl_percent(self) -> float:
        """Calculate unrealized P&L percentage"""
        cost = self.cost_basis or (self.shares * self.purchase_price)
        if cost == 0:
            return 0.0
        return (self.unrealized_pnl() / cost) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses"""
        return {
            "id": self.id,
            "ticker": self.asset.ticker if self.asset else None,
            "name": self.asset.name if self.asset else None,
            "shares": self.shares,
            "purchase_price": self.purchase_price,
            "current_price": self.asset.current_price if self.asset else self.purchase_price,
            "current_value": self.current_value(),
            "unrealized_pnl": self.unrealized_pnl(),
            "unrealized_pnl_percent": self.unrealized_pnl_percent(),
            "purchase_date": self.purchase_date.isoformat() if self.purchase_date else None,
            "asset_type": self.asset.asset_type if self.asset else "unknown",
            "sector": self.asset.sector if self.asset else None
        }

class Alert(Base):
    """Simple alerting system for price/risk thresholds"""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=True)  # Optional portfolio-specific
    
    # Alert configuration - stored as JSON for flexibility
    alert_type = Column(String(50), nullable=False)  # "price_change", "risk_threshold", "portfolio_value"
    condition = Column(JSON, nullable=False)  # {"metric": "price", "operator": ">", "value": 150.0, "ticker": "AAPL"}
    
    # Alert state
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_triggered = Column(DateTime)
    trigger_count = Column(Integer, default=0)
    
    # Notification preferences
    notification_channels = Column(JSON, default=lambda: ["webapp"])  # ["email", "webapp", "sms"]
    
    # Relationships
    user = relationship("User", back_populates="alerts")
    portfolio = relationship("Portfolio", back_populates="alerts")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses"""
        return {
            "id": self.id,
            "alert_type": self.alert_type,
            "condition": self.condition,
            "is_active": self.is_active,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
            "trigger_count": self.trigger_count,
            "notification_channels": self.notification_channels,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class ChatConversation(Base):
    """Chat conversation sessions for Investment Committee"""
    __tablename__ = "chat_conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    title = Column(String(200), nullable=False)
    
    # Conversation metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    last_message_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Conversation summary and context
    summary = Column(String(500))  # Auto-generated summary
    total_messages = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User")
    portfolio = relationship("Portfolio")
    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "portfolio_id": self.portfolio_id,
            "title": self.title,
            "summary": self.summary,
            "total_messages": self.total_messages,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
            "is_active": self.is_active
        }

class ChatMessage(Base):
    """Individual messages within chat conversations"""
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("chat_conversations.id"), nullable=False)
    
    # Message content
    role = Column(String(20), nullable=False)  # "user", "assistant", "system"
    content = Column(String(10000), nullable=False)
    
    # Specialist routing information
    specialist_id = Column(String(50))  # Which specialist handled this
    routing_confidence = Column(Float)  # Routing confidence score
    
    # Message metadata
    timestamp = Column(DateTime, default=datetime.utcnow)
    tokens_used = Column(Integer, default=0)
    
    # Analysis context - store portfolio state at time of message
    portfolio_context = Column(JSON)  # Portfolio value, risk metrics, etc.
    
    # Relationships
    conversation = relationship("ChatConversation", back_populates="messages")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "specialist_id": self.specialist_id,
            "routing_confidence": self.routing_confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "tokens_used": self.tokens_used,
            "portfolio_context": self.portfolio_context
        }

# ============================================================================
# ADVANCED AI MODELS FOR WEEK 1 & 2 CAPABILITIES
# ============================================================================

# 1. CONVERSATION MEMORY SYSTEM (Week 1 Advanced Memory)

class ConversationTurn(Base):
    """Individual conversation turns for advanced memory system"""
    __tablename__ = "conversation_turns"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    conversation_id = Column(String(100), nullable=False)  # UUID for conversation grouping
    
    # Turn content
    user_query = Column(Text, nullable=False)
    agent_response = Column(Text, nullable=False)
    specialist = Column(String(50), nullable=False)  # quantitative_analyst, portfolio_manager, etc.
    
    # Analysis metadata
    confidence = Column(Float, default=0.0)
    risk_score = Column(Integer, default=50)
    collaboration_involved = Column(Boolean, default=False)
    secondary_specialists = Column(JSON, default=list)  # List of additional specialists consulted
    
    # Context and embeddings
    portfolio_context = Column(JSON)  # Portfolio state at time of turn
    semantic_embedding = Column(JSON)  # Stored as JSON array for similarity search
    
    # User feedback and learning
    user_satisfaction = Column(Float, nullable=True)  # 0.0-1.0 rating
    user_engagement_time = Column(Integer, default=0)  # Seconds spent on response
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="conversation_turns")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "user_query": self.user_query,
            "agent_response": self.agent_response,
            "specialist": self.specialist,
            "confidence": self.confidence,
            "risk_score": self.risk_score,
            "collaboration_involved": self.collaboration_involved,
            "secondary_specialists": self.secondary_specialists,
            "user_satisfaction": self.user_satisfaction,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class UserProfile(Base):
    """User learning profile for personalized routing"""
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    
    # Learning metrics
    expertise_level = Column(String(20), default="beginner")  # beginner, intermediate, advanced, expert
    complexity_preference = Column(Float, default=0.5)  # 0.0 (simple) to 1.0 (complex)
    collaboration_preference = Column(Float, default=0.5)  # Preference for multi-agent responses
    
    # Agent satisfaction scores (JSON dict)
    agent_satisfaction_scores = Column(JSON, default=dict)  # {"quantitative_analyst": 0.8, ...}
    
    # Usage patterns
    total_conversations = Column(Integer, default=0)
    total_turns = Column(Integer, default=0)
    avg_session_length = Column(Float, default=0.0)
    
    # Preference learning
    preferred_response_length = Column(String(20), default="medium")  # short, medium, long
    preferred_detail_level = Column(String(20), default="standard")  # basic, standard, detailed
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="user_profile")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "expertise_level": self.expertise_level,
            "complexity_preference": self.complexity_preference,
            "collaboration_preference": self.collaboration_preference,
            "agent_satisfaction_scores": self.agent_satisfaction_scores or {},
            "total_conversations": self.total_conversations,
            "total_turns": self.total_turns,
            "preferred_response_length": self.preferred_response_length,
            "preferred_detail_level": self.preferred_detail_level,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

# 2. PROACTIVE INSIGHTS SYSTEM (Week 2)

class ProactiveInsight(Base):
    """Proactive insights generated by the system"""
    __tablename__ = "proactive_insights"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=True)
    
    # Insight identification
    insight_id = Column(String(100), unique=True, nullable=False)  # Unique insight identifier
    insight_type = Column(String(50), nullable=False)  # portfolio_drift, behavioral_pattern, market_opportunity
    priority = Column(String(20), nullable=False)  # critical, high, medium, low, info
    
    # Content
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    recommendations = Column(JSON, default=list)  # List of recommendation strings
    conversation_starters = Column(JSON, default=list)  # List of conversation starter strings
    
    # Metadata and analysis data
    data = Column(JSON, default=dict)  # Analysis data, metrics, etc.
    
    # Lifecycle
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # User engagement tracking
    view_count = Column(Integer, default=0)
    click_count = Column(Integer, default=0)
    dismiss_count = Column(Integer, default=0)
    action_taken = Column(Boolean, default=False)
    
    # Relationships
    user = relationship("User", back_populates="proactive_insights")
    portfolio = relationship("Portfolio", back_populates="proactive_insights")
    engagements = relationship("InsightEngagement", back_populates="insight", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.insight_id,
            "type": self.insight_type,
            "priority": self.priority,
            "title": self.title,
            "description": self.description,
            "recommendations": self.recommendations or [],
            "conversation_starters": self.conversation_starters or [],
            "data": self.data or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "engagement_stats": {
                "view_count": self.view_count,
                "click_count": self.click_count,
                "action_taken": self.action_taken
            }
        }

class InsightEngagement(Base):
    """Track user engagement with proactive insights"""
    __tablename__ = "insight_engagements"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    insight_id = Column(Integer, ForeignKey("proactive_insights.id"), nullable=False)
    
    # Engagement details
    engagement_type = Column(String(50), nullable=False)  # viewed, clicked, dismissed, acted_upon
    engagement_data = Column(JSON, default=dict)  # Additional context
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    insight = relationship("ProactiveInsight", back_populates="engagements")

# 3. PORTFOLIO EVOLUTION TRACKING

class PortfolioSnapshot(Base):
    """Historical snapshots of portfolio state for drift detection"""
    __tablename__ = "portfolio_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    
    # Snapshot data
    total_value = Column(Float, nullable=False)
    holdings_data = Column(JSON, nullable=False)  # Complete holdings at time of snapshot
    allocation_weights = Column(JSON, default=dict)  # Asset allocation percentages
    
    # Risk metrics
    risk_score = Column(Float, default=0.0)
    concentration_ratio = Column(Float, default=0.0)  # Herfindahl index
    volatility_estimate = Column(Float, default=0.0)
    
    # Market context
    market_conditions = Column(JSON, default=dict)  # Market state at snapshot time
    
    # Metadata
    snapshot_type = Column(String(50), default="scheduled")  # scheduled, triggered, manual
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="snapshots")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "portfolio_id": self.portfolio_id,
            "total_value": self.total_value,
            "holdings_data": self.holdings_data,
            "allocation_weights": self.allocation_weights,
            "risk_score": self.risk_score,
            "concentration_ratio": self.concentration_ratio,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

# 4. ENHANCED CONVERSATION TRACKING (Better than existing ChatMessage)

class EnhancedConversation(Base):
    """Enhanced conversation tracking with memory features"""
    __tablename__ = "enhanced_conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    
    # Conversation identification
    conversation_id = Column(String(100), unique=True, nullable=False)  # UUID
    session_id = Column(String(100), nullable=True)  # Frontend session grouping
    
    # Content
    user_query = Column(Text, nullable=False)
    agent_response = Column(Text, nullable=False)
    
    # Routing and analysis
    specialist = Column(String(50), nullable=False)
    routing_confidence = Column(Float, default=0.0)
    analysis_confidence = Column(Float, default=0.0)
    
    # Enhanced features
    collaboration_used = Column(Boolean, default=False)
    specialists_consulted = Column(JSON, default=list)
    tools_used = Column(JSON, default=list)
    
    # Context and insights
    portfolio_context = Column(JSON, default=dict)
    proactive_insights_count = Column(Integer, default=0)
    related_insights_count = Column(Integer, default=0)
    
    # Performance metrics
    response_time_ms = Column(Integer, default=0)
    tokens_used = Column(Integer, default=0)
    
    # User feedback
    user_rating = Column(Float, nullable=True)  # 1-5 star rating
    user_feedback_text = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="enhanced_conversations")
    portfolio = relationship("Portfolio", back_populates="enhanced_conversations")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "user_query": self.user_query,
            "agent_response": self.agent_response,
            "specialist": self.specialist,
            "routing_confidence": self.routing_confidence,
            "analysis_confidence": self.analysis_confidence,
            "collaboration_used": self.collaboration_used,
            "specialists_consulted": self.specialists_consulted or [],
            "tools_used": self.tools_used or [],
            "proactive_insights_count": self.proactive_insights_count,
            "user_rating": self.user_rating,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

# 5. SYSTEM ANALYTICS AND PERFORMANCE

class SystemMetrics(Base):
    """System performance and usage metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Metric identification
    metric_type = Column(String(50), nullable=False)  # routing_accuracy, response_time, user_satisfaction
    metric_name = Column(String(100), nullable=False)
    
    # Values
    value = Column(Float, nullable=False)
    count = Column(Integer, default=1)
    
    # Context
    context = Column(JSON, default=dict)  # Additional context data
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Optional user-specific
    
    # Timestamps
    recorded_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User")

# ============================================================================
# UTILITY FUNCTIONS FOR DATABASE OPERATIONS
# ============================================================================

def calculate_portfolio_summary(portfolio: Portfolio) -> Dict[str, Any]:
    """Calculate portfolio summary with current values"""
    if not portfolio.holdings:
        return {
            "id": portfolio.id,
            "name": portfolio.name,
            "total_value": 0.0,
            "total_cost": 0.0,
            "total_pnl": 0.0,
            "total_pnl_percent": 0.0,
            "holdings_count": 0,
            "holdings": []
        }
    
    total_value = sum(holding.current_value() for holding in portfolio.holdings)
    total_cost = sum(holding.cost_basis or (holding.shares * holding.purchase_price) 
                    for holding in portfolio.holdings)
    total_pnl = total_value - total_cost
    
    return {
        "id": portfolio.id,
        "name": portfolio.name,
        "total_value": round(total_value, 2),
        "total_cost": round(total_cost, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_percent": round((total_pnl / total_cost * 100), 2) if total_cost > 0 else 0,
        "holdings_count": len(portfolio.holdings),
        "holdings": [holding.to_dict() for holding in portfolio.holdings],
        "top_holdings": sorted(
            [holding.to_dict() for holding in portfolio.holdings],
            key=lambda x: x["current_value"],
            reverse=True
        )[:5]  # Top 5 holdings by value
    }

def create_sample_portfolio_data(user_id: int) -> Dict[str, List[Dict]]:
    """Create sample data for development/testing"""
    return {
        "assets": [
            {"ticker": "AAPL", "name": "Apple Inc.", "asset_type": "stock", "sector": "Technology", "current_price": 175.0},
            {"ticker": "MSFT", "name": "Microsoft Corporation", "asset_type": "stock", "sector": "Technology", "current_price": 340.0},
            {"ticker": "TSLA", "name": "Tesla Inc.", "asset_type": "stock", "sector": "Consumer Discretionary", "current_price": 250.0},
            {"ticker": "SPY", "name": "SPDR S&P 500 ETF", "asset_type": "etf", "sector": "Diversified", "current_price": 450.0},
            {"ticker": "QQQ", "name": "Invesco QQQ ETF", "asset_type": "etf", "sector": "Technology", "current_price": 380.0},
            {"ticker": "NVDA", "name": "NVIDIA Corporation", "asset_type": "stock", "sector": "Technology", "current_price": 420.0},
            {"ticker": "GOOGL", "name": "Alphabet Inc.", "asset_type": "stock", "sector": "Technology", "current_price": 135.0},
            {"ticker": "AMZN", "name": "Amazon.com Inc.", "asset_type": "stock", "sector": "Consumer Discretionary", "current_price": 145.0}
        ],
        "portfolios": [
            {"user_id": user_id, "name": "Main Portfolio", "description": "Primary investment portfolio"},
            {"user_id": user_id, "name": "Growth Portfolio", "description": "High-growth technology focus"}
        ],
        "sample_holdings": [
            # Main Portfolio holdings
            {"portfolio_name": "Main Portfolio", "ticker": "AAPL", "shares": 100, "purchase_price": 150.0},
            {"portfolio_name": "Main Portfolio", "ticker": "MSFT", "shares": 50, "purchase_price": 300.0},
            {"portfolio_name": "Main Portfolio", "ticker": "SPY", "shares": 200, "purchase_price": 420.0},
            {"portfolio_name": "Main Portfolio", "ticker": "QQQ", "shares": 100, "purchase_price": 350.0},
            # Growth Portfolio holdings
            {"portfolio_name": "Growth Portfolio", "ticker": "TSLA", "shares": 75, "purchase_price": 200.0},
            {"portfolio_name": "Growth Portfolio", "ticker": "NVDA", "shares": 25, "purchase_price": 380.0},
            {"portfolio_name": "Growth Portfolio", "ticker": "GOOGL", "shares": 50, "purchase_price": 120.0}
        ]
    }

# ============================================================================
# HELPER FUNCTIONS FOR NEW AI MODELS
# ============================================================================

def create_conversation_turn(user_id: int, conversation_id: str, user_query: str, 
                           agent_response: str, specialist: str, **kwargs) -> ConversationTurn:
    """Helper to create conversation turns"""
    return ConversationTurn(
        user_id=user_id,
        conversation_id=conversation_id,
        user_query=user_query,
        agent_response=agent_response,
        specialist=specialist,
        confidence=kwargs.get('confidence', 0.85),
        risk_score=kwargs.get('risk_score', 50),
        collaboration_involved=kwargs.get('collaboration_involved', False),
        secondary_specialists=kwargs.get('secondary_specialists', []),
        portfolio_context=kwargs.get('portfolio_context', {})
    )

def get_user_profile_or_create(db_session, user_id: int) -> UserProfile:
    """Get or create user profile for learning"""
    profile = db_session.query(UserProfile).filter(UserProfile.user_id == user_id).first()
    if not profile:
        profile = UserProfile(user_id=user_id)
        db_session.add(profile)
        db_session.commit()
    return profile

def create_proactive_insight(user_id: int, insight_type: str, priority: str,
                           title: str, description: str, **kwargs) -> ProactiveInsight:
    """Helper to create proactive insights"""
    insight_id = f"{insight_type}_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return ProactiveInsight(
        user_id=user_id,
        portfolio_id=kwargs.get('portfolio_id'),
        insight_id=insight_id,
        insight_type=insight_type,
        priority=priority,
        title=title,
        description=description,
        recommendations=kwargs.get('recommendations', []),
        conversation_starters=kwargs.get('conversation_starters', []),
        data=kwargs.get('data', {}),
        expires_at=kwargs.get('expires_at')
    )

def record_system_metric(metric_type: str, metric_name: str, value: float, 
                        context: dict = None, user_id: int = None) -> SystemMetrics:
    """Helper to record system metrics"""
    return SystemMetrics(
        metric_type=metric_type,
        metric_name=metric_name,
        value=value,
        context=context or {},
        user_id=user_id
    )

# Export all models for imports
__all__ = [
    "Base", "User", "Asset", "Portfolio", "Holding", "Alert", "Transaction",
    "ChatConversation", "ChatMessage",
    # Advanced AI models
    "ConversationTurn", "UserProfile", "ProactiveInsight", "InsightEngagement",
    "PortfolioSnapshot", "EnhancedConversation", "SystemMetrics",
    # Helper functions
    "calculate_portfolio_summary", "create_sample_portfolio_data",
    "create_conversation_turn", "get_user_profile_or_create", 
    "create_proactive_insight", "record_system_metric"
]