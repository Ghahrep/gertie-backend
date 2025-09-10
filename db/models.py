# Task 1.2: Create Clean Database Schema
# File: db/models.py

from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Boolean, JSON
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

# Utility functions for portfolio operations
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

# Export all models for imports
__all__ = [
    "Base", "User", "Asset", "Portfolio", "Holding", "Alert",
    "calculate_portfolio_summary", "create_sample_portfolio_data"
]