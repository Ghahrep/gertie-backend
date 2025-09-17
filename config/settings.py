# config/settings.py - Enhanced Configuration with Portfolio Signature Settings
"""
Configuration Settings for Clean Financial Platform
==================================================
Enhanced configuration with portfolio signature and caching settings.
"""

import os
from typing import Optional, List
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application Configuration
    app_name: str = "Clean Financial Platform"
    app_version: str = "2.0.0"
    debug: bool = False
    environment: str = "development"
    
    # Database Configuration
    database_url: str = "sqlite:///./clean_financial.db"
    database_echo: bool = False
    
    # Security Configuration
    secret_key: str = "your-secret-key-change-in-production"
    access_token_expire_minutes: int = 30
    algorithm: str = "HS256"
    
    # Portfolio Signature Configuration
    signature_cache_enabled: bool = True
    signature_cache_ttl: int = 300  # 5 minutes
    signature_force_refresh_threshold: int = 10  # Max concurrent force refreshes
    signature_batch_size: int = 50  # Max portfolios in batch request
    signature_timeout_seconds: int = 30  # Analysis timeout
    
    # Caching Configuration
    cache_enabled: bool = True
    cache_type: str = "memory"  # "memory", "redis", "hybrid"
    cache_default_ttl: int = 300
    cache_max_memory_entries: int = 1000
    redis_url: Optional[str] = None
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    # Market Data Configuration
    market_data_provider: str = "yahoo"  # "yahoo", "alpha_vantage", "polygon"
    market_data_cache_ttl: int = 60  # 1 minute for real-time data
    market_data_timeout: int = 10
    alpha_vantage_api_key: Optional[str] = None
    polygon_api_key: Optional[str] = None
    
    # Risk Analysis Configuration
    risk_analysis_enabled: bool = True
    risk_lookback_days: int = 252  # 1 year of trading days
    risk_confidence_level: float = 0.95  # 95% confidence for VaR/CVaR
    risk_monte_carlo_simulations: int = 10000
    risk_garch_max_lag: int = 5
    
    # Performance Configuration
    max_workers: int = 4
    request_timeout: int = 30
    max_request_size: int = 1024 * 1024  # 1MB
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Monitoring and Logging
    log_level: str = "INFO"
    log_format: str = "json"
    enable_metrics: bool = True
    sentry_dsn: Optional[str] = None
    
    # WebSocket Configuration
    websocket_enabled: bool = True
    websocket_heartbeat_interval: int = 30
    websocket_max_connections: int = 100
    
    # API Configuration
    api_prefix: str = "/api/v1"
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_allow_headers: List[str] = ["*"]
    
    # Documentation Configuration
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"
    
    # File Upload Configuration
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    allowed_upload_types: List[str] = [".csv", ".xlsx", ".json"]
    upload_directory: str = "./uploads"
    
    # Email Configuration (for alerts)
    email_enabled: bool = False
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True
    
    # Alert Configuration
    alerts_enabled: bool = True
    alert_check_interval: int = 300  # 5 minutes
    max_alerts_per_portfolio: int = 10
    alert_cooldown_minutes: int = 60
    
    # Backup Configuration
    backup_enabled: bool = False
    backup_interval_hours: int = 24
    backup_retention_days: int = 30
    backup_directory: str = "./backups"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Environment variable prefixes
        env_prefix = ""
        
        # Field aliases for environment variables
        fields = {
            "database_url": {"env": "DATABASE_URL"},
            "secret_key": {"env": "SECRET_KEY"},
            "redis_url": {"env": "REDIS_URL"},
            "redis_password": {"env": "REDIS_PASSWORD"},
            "alpha_vantage_api_key": {"env": "ALPHA_VANTAGE_API_KEY"},
            "polygon_api_key": {"env": "POLYGON_API_KEY"},
            "sentry_dsn": {"env": "SENTRY_DSN"},
            "smtp_host": {"env": "SMTP_HOST"},
            "smtp_username": {"env": "SMTP_USERNAME"},
            "smtp_password": {"env": "SMTP_PASSWORD"}
        }

class PortfolioSignatureSettings:
    """Portfolio signature specific settings and validation"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    @property
    def normalization_config(self) -> dict:
        """Normalization configuration for risk metrics"""
        return {
            "risk_score": {
                "min": 0.0,
                "max": 100.0,
                "method": "linear"
            },
            "tail_risk": {
                "min": 0.0,
                "max": 1.0,
                "cvar_range": (-0.5, 0.0)  # CVaR typically ranges from -50% to 0%
            },
            "correlation": {
                "min": 0.0,
                "max": 1.0,
                "method": "average_off_diagonal"
            },
            "volatility_forecast": {
                "min": 0.0,
                "max": 100.0,
                "scaling_factor": 100.0
            },
            "complexity": {
                "min": 0.0,
                "max": 1.0,
                "multifractal_range": (0.1, 2.0)
            },
            "concentration": {
                "min": 0.0,
                "max": 1.0,
                "method": "herfindahl_index"
            }
        }
    
    @property
    def risk_level_thresholds(self) -> dict:
        """Risk level classification thresholds"""
        return {
            "low": {"risk_score": 30, "tail_risk": 0.3},
            "medium": {"risk_score": 60, "tail_risk": 0.6},
            "high": {"risk_score": 80, "tail_risk": 0.8},
            "critical": {"risk_score": 90, "tail_risk": 0.9}
        }
    
    @property
    def alert_thresholds(self) -> dict:
        """Alert thresholds for risk metrics"""
        return {
            "tail_risk_high": 0.8,
            "correlation_spike": 0.9,
            "volatility_surge": 80.0,
            "concentration_risk": 0.7,
            "risk_score_critical": 85.0
        }
    
    def validate_signature_data(self, signature: dict) -> bool:
        """Validate portfolio signature data"""
        required_fields = [
            'risk_score', 'tail_risk', 'correlation', 'volatility_forecast',
            'complexity', 'concentration', 'risk_level', 'data_source',
            'last_updated', 'portfolio_id'
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in signature:
                return False
        
        # Check value ranges
        config = self.normalization_config
        
        if not (config['risk_score']['min'] <= signature['risk_score'] <= config['risk_score']['max']):
            return False
        
        if not (config['tail_risk']['min'] <= signature['tail_risk'] <= config['tail_risk']['max']):
            return False
        
        if not (config['correlation']['min'] <= signature['correlation'] <= config['correlation']['max']):
            return False
        
        return True

class CacheSettings:
    """Cache configuration and utilities"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    @property
    def cache_config(self) -> dict:
        """Cache configuration dictionary"""
        return {
            "default_ttl": self.settings.cache_default_ttl,
            "max_memory_entries": self.settings.cache_max_memory_entries,
            "redis_url": self.settings.redis_url,
            "redis_password": self.settings.redis_password,
            "redis_db": self.settings.redis_db,
            "enable_redis": self.settings.cache_type in ["redis", "hybrid"],
            "enable_memory": self.settings.cache_type in ["memory", "hybrid"]
        }
    
    def get_cache_key_prefix(self, key_type: str) -> str:
        """Generate cache key prefix"""
        prefixes = {
            "portfolio_signature": "ps",
            "market_data": "md",
            "risk_metrics": "rm",
            "user_session": "us"
        }
        
        env_prefix = self.settings.environment[:3].lower()
        return f"{env_prefix}:{prefixes.get(key_type, 'gen')}"

# Global settings instance
settings = Settings()

# Specialized settings
portfolio_signature_settings = PortfolioSignatureSettings(settings)
cache_settings = CacheSettings(settings)

def get_settings() -> Settings:
    """Get application settings"""
    return settings

def get_portfolio_signature_settings() -> PortfolioSignatureSettings:
    """Get portfolio signature settings"""
    return portfolio_signature_settings

def get_cache_settings() -> CacheSettings:
    """Get cache settings"""
    return cache_settings

# Development overrides
if settings.environment == "development":
    settings.debug = True
    settings.database_echo = True
    settings.log_level = "DEBUG"

# Production security checks
if settings.environment == "production":
    if settings.secret_key == "your-secret-key-change-in-production":
        raise ValueError("Secret key must be changed in production")
    
    if not settings.database_url.startswith(("postgresql://", "mysql://")):
        print("WARNING: Using SQLite in production is not recommended")
    
    # Ensure HTTPS in production
    settings.cors_origins = [origin for origin in settings.cors_origins if origin.startswith("https://")]

# Environment-specific configurations
DATABASE_CONFIGS = {
    "development": {
        "pool_size": 5,
        "max_overflow": 10,
        "pool_timeout": 30,
        "pool_recycle": 3600
    },
    "testing": {
        "pool_size": 1,
        "max_overflow": 0,
        "pool_timeout": 10,
        "pool_recycle": -1
    },
    "production": {
        "pool_size": 20,
        "max_overflow": 30,
        "pool_timeout": 60,
        "pool_recycle": 7200
    }
}

def get_database_config() -> dict:
    """Get database configuration for current environment"""
    return DATABASE_CONFIGS.get(settings.environment, DATABASE_CONFIGS["development"])