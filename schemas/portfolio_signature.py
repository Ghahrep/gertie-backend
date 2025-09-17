# schemas/portfolio_signature.py - Portfolio Signature Response Schemas
"""
Portfolio Signature Response Schemas
===================================
Pydantic schemas for portfolio signature API responses.
Provides type safety and automatic API documentation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class DataSourceType(str, Enum):
    """Data source indicators for transparency"""
    LIVE = "live"
    CACHED = "cached" 
    SYNTHETIC = "synthetic"

class RiskLevel(str, Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PortfolioSignatureResponse(BaseModel):
    """Main portfolio signature response schema"""
    
    # Core Metrics (normalized 0-1 or 0-100)
    risk_score: float = Field(..., ge=0, le=100, description="Overall risk sentiment (0-100)")
    tail_risk: float = Field(..., ge=0, le=1, description="Normalized CVaR tail risk (0-1)")
    correlation: float = Field(..., ge=0, le=1, description="Average portfolio correlation (0-1)")
    volatility_forecast: float = Field(..., ge=0, le=100, description="GARCH volatility trend intensity (0-100)")
    complexity: float = Field(..., ge=0, le=1, description="Multifractal complexity measure (0-1)")
    concentration: float = Field(..., ge=0, le=1, description="Portfolio concentration score (0-1)")
    
    # Categorical Classifications
    risk_level: RiskLevel = Field(..., description="Overall risk classification")
    
    # Metadata
    data_source: DataSourceType = Field(..., description="Data freshness indicator")
    last_updated: datetime = Field(..., description="Last calculation timestamp")
    portfolio_id: int = Field(..., description="Portfolio identifier")
    
    # Optional Raw Values (for debugging/validation)
    raw_metrics: Optional[Dict[str, Any]] = Field(None, description="Raw metric values for validation")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "risk_score": 72.5,
                "tail_risk": 0.3,
                "correlation": 0.65,
                "volatility_forecast": 45.0,
                "complexity": 0.8,
                "concentration": 0.25,
                "risk_level": "medium",
                "data_source": "live",
                "last_updated": "2025-09-11T14:30:00Z",
                "portfolio_id": 123,
                "raw_metrics": {
                    "cvar_95": -0.025,
                    "correlation_matrix_avg": 0.65,
                    "garch_volatility": 0.15
                }
            }
        }

class PortfolioSignatureBatch(BaseModel):
    """Batch response for multiple portfolio signatures"""
    
    signatures: List[PortfolioSignatureResponse] = Field(..., description="Portfolio signatures")
    total_count: int = Field(..., description="Total portfolios processed")
    cache_hit_rate: Optional[float] = Field(None, ge=0, le=1, description="Cache efficiency metric")
    processing_time_ms: Optional[float] = Field(None, description="Total processing time")
    
    class Config:
        schema_extra = {
            "example": {
                "signatures": [
                    {
                        "risk_score": 72.5,
                        "tail_risk": 0.3,
                        "correlation": 0.65,
                        "volatility_forecast": 45.0,
                        "complexity": 0.8,
                        "concentration": 0.25,
                        "risk_level": "medium",
                        "data_source": "cached",
                        "last_updated": "2025-09-11T14:30:00Z",
                        "portfolio_id": 123
                    }
                ],
                "total_count": 1,
                "cache_hit_rate": 0.8,
                "processing_time_ms": 150.5
            }
        }

class SignatureUpdateRequest(BaseModel):
    """Request schema for signature updates"""
    
    force_refresh: bool = Field(False, description="Force recalculation bypassing cache")
    include_raw_metrics: bool = Field(False, description="Include raw values in response")
    
    class Config:
        schema_extra = {
            "example": {
                "force_refresh": True,
                "include_raw_metrics": False
            }
        }

class RiskAlert(BaseModel):
    """Risk alert schema for threshold breaches"""
    
    portfolio_id: int = Field(..., description="Portfolio identifier")
    alert_type: str = Field(..., description="Type of risk alert")
    metric_name: str = Field(..., description="Metric that triggered alert")
    current_value: float = Field(..., description="Current metric value")
    threshold_value: float = Field(..., description="Alert threshold")
    severity: RiskLevel = Field(..., description="Alert severity level")
    triggered_at: datetime = Field(..., description="Alert timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "portfolio_id": 123,
                "alert_type": "tail_risk_breach",
                "metric_name": "tail_risk",
                "current_value": 0.85,
                "threshold_value": 0.8,
                "severity": "high",
                "triggered_at": "2025-09-11T14:30:00Z"
            }
        }