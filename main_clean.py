# main_clean.py - Enhanced with WebSocket Integration
"""
Clean Financial Platform API - Sprint 2B + Task 2.6
========================================

Enhanced FastAPI application with JWT authentication, pagination, validation, monitoring, and WebSocket support.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database imports
from sqlalchemy.orm import Session
from db.session import SessionLocal
from db.models import Portfolio, User, Holding, Asset

# Service imports
from services.financial_analysis import FinancialAnalysisService, MarketDataProvider

# Initialize FastAPI app FIRST
app = FastAPI(
    title="Financial Platform - Clean Architecture",
    description="Authenticated financial analysis API with direct tool integration and WebSocket support",
    version="2.1.0"
)

print("FastAPI app initialized")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("CORS middleware added")

# Import and include authentication router EARLY
print("Importing authentication router...")
try:
    from auth.endpoints import auth_router
    print(f"✅ Auth router imported successfully with {len(auth_router.routes)} routes")
    
    # Include authentication router
    app.include_router(auth_router)
    print("✅ Auth router included successfully")
    
    # Verify routes were added
    total_routes = len(app.routes)
    auth_routes = [r for r in app.routes if hasattr(r, 'path') and r.path.startswith('/auth')]
    print(f"✅ Verification: Total app routes: {total_routes}, Auth routes: {len(auth_routes)}")
    
    if len(auth_routes) == 0:
        print("❌ WARNING: Auth routes not found in app after inclusion!")
    else:
        print("Auth routes found:")
        for route in auth_routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                print(f"   {route.methods} {route.path}")
    
except Exception as e:
    print(f"❌ Auth router setup failed: {e}")
    import traceback
    traceback.print_exc()
    print("API will run without authentication")

# WebSocket Integration Setup
print("Setting up WebSocket integration...")
try:
    # Create websocket directory structure if it doesn't exist
    import os
    os.makedirs("websocket", exist_ok=True)
    
    # Create __init__.py if it doesn't exist
    init_file = "websocket/__init__.py"
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("")
    
    # Try to import WebSocket router
    from websocket.endpoints import websocket_router
    app.include_router(websocket_router)
    print("✅ WebSocket router included successfully")
    
    # Add route count verification for WebSocket
    websocket_routes = [r for r in app.routes if hasattr(r, 'path') and r.path.startswith('/ws')]
    print(f"✅ WebSocket routes added: {len(websocket_routes)}")
    for route in websocket_routes:
        if hasattr(route, 'path'):
            print(f"   WS: {route.path}")
    
    WEBSOCKET_AVAILABLE = True
    
except Exception as e:
    print(f"❌ WebSocket setup failed: {e}")
    print("API will run without WebSocket functionality")
    WEBSOCKET_AVAILABLE = False

# Import authentication middleware functions AFTER router inclusion
print("Importing auth middleware...")
try:
    from auth.middleware import (
        get_current_active_user,
        get_user_portfolio,
        get_user_portfolio_write,
        check_portfolio_access,
        get_admin_user
    )
    print("✅ Auth middleware imported successfully")
except Exception as e:
    print(f"❌ Auth middleware import failed: {e}")

# Enhanced API utilities - these will be created if missing
try:
    from utils.pagination import (
        PaginationParams, DateRangeParams, HoldingsFilterParams, 
        create_paginated_response, apply_pagination, apply_sorting
    )
    PAGINATION_AVAILABLE = True
    print("✅ Pagination utilities available")
except ImportError:
    PAGINATION_AVAILABLE = False
    print("Warning: Pagination utilities not available")

try:
    from utils.validation import (
        validate_request_data, FinancialValidator, PortfolioBusinessRules,
        ValidationError, BusinessRuleError
    )
    VALIDATION_AVAILABLE = True
    print("✅ Validation utilities available")
except ImportError:
    VALIDATION_AVAILABLE = False
    print("Warning: Validation utilities not available")

try:
    from utils.monitoring import rate_limit, performance_monitor
    MONITORING_AVAILABLE = True
    print("✅ Monitoring utilities available")
except ImportError:
    MONITORING_AVAILABLE = False
    print("Warning: Monitoring utilities not available")

# Add monitoring middleware if available
if MONITORING_AVAILABLE:
    try:
        from utils.monitoring import RateLimitMiddleware, MonitoringMiddleware
        app.add_middleware(MonitoringMiddleware)
        app.add_middleware(RateLimitMiddleware)
        print("✅ Monitoring and rate limiting middleware enabled")
    except ImportError:
        print("Warning: Could not import monitoring middleware")

# Initialize services
print("Initializing services...")
market_data_provider = MarketDataProvider()
financial_service = FinancialAnalysisService(market_data_provider)
print("✅ Services initialized")

# Pydantic models for API
class AnalysisRequest(BaseModel):
    query: str
    portfolio_id: Optional[int] = None
    chat_history: Optional[List[Dict[str, str]]] = None

class RiskAnalysisRequest(BaseModel):
    portfolio_id: int

class BehaviorAnalysisRequest(BaseModel):
    chat_history: List[Dict[str, str]]

class TradeOrdersRequest(BaseModel):
    target_weights: Dict[str, float]
    min_trade_amount: Optional[float] = 100.0

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Health check endpoint (public) - Updated with WebSocket status
@app.get("/health")
async def health_check():
    """Public health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "service": "financial-platform-clean",
        "version": "2.1.0",
        "authentication": "enabled",
        "enhanced_features": {
            "pagination": PAGINATION_AVAILABLE,
            "validation": VALIDATION_AVAILABLE,
            "monitoring": MONITORING_AVAILABLE,
            "websocket": WEBSOCKET_AVAILABLE
        }
    }

# WebSocket Test Client Endpoint
@app.get("/test-websocket")
async def serve_websocket_test():
    """Serve the WebSocket test client"""
    try:
        with open("test_websocket.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            content="""
            <!DOCTYPE html>
            <html>
                <head>
                    <title>WebSocket Test Client Not Found</title>
                    <style>
                        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                        .container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                        .endpoint { background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 4px; font-family: monospace; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>WebSocket Test Client Not Found</h1>
                        <p>The test client HTML file (test_websocket.html) is not available.</p>
                        <p>You can test WebSocket endpoints directly using a WebSocket client:</p>
                        <div class="endpoint">ws://localhost:8000/ws/test - Test endpoint with message handling</div>
                        <div class="endpoint">ws://localhost:8000/ws/echo - Simple echo endpoint</div>
                        <div class="endpoint">GET /ws/status - WebSocket status information</div>
                        
                        <h3>Quick Test with JavaScript Console:</h3>
                        <pre>
// Test in browser console:
const ws = new WebSocket('ws://localhost:8000/ws/test');
ws.onmessage = (event) => console.log('Received:', JSON.parse(event.data));
ws.onopen = () => ws.send('Hello WebSocket!');
                        </pre>
                    </div>
                </body>
            </html>
            """,
            status_code=200
        )

# Service status endpoint (public)
@app.get("/status")
async def service_status():
    """Get service status and capabilities"""
    try:
        # Test database connection
        db = SessionLocal()
        portfolio_count = db.query(Portfolio).count()
        user_count = db.query(User).count()
        db.close()
        
        # Count actual routes
        total_routes = len(app.routes)
        auth_routes = [r for r in app.routes if hasattr(r, 'path') and r.path.startswith('/auth')]
        websocket_routes = [r for r in app.routes if hasattr(r, 'path') and r.path.startswith('/ws')]
        
        return {
            "service": "FinancialAnalysisService",
            "status": "operational",
            "version": "2.1.0",
            "authentication": "enabled",
            "tools_available": 24,
            "database_connected": True,
            "portfolios_in_system": portfolio_count,
            "users_in_system": user_count,
            "enhanced_features": {
                "pagination": PAGINATION_AVAILABLE,
                "validation": VALIDATION_AVAILABLE,
                "monitoring": MONITORING_AVAILABLE,
                "websocket": WEBSOCKET_AVAILABLE
            },
            "capabilities": {
                "risk_analysis": True,
                "behavioral_analysis": True,
                "regime_analysis": True,
                "fractal_analysis": True,
                "portfolio_management": True,
                "trade_generation": True,
                "user_authentication": True,
                "access_control": True,
                "real_time_communication": WEBSOCKET_AVAILABLE
            },
            "endpoints": {
                "public": ["/health", "/status", "/auth/*", "/test-websocket"],
                "authenticated": ["/analyze", "/portfolio/*", "/user/*"],
                "websocket": ["/ws/test", "/ws/echo", "/ws/status"] if WEBSOCKET_AVAILABLE else [],
                "total_endpoints": total_routes,
                "auth_endpoints": len(auth_routes),
                "websocket_endpoints": len(websocket_routes)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "service": "FinancialAnalysisService",
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Main analysis endpoint - unified interface (authenticated)
@app.post("/analyze")
async def analyze(
    request: AnalysisRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Unified analysis endpoint with query routing (requires authentication)
    
    Supports queries like:
    - "analyze portfolio risk" 
    - "check for behavioral biases"
    - "identify market regimes"
    - "perform fractal analysis"
    """
    try:
        # If portfolio_id is provided, verify user has access
        if request.portfolio_id:
            db = SessionLocal()
            try:
                check_portfolio_access(request.portfolio_id, current_user, db)
            except HTTPException:
                raise
            finally:
                db.close()
        
        result = financial_service.analyze(
            query=request.query,
            portfolio_id=request.portfolio_id,
            chat_history=request.chat_history or []
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        
        return {
            "success": True,
            "analysis_type": result.analysis_type,
            "execution_time": result.execution_time,
            "data": result.data,
            "user_id": current_user.id,
            "analyzed_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Specific analysis endpoints (authenticated)
@app.post("/analyze/risk")
async def analyze_risk(
    request: RiskAnalysisRequest,
    current_user: User = Depends(get_current_active_user),
    portfolio: Portfolio = Depends(get_user_portfolio)
):
    """Dedicated risk analysis endpoint (requires portfolio access)"""
    try:
        result = financial_service.analyze_risk(request.portfolio_id)
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        
        return {
            "success": True,
            "portfolio_name": portfolio.name,
            "portfolio_id": portfolio.id,
            "user_id": current_user.id,
            "analysis": result.data,
            "execution_time": result.execution_time,
            "analyzed_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Risk analysis failed for user {current_user.id}, portfolio {request.portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/portfolios")
async def get_user_portfolios(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all portfolios for the current user"""
    try:
        portfolios = db.query(Portfolio).filter(Portfolio.user_id == current_user.id).all()
        
        portfolio_list = []
        for portfolio in portfolios:
            # Calculate basic metrics
            total_value = 0
            holdings_count = len(portfolio.holdings)
            
            for holding in portfolio.holdings:
                current_price = holding.asset.current_price or 100
                value = holding.shares * current_price
                total_value += value
            
            portfolio_list.append({
                "id": portfolio.id,
                "name": portfolio.name,
                "description": portfolio.description,
                "currency": portfolio.currency,
                "total_value": total_value,
                "holdings_count": holdings_count,
                "created_at": portfolio.created_at.isoformat() if hasattr(portfolio, 'created_at') else None,
                "is_active": portfolio.is_active
            })
        
        return {
            "user_id": current_user.id,
            "portfolios": portfolio_list,
            "total_portfolios": len(portfolio_list),
            "retrieved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve portfolios for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve portfolios")

@app.get("/portfolio/{portfolio_id}/summary")
async def get_portfolio_summary(
    portfolio_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get portfolio summary with holdings breakdown (requires portfolio access)"""
    try:
        # Check portfolio access with the db session
        portfolio = check_portfolio_access(portfolio_id, current_user, db)
        
        # Calculate holdings values with eager loading
        holdings_data = {}
        for holding in portfolio.holdings:
            current_price = holding.asset.current_price or 100  # Default price if missing
            value = holding.shares * current_price
            holdings_data[holding.asset.ticker] = value
        
        # Import and use the portfolio summary function
        from tools.portfolio_tools import calculate_portfolio_summary
        summary = calculate_portfolio_summary(holdings_data)
        
        if not summary.get('success'):
            raise HTTPException(status_code=400, detail=summary.get('error', 'Summary calculation failed'))
        
        # Add portfolio metadata and user context
        summary.update({
            "portfolio_id": portfolio_id,
            "portfolio_name": portfolio.name,
            "description": portfolio.description,
            "currency": portfolio.currency,
            "user_id": current_user.id,
            "last_updated": datetime.now().isoformat()
        })
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Portfolio summary failed for user {current_user.id}, portfolio {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/{portfolio_id}/holdings")
async def get_portfolio_holdings(
    portfolio_id: int,
    page: int = 1,
    limit: int = 50,
    sort_by: Optional[str] = None,
    sort_order: str = "desc",
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    sector: Optional[str] = None,
    asset_type: Optional[str] = None,
    search: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get portfolio holdings with pagination and filtering"""
    try:
        # Check portfolio access
        portfolio = check_portfolio_access(portfolio_id, current_user, db)
        
        # Build base query
        query = db.query(Holding).join(Asset).filter(Holding.portfolio_id == portfolio_id)
        
        # Apply filters
        if min_value is not None:
            query = query.filter(Holding.shares * Asset.current_price >= min_value)
        
        if max_value is not None:
            query = query.filter(Holding.shares * Asset.current_price <= max_value)
        
        if sector:
            query = query.filter(Asset.sector.ilike(f"%{sector}%"))
        
        if asset_type:
            query = query.filter(Asset.asset_type.ilike(f"%{asset_type}%"))
        
        if search:
            from sqlalchemy import or_
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    Asset.ticker.ilike(search_term),
                    Asset.name.ilike(search_term)
                )
            )
        
        # Apply sorting
        allowed_sort_fields = ['shares', 'purchase_price', 'purchase_date', 'ticker', 'name', 'current_price']
        if sort_by and sort_by in allowed_sort_fields:
            if sort_by in ['ticker', 'name', 'current_price']:
                sort_attr = getattr(Asset, sort_by)
            else:
                sort_attr = getattr(Holding, sort_by)
            
            if sort_order.lower() == "desc":
                query = query.order_by(sort_attr.desc())
            else:
                query = query.order_by(sort_attr.asc())
        else:
            # Default sort by purchase date
            query = query.order_by(Holding.purchase_date.desc())
        
        # Count total results
        total_count = query.count()
        
        # Apply pagination
        offset = (page - 1) * limit
        holdings = query.offset(offset).limit(limit).all()
        
        # Transform holdings data
        holdings_data = []
        total_value = 0
        
        for holding in holdings:
            current_price = holding.asset.current_price or 100
            market_value = holding.shares * current_price
            total_value += market_value
            
            holdings_data.append({
                "id": holding.id,
                "ticker": holding.asset.ticker,
                "name": holding.asset.name,
                "sector": holding.asset.sector,
                "asset_type": holding.asset.asset_type,
                "shares": holding.shares,
                "purchase_price": holding.purchase_price,
                "current_price": current_price,
                "market_value": market_value,
                "cost_basis": holding.cost_basis,
                "purchase_date": holding.purchase_date.isoformat() if holding.purchase_date else None,
                "unrealized_pnl": market_value - holding.cost_basis if holding.cost_basis else None,
                "unrealized_pnl_pct": ((market_value - holding.cost_basis) / holding.cost_basis * 100) if holding.cost_basis else None
            })
        
        # Calculate weights
        for holding_data in holdings_data:
            holding_data["weight_pct"] = (holding_data["market_value"] / total_value * 100) if total_value > 0 else 0
        
        # Calculate pagination info
        total_pages = (total_count + limit - 1) // limit
        has_next = page < total_pages
        has_prev = page > 1
        
        return {
            "portfolio_id": portfolio_id,
            "portfolio_name": portfolio.name,
            "user_id": current_user.id,
            "total_portfolio_value": total_value,
            "filters_applied": {
                "min_value": min_value,
                "max_value": max_value,
                "sector": sector,
                "asset_type": asset_type,
                "search": search
            },
            "pagination": {
                "current_page": page,
                "per_page": limit,
                "total_pages": total_pages,
                "total_items": total_count,
                "has_next": has_next,
                "has_prev": has_prev
            },
            "holdings": holdings_data,
            "enhanced_features_available": PAGINATION_AVAILABLE,
            "last_updated": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Portfolio holdings failed for user {current_user.id}, portfolio {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/portfolio/{portfolio_id}/trade-orders")
async def generate_trade_orders_endpoint(
    portfolio_id: int,
    request: TradeOrdersRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Generate trade orders for portfolio rebalancing"""
    try:
        # Check portfolio access
        portfolio = check_portfolio_access(portfolio_id, current_user, db, require_write=True)
        
        # Get current holdings and total value
        current_holdings = {}
        total_value = 0
        
        for holding in portfolio.holdings:
            current_price = holding.asset.current_price or 100
            value = holding.shares * current_price
            current_holdings[holding.asset.ticker] = value
            total_value += value
        
        if total_value == 0:
            raise HTTPException(status_code=400, detail="Portfolio has no value")
        
        # Generate trade orders
        from tools.portfolio_tools import generate_trade_orders
        trade_result = generate_trade_orders(
            current_holdings=current_holdings,
            target_weights=request.target_weights,
            total_portfolio_value=total_value,
            min_trade_amount=request.min_trade_amount
        )
        
        if not trade_result.get('success'):
            raise HTTPException(status_code=400, detail=trade_result.get('error', 'Trade generation failed'))
        
        return {
            "portfolio_id": portfolio_id,
            "portfolio_name": portfolio.name,
            "user_id": current_user.id,
            "generated_at": datetime.now().isoformat(),
            **trade_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trade order generation failed for user {current_user.id}, portfolio {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Always define admin endpoints (with graceful degradation)
@app.get("/admin/metrics/system")
async def get_system_metrics(
    current_user: User = Depends(get_admin_user)
):
    """Get system performance metrics (admin only)"""
    try:
        if MONITORING_AVAILABLE:
            metrics = performance_monitor.get_system_metrics()
            return {
                "timestamp": datetime.now().isoformat(),
                "system_metrics": metrics
            }
        else:
            return {
                "timestamp": datetime.now().isoformat(),
                "system_metrics": {
                    "status": "monitoring_unavailable",
                    "message": "Enhanced monitoring features not available"
                }
            }
    except Exception as e:
        logger.error(f"System metrics error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system metrics")

@app.get("/admin/metrics/requests")
async def get_request_metrics(
    minutes: int = 60,
    current_user: User = Depends(get_admin_user)
):
    """Get API request performance metrics (admin only)"""
    try:
        if minutes > 1440:  # Limit to 24 hours
            raise HTTPException(status_code=400, detail="Maximum 1440 minutes (24 hours)")
        
        if MONITORING_AVAILABLE:
            metrics = performance_monitor.get_request_metrics(minutes)
            return {
                "timestamp": datetime.now().isoformat(),
                "request_metrics": metrics
            }
        else:
            return {
                "timestamp": datetime.now().isoformat(),
                "request_metrics": {
                    "status": "monitoring_unavailable",
                    "message": "Enhanced monitoring features not available"
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request metrics error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve request metrics")

@app.get("/user/rate-limit-status")
async def get_rate_limit_status(
    current_user: User = Depends(get_current_active_user)
):
    """Get current rate limit status for the user"""
    try:
        if MONITORING_AVAILABLE:
            from utils.monitoring import rate_limiter
            client_key = f"user_{current_user.id}"
            
            # Check different endpoint limits
            endpoints = {
                'general': {'max_requests': 100, 'window_seconds': 3600},
                'analysis': {'max_requests': 20, 'window_seconds': 3600},
                'trading': {'max_requests': 10, 'window_seconds': 3600}
            }
            
            rate_limits = {}
            for endpoint_name, config in endpoints.items():
                allowed, rate_info = rate_limiter.is_allowed(
                    key=f"{endpoint_name}_{client_key}",
                    max_requests=config['max_requests'],
                    window_seconds=config['window_seconds']
                )
                
                rate_limits[endpoint_name] = {
                    "limit": config['max_requests'],
                    "remaining": rate_info['requests_remaining'],
                    "reset_time": datetime.fromtimestamp(rate_info['reset_time']).isoformat(),
                    "window_seconds": config['window_seconds']
                }
            
            return {
                "user_id": current_user.id,
                "rate_limits": rate_limits,
                "checked_at": datetime.now().isoformat()
            }
        else:
            return {
                "user_id": current_user.id,
                "rate_limits": {
                    "status": "monitoring_unavailable",
                    "message": "Rate limiting features not available"
                },
                "checked_at": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Rate limit status error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve rate limit status")

# Debug endpoint to verify the server state
@app.get("/debug/routes")
async def debug_routes():
    """Debug endpoint to check loaded routes"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": getattr(route, 'name', 'unknown')
            })
    
    auth_routes = [r for r in routes if r['path'].startswith('/auth')]
    websocket_routes = [r for r in routes if r['path'].startswith('/ws')]
    
    return {
        "total_routes": len(routes),
        "auth_routes_count": len(auth_routes),
        "websocket_routes_count": len(websocket_routes),
        "auth_routes": auth_routes,
        "websocket_routes": websocket_routes,
        "all_routes": routes
    }

if __name__ == "__main__":
    print("\n" + "="*50)
    print("STARTING SERVER")
    print("="*50)
    
    # Final route verification before starting
    total_routes = len(app.routes)
    auth_routes = [r for r in app.routes if hasattr(r, 'path') and r.path.startswith('/auth')]
    websocket_routes = [r for r in app.routes if hasattr(r, 'path') and r.path.startswith('/ws')]
    
    print(f"Final verification: {total_routes} total routes")
    print(f"  - Auth routes: {len(auth_routes)}")
    print(f"  - WebSocket routes: {len(websocket_routes)}")
    
    if len(auth_routes) > 0:
        print("✅ AUTH ROUTES SUCCESSFULLY LOADED")
    else:
        print("❌ NO AUTH ROUTES FOUND - AUTHENTICATION WILL NOT WORK")
    
    if len(websocket_routes) > 0:
        print("✅ WEBSOCKET ROUTES SUCCESSFULLY LOADED")
    else:
        print("⚠️  NO WEBSOCKET ROUTES FOUND - WEBSOCKET FEATURES UNAVAILABLE")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)