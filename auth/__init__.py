# auth/__init__.py
"""
Authentication package for Financial Platform API
"""

from .middleware import (
    get_current_user,
    get_current_active_user,
    get_user_portfolio,
    get_user_portfolio_write,
    create_jwt_token,
    verify_jwt_token
)

from .endpoints import auth_router

__all__ = [
    "get_current_user",
    "get_current_active_user", 
    "get_user_portfolio",
    "get_user_portfolio_write",
    "create_jwt_token",
    "verify_jwt_token",
    "auth_router"
]
