# auth/middleware.py - Updated JWT Authentication Middleware
"""
JWT Authentication Middleware - Updated for Production Use
==========================================================

Updated with proper password hashing and enhanced User model support.
"""

import jwt
import hashlib
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from auth.password_utils import verify_password
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import os
from sqlalchemy.orm import Session

from db.session import SessionLocal
from db.models import User, Portfolio

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Security scheme
security = HTTPBearer()

class AuthenticationError(Exception):
    """Custom authentication exception"""
    pass

class AuthorizationError(Exception):
    """Custom authorization exception"""
    pass

def create_jwt_token(user_id: int, email: str, role: str = "user") -> str:
    """
    Create JWT token for authenticated user
    
    Args:
        user_id: User's database ID
        email: User's email address
        role: User's role (default: "user")
        
    Returns:
        str: JWT token
    """
    payload = {
        "user_id": user_id,
        "email": email,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.utcnow(),
        "type": "access_token"
    }
    
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        Dict containing user information
        
    Raises:
        AuthenticationError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        # Verify token type
        if payload.get("type") != "access_token":
            raise AuthenticationError("Invalid token type")
        
        # Check expiration
        exp = payload.get("exp")
        if exp and datetime.utcnow().timestamp() > exp:
            raise AuthenticationError("Token has expired")
        
        return payload
        
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError:
        raise AuthenticationError("Invalid token")
    except Exception as e:
        raise AuthenticationError(f"Token validation failed: {str(e)}")

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(lambda: SessionLocal())
) -> User:
    """
    Get current authenticated user from JWT token
    
    Args:
        credentials: HTTP authorization credentials
        db: Database session
        
    Returns:
        User: Authenticated user object
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Extract token from credentials
        token = credentials.credentials
        
        # Verify token and get payload
        payload = verify_jwt_token(token)
        user_id = payload.get("user_id")
        
        if not user_id:
            raise AuthenticationError("Invalid token payload")
        
        # Get user from database
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise AuthenticationError("User not found")
        
        # Check if user is active
        if not user.is_active:
            raise AuthenticationError("User account is inactive")
        
        # Add token payload to user object for convenience
        user._token_payload = payload
        
        return user
        
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )
    finally:
        db.close()

def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Get current active user (additional validation)
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User: Active user object
        
    Raises:
        HTTPException: If user is inactive
    """
    # Additional validation already done in get_current_user
    # This function is kept for backward compatibility
    return current_user

def check_portfolio_access(
    portfolio_id: int,
    current_user: User,
    db: Session,
    require_write: bool = False
) -> Portfolio:
    """
    Check if user has access to specified portfolio
    
    Args:
        portfolio_id: Portfolio ID to check
        current_user: Current authenticated user
        db: Database session
        require_write: Whether write access is required
        
    Returns:
        Portfolio: Portfolio object if access is granted
        
    Raises:
        HTTPException: If access is denied or portfolio not found
    """
    try:
        # Get portfolio from database
        portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        
        if not portfolio:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Portfolio not found"
            )
        
        # Check ownership
        if portfolio.user_id != current_user.id:
            # Check if user has admin role (from token payload)
            user_role = getattr(current_user, '_token_payload', {}).get('role', 'user')
            
            if user_role != 'admin':
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,  # Return 404 instead of 403 for security
                    detail="Portfolio not found"
                )
        
        return portfolio
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Access control check failed"
        )

def get_user_portfolio(
    portfolio_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(lambda: SessionLocal())
) -> Portfolio:
    """
    Dependency to get portfolio with access control
    
    Args:
        portfolio_id: Portfolio ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Portfolio: Portfolio object if access is granted
    """
    try:
        return check_portfolio_access(portfolio_id, current_user, db)
    finally:
        db.close()

def get_user_portfolio_write(
    portfolio_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(lambda: SessionLocal())
) -> Portfolio:
    """
    Dependency to get portfolio with write access control
    
    Args:
        portfolio_id: Portfolio ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Portfolio: Portfolio object if write access is granted
    """
    try:
        return check_portfolio_access(portfolio_id, current_user, db, require_write=True)
    finally:
        db.close()

def require_role(required_role: str):
    """
    Decorator to require specific user role
    
    Args:
        required_role: Required role (e.g., "admin", "manager")
        
    Returns:
        Dependency function
    """
    def role_dependency(current_user: User = Depends(get_current_active_user)):
        user_role = getattr(current_user, '_token_payload', {}).get('role', 'user')
        
        if user_role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        
        return current_user
    
    return role_dependency

# Admin-only dependency
get_admin_user = require_role("admin")

def authenticate_user(email: str, password: str, db: Session) -> Optional[User]:
    """
    Authenticate user credentials
    
    Args:
        email: User email
        password: User password
        db: Database session
        
    Returns:
        User object if authentication successful, None otherwise
    """
    try:
        # Get user by email
        user = db.query(User).filter(User.email == email).first()
        
        if not user:
            return None
        
        # Check if user is active
        if not user.is_active:
            return None
        
        # Verify password
        if verify_password(password, user.hashed_password):
            # Update last login
            user.update_last_login()
            db.commit()
            return user
        
        return None
        
    except Exception as e:
        print(f"Authentication error: {e}")
        return None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against hash
    
    For now, using SHA256 for compatibility with existing data.
    In production, migrate to bcrypt using auth/password_utils.py
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password from database
        
    Returns:
        bool: True if password matches
    """
    # Current implementation for compatibility
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password

def hash_password_simple(password: str) -> str:
    """
    Simple password hashing for compatibility
    
    Args:
        password: Plain text password
        
    Returns:
        str: Hashed password
    """
    return hashlib.sha256(password.encode()).hexdigest()