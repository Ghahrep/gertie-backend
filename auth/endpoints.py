# auth/endpoints.py - Authentication Endpoints (Fixed for compatibility)
"""
Authentication Endpoints for Clean Financial Platform API
========================================================

Provides login, logout, and token management endpoints.
Compatible with both JSON and OAuth2 form data.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, Union
from datetime import datetime


from db.session import SessionLocal
from db.models import User
from .middleware import (
    create_jwt_token,
    authenticate_user,
    get_current_user,
    get_current_active_user
)

# Create auth router
auth_router = APIRouter(prefix="/auth", tags=["authentication"])

# Pydantic models for requests/responses
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: Dict[str, Any]

class UserProfile(BaseModel):
    id: int
    email: str
    role: str
    created_at: str
    is_active: bool = True

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@auth_router.post("/login", response_model=LoginResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return JWT token
    Compatible with OAuth2 form data (username/password)
    
    Args:
        form_data: OAuth2 form data (username treated as email)
        db: Database session
        
    Returns:
        LoginResponse: Token and user information
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        # OAuth2PasswordRequestForm uses 'username' field, but we treat it as email
        email = form_data.username
        password = form_data.password
        
        # Authenticate user
        user = authenticate_user(email, password, db)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Determine user role (customize based on your User model)
        user_role = getattr(user, 'role', 'user')
        
        # Create JWT token
        access_token = create_jwt_token(
            user_id=user.id,
            email=user.email,
            role=user_role
        )
        
        # Prepare user data for response
        user_data = {
            "id": user.id,
            "email": user.email,
            "role": user_role,
            "created_at": user.created_at.isoformat() if hasattr(user, 'created_at') else None,
            "is_active": getattr(user, 'is_active', True)
        }
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=24 * 3600,  # 24 hours in seconds
            user=user_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Login error: {e}")  # Debug logging
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

# Alternative JSON login endpoint for frontend compatibility
@auth_router.post("/login-json", response_model=LoginResponse)
async def login_json(request: LoginRequest, db: Session = Depends(get_db)):
    """
    JSON-based login endpoint for frontend applications
    
    Args:
        request: Login credentials (JSON)
        db: Database session
        
    Returns:
        LoginResponse: Token and user information
    """
    try:
        # Authenticate user
        user = authenticate_user(request.email, request.password, db)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Determine user role
        user_role = getattr(user, 'role', 'user')
        
        # Create JWT token
        access_token = create_jwt_token(
            user_id=user.id,
            email=user.email,
            role=user_role
        )
        
        # Prepare user data for response
        user_data = {
            "id": user.id,
            "email": user.email,
            "role": user_role,
            "created_at": user.created_at.isoformat() if hasattr(user, 'created_at') else None,
            "is_active": getattr(user, 'is_active', True)
        }
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=24 * 3600,
            user=user_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"JSON Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@auth_router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """
    Logout user (token invalidation would be handled client-side)
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Success message
    """
    return {
        "message": "Successfully logged out",
        "user_id": current_user.id,
        "timestamp": datetime.utcnow().isoformat()
    }

@auth_router.get("/profile", response_model=UserProfile)
async def get_user_profile(current_user: User = Depends(get_current_active_user)):
    """
    Get current user profile information
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        UserProfile: User profile data
    """
    try:
        # Get role from token payload
        user_role = getattr(current_user, '_token_payload', {}).get('role', 'user')
        
        return UserProfile(
            id=current_user.id,
            email=current_user.email,
            role=user_role,
            created_at=current_user.created_at.isoformat() if hasattr(current_user, 'created_at') else datetime.utcnow().isoformat(),
            is_active=getattr(current_user, 'is_active', True)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile"
        )

@auth_router.post("/refresh", response_model=TokenResponse)
async def refresh_token(current_user: User = Depends(get_current_user)):
    """
    Refresh JWT token for authenticated user
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        TokenResponse: New access token
    """
    try:
        # Get role from current token
        user_role = getattr(current_user, '_token_payload', {}).get('role', 'user')
        
        # Create new token
        new_token = create_jwt_token(
            user_id=current_user.id,
            email=current_user.email,
            role=user_role
        )
        
        return TokenResponse(
            access_token=new_token,
            token_type="bearer",
            expires_in=24 * 3600  # 24 hours in seconds
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

@auth_router.get("/verify")
async def verify_token(current_user: User = Depends(get_current_active_user)):
    """
    Verify current token is valid
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Token verification status
    """
    token_payload = getattr(current_user, '_token_payload', {})
    
    return {
        "valid": True,
        "user_id": current_user.id,
        "email": current_user.email,
        "role": token_payload.get('role', 'user'),
        "expires_at": datetime.fromtimestamp(token_payload.get('exp', 0)).isoformat() if token_payload.get('exp') else None,
        "verified_at": datetime.utcnow().isoformat()
    }

# Test endpoint to check available users (for debugging)
@auth_router.get("/debug/users")
async def debug_users(db: Session = Depends(get_db)):
    """
    Debug endpoint to check available users (remove in production)
    """
    try:
        users = db.query(User).limit(5).all()
        return {
            "available_users": [
                {
                    "id": user.id,
                    "email": user.email,
                    "has_password": bool(getattr(user, 'hashed_password', None))
                }
                for user in users
            ],
            "total_users": db.query(User).count(),
            "note": "Remove this endpoint in production"
        }
    except Exception as e:
        return {"error": str(e)}

# Optional: Admin-only endpoints
@auth_router.get("/admin/users")
async def list_users(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    List all users (admin only)
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        List of users
    """
    # Check admin role
    user_role = getattr(current_user, '_token_payload', {}).get('role', 'user')
    
    if user_role != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        users = db.query(User).all()
        
        return {
            "users": [
                {
                    "id": user.id,
                    "email": user.email,
                    "created_at": user.created_at.isoformat() if hasattr(user, 'created_at') else None,
                    "is_active": getattr(user, 'is_active', True)
                }
                for user in users
            ],
            "total_count": len(users),
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users"
        )

