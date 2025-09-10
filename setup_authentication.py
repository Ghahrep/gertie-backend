# setup_authentication.py - Authentication Setup Script
"""
Authentication Setup Script for Sprint 2B
==========================================

Sets up JWT authentication system for the clean financial platform API.
"""

import os
import sys
import secrets

def setup_directories():
    """Create necessary directories for authentication"""
    directories = ["auth"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory exists: {directory}")

def create_init_file():
    """Create __init__.py file for auth package"""
    init_content = '''# auth/__init__.py
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
'''
    
    with open("auth/__init__.py", "w") as f:
        f.write(init_content)
    print("Created auth/__init__.py")

def generate_jwt_secret():
    """Generate a secure JWT secret key"""
    secret_key = secrets.token_urlsafe(32)
    
    env_file = ".env"
    env_content = f"\n# JWT Configuration\nJWT_SECRET_KEY={secret_key}\n"
    
    if os.path.exists(env_file):
        with open(env_file, "a") as f:
            f.write(env_content)
        print(f"Appended JWT_SECRET_KEY to existing {env_file}")
    else:
        with open(env_file, "w") as f:
            f.write(env_content)
        print(f"Created {env_file} with JWT_SECRET_KEY")
    
    print(f"Generated secure JWT secret key: {secret_key[:8]}...")

def check_dependencies():
    """Check for required authentication dependencies"""
    required_packages = [
        "PyJWT",
        "python-multipart",
        "passlib",
        "bcrypt"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "PyJWT":
                import jwt
            elif package == "python-multipart":
                import multipart
            elif package == "passlib":
                import passlib
            elif package == "bcrypt":
                import bcrypt
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("All authentication dependencies installed")
        return True

def create_example_user_script():
    """Create script to add example users for testing"""
    script_content = '''# create_test_users.py - Create Test Users
"""
Script to create test users for authentication testing
"""

import hashlib
from sqlalchemy.orm import Session
from db.session import SessionLocal
from db.models import User

def hash_password(password: str) -> str:
    """Simple password hashing for testing (use proper hashing in production)"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_test_users():
    """Create test users in the database"""
    db = SessionLocal()
    
    try:
        # Test user 1
        user1 = User(
            email="user1@example.com",
            hashed_password=hash_password("password123")
        )
        
        # Test user 2  
        user2 = User(
            email="user2@example.com",
            hashed_password=hash_password("password456")
        )
        
        # Admin user
        admin = User(
            email="admin@example.com", 
            hashed_password=hash_password("adminpass")
        )
        
        # Add users if they don't exist
        existing_user1 = db.query(User).filter(User.email == "user1@example.com").first()
        if not existing_user1:
            db.add(user1)
            print("Created user1@example.com")
        
        existing_user2 = db.query(User).filter(User.email == "user2@example.com").first()
        if not existing_user2:
            db.add(user2)
            print("Created user2@example.com")
            
        existing_admin = db.query(User).filter(User.email == "admin@example.com").first()
        if not existing_admin:
            db.add(admin)
            print("Created admin@example.com")
        
        db.commit()
        print("Test users created successfully")
        
    except Exception as e:
        print(f"Error creating test users: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    create_test_users()
'''
    
    with open("create_test_users.py", "w") as f:
        f.write(script_content)
    print("Created create_test_users.py script")

def print_next_steps():
    """Print instructions for completing authentication setup"""
    print("\n" + "=" * 60)
    print("📋 Manual Steps Required:")
    print("=" * 60)
    
    print("\n1. Copy Authentication Files:")
    print("   - Copy auth/middleware.py from the middleware artifact")
    print("   - Copy auth/endpoints.py from the endpoints artifact") 
    print("   - Copy updated main_clean.py from the main application artifact")
    
    print("\n2. Update Your User Model:")
    print("   - Ensure your User model has required fields:")
    print("     * id, email, hashed_password")
    print("     * Optional: role, is_active, created_at")
    
    print("\n3. Implement Password Verification:")
    print("   - Update verify_password() in auth/middleware.py")
    print("   - Use bcrypt, passlib, or your existing method")
    
    print("\n4. Create Test Users:")
    print("   - Run: python create_test_users.py")
    print("   - Or manually add users to your database")
    
    print("\n5. Test Authentication:")
    print("   - Run: python test_authentication.py")
    print("   - Start API: python main_clean.py")
    print("   - Test login at: POST http://localhost:8000/auth/login")

def main():
    """Run authentication setup"""
    print("🔐 Authentication Setup - Sprint 2B Task 2.4")
    print("=" * 60)
    
    # Setup directories
    setup_directories()
    
    # Create __init__.py
    create_init_file()
    
    # Generate JWT secret
    generate_jwt_secret()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Create test user script
    create_example_user_script()
    
    # Print next steps
    print_next_steps()
    
    if deps_ok:
        print("\n✅ Authentication setup complete!")
        print("Follow the manual steps above to finish integration.")
    else:
        print("\n⚠️  Install missing dependencies before proceeding.")

if __name__ == "__main__":
    main()