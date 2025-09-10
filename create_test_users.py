# create_test_users.py - Create Test Users
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
