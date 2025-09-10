# simple_create_users.py - Create test users with proper password hashing
"""
Create test users for authentication testing
"""

from sqlalchemy.orm import Session
from db.session import SessionLocal
from db.models import User
from datetime import datetime

# Try different password hashing approaches
def hash_password(password: str) -> str:
    """Hash password using available method"""
    try:
        # Try bcrypt first (most common)
        import bcrypt
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    except ImportError:
        pass
    
    try:
        # Try passlib (another common option)
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return pwd_context.hash(password)
    except ImportError:
        pass
    
    try:
        # Try werkzeug (if available)
        from werkzeug.security import generate_password_hash
        return generate_password_hash(password)
    except ImportError:
        pass
    
    # Fallback to simple hash (NOT SECURE - only for testing)
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

def check_existing_users():
    """Check what users already exist"""
    db = SessionLocal()
    try:
        users = db.query(User).all()
        print(f"Found {len(users)} existing users:")
        for user in users:
            print(f"  - {user.email} (role: {getattr(user, 'role', 'unknown')})")
        return users
    except Exception as e:
        print(f"Error checking users: {e}")
        return []
    finally:
        db.close()

def create_test_user(email: str, password: str, role: str = "user"):
    """Create a single test user"""
    db = SessionLocal()
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == email).first()
        
        if existing_user:
            print(f"User {email} already exists")
            return existing_user
        
        # Hash the password
        hashed_password = hash_password(password)
        
        # Create user object - adapt fields based on your User model
        user_data = {
            "email": email,
            "hashed_password": hashed_password,
            "is_active": True
        }
        
        # Add role if the field exists
        if hasattr(User, 'role'):
            user_data["role"] = role
        
        # Add created_at if the field exists
        if hasattr(User, 'created_at'):
            user_data["created_at"] = datetime.now()
        
        test_user = User(**user_data)
        
        db.add(test_user)
        db.commit()
        db.refresh(test_user)
        
        print(f"✅ Created user: {email} / {password} (role: {role})")
        return test_user
        
    except Exception as e:
        print(f"❌ Error creating user {email}: {e}")
        db.rollback()
        return None
    finally:
        db.close()

def verify_user_login(email: str, password: str):
    """Test if user can login with given credentials"""
    try:
        from auth.middleware import authenticate_user
        db = SessionLocal()
        user = authenticate_user(email, password, db)
        db.close()
        
        if user:
            print(f"✅ Login test successful for {email}")
            return True
        else:
            print(f"❌ Login test failed for {email}")
            return False
    except Exception as e:
        print(f"❌ Login test error for {email}: {e}")
        return False

def main():
    print("🔧 Creating Test Users for Authentication")
    print("=" * 50)
    
    # Check existing users first
    existing_users = check_existing_users()
    
    # Create test users
    test_users = [
        ("user1@example.com", "user123", "user"),
        ("admin@example.com", "admin123", "admin"),
        ("user2@example.com", "user456", "user")
    ]
    
    print(f"\nCreating {len(test_users)} test users...")
    
    created_users = []
    for email, password, role in test_users:
        user = create_test_user(email, password, role)
        if user:
            created_users.append((email, password))
    
    # Test login for created users
    if created_users:
        print(f"\nTesting login for {len(created_users)} users...")
        for email, password in created_users:
            verify_user_login(email, password)
    
    print("\n" + "=" * 50)
    print("✅ User creation complete!")
    print("\nTest credentials:")
    print("- User: user1@example.com / user123")
    print("- Admin: admin@example.com / admin123")
    print("- User2: user2@example.com / user456")
    print("\nNow run: python simple_test_runner.py")

if __name__ == "__main__":
    main()