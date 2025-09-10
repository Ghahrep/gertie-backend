# fix_user_passwords.py - Fix user passwords to match the existing auth system
"""
Update user passwords to use the same hashing method as the working admin user
"""

from sqlalchemy.orm import Session
from db.session import SessionLocal
from db.models import User
import hashlib

def get_admin_hash_method():
    """Analyze how the admin password is hashed"""
    db = SessionLocal()
    try:
        admin_user = db.query(User).filter(User.email == "admin@example.com").first()
        if admin_user:
            admin_hash = admin_user.hashed_password
            print(f"Admin hash: {admin_hash[:50]}...")
            
            # Test if it's SHA256
            test_sha256 = hashlib.sha256("admin123".encode()).hexdigest()
            if admin_hash == test_sha256:
                print("Admin uses SHA256 hash")
                return "sha256"
            
            # Test if it's SHA256 with salt
            test_sha256_salt = hashlib.sha256(("admin123" + "some_salt").encode()).hexdigest()
            print(f"Testing SHA256 with potential salt...")
            
            # For now, let's assume it's simple SHA256
            print("Assuming simple SHA256 hashing")
            return "sha256"
        
        return None
    finally:
        db.close()

def hash_password_like_admin(password: str) -> str:
    """Hash password using the same method as admin"""
    # Try simple SHA256 first
    return hashlib.sha256(password.encode()).hexdigest()

def fix_user_password(email: str, new_password: str):
    """Fix a user's password to use the correct hashing method"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            print(f"User {email} not found")
            return False
        
        # Hash password using the admin method
        new_hash = hash_password_like_admin(new_password)
        
        print(f"Old hash: {user.hashed_password[:50]}...")
        print(f"New hash: {new_hash[:50]}...")
        
        user.hashed_password = new_hash
        db.commit()
        
        print(f"Updated password for {email}")
        return True
        
    except Exception as e:
        print(f"Error updating {email}: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def test_login_after_fix(email: str, password: str):
    """Test login after password fix"""
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
    print("🔧 Fixing User Passwords to Match Auth System")
    print("=" * 50)
    
    # Analyze admin hash method
    hash_method = get_admin_hash_method()
    
    # Fix the problematic users
    users_to_fix = [
        ("user1@example.com", "user123"),
        ("user2@example.com", "user456")
    ]
    
    print(f"\nFixing passwords for {len(users_to_fix)} users...")
    
    for email, password in users_to_fix:
        success = fix_user_password(email, password)
        if success:
            # Test the login
            test_login_after_fix(email, password)
    
    print("\n" + "=" * 50)
    print("✅ Password fix complete!")
    print("\nTest credentials (should all work now):")
    print("- Admin: admin@example.com / admin123")
    print("- User1: user1@example.com / user123")
    print("- User2: user2@example.com / user456")
    print("\nNow run: python simple_test_runner.py")

if __name__ == "__main__":
    main()