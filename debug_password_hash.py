# Debug password hashing compatibility

from auth.middleware import authenticate_user
from db.session import SessionLocal
from db.models import User

def test_password_verification():
    db = SessionLocal()
    
    # Test the working admin user
    admin_user = db.query(User).filter(User.email == "admin@example.com").first()
    if admin_user:
        print(f"Admin user hash: {admin_user.hashed_password[:50]}...")
        print(f"Admin hash method: {'bcrypt' if admin_user.hashed_password.startswith('$2b$') else 'other'}")
    
    # Test the new user1
    user1 = db.query(User).filter(User.email == "user1@example.com").first()
    if user1:
        print(f"User1 hash: {user1.hashed_password[:50]}...")
        print(f"User1 hash method: {'bcrypt' if user1.hashed_password.startswith('$2b$') else 'other'}")
    
    # Test manual authentication
    print("\nTesting manual authentication:")
    result = authenticate_user("admin@example.com", "admin123", db)
    print(f"Admin login: {'SUCCESS' if result else 'FAILED'}")
    
    result = authenticate_user("user1@example.com", "user123", db)
    print(f"User1 login: {'SUCCESS' if result else 'FAILED'}")
    
    db.close()

if __name__ == "__main__":
    test_password_verification()