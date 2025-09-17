# create_test_user.py
from db.session import SessionLocal
from db.models import User
from datetime import datetime

db = SessionLocal()

# Check if test user already exists
existing_user = db.query(User).filter(User.email == "integration_test@example.com").first()

if not existing_user:
    # Create test user with a simple password hash (you might need to adjust this based on your auth system)
    test_user = User(
        email="integration_test@example.com",
        hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # password: "secret"
        is_active=True,
        role="user",
        created_at=datetime.utcnow()
    )
    db.add(test_user)
    db.commit()
    print("Test user created successfully with email: integration_test@example.com")
    print("Password: secret")
else:
    print("Test user already exists")

db.close()