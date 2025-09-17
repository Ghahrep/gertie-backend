# simple_check_user.py
from db.session import SessionLocal
from db.models import User

db = SessionLocal()

# Check if the user exists
user = db.query(User).filter(User.email == "integration_test@example.com").first()

if user:
    print(f"✅ User found: {user.email}")
    print(f"   Is active: {user.is_active}")
    print(f"   Role: {user.role}")
    print(f"   Password hash: {user.hashed_password}")
else:
    print("❌ User not found in database")

# List all users to see what exists
print("\nAll users in database:")
all_users = db.query(User).all()
for u in all_users:
    print(f"   Email: {u.email}, Active: {u.is_active}, Role: {u.role}")

db.close()