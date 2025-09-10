# migrate_user_auth.py - Database Migration for User Authentication
"""
Database Migration Script for User Authentication Fields
=======================================================

Adds is_active, role, and last_login fields to existing User table.
"""

from sqlalchemy import text, Column, Boolean, String, DateTime
from sqlalchemy.exc import OperationalError
from db.session import engine, SessionLocal
from db.models import User
import logging

logger = logging.getLogger(__name__)

def check_column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in the table"""
    try:
        with engine.connect() as conn:
            # SQLite-specific query to check column existence
            result = conn.execute(text(f"PRAGMA table_info({table_name})"))
            columns = [row[1] for row in result.fetchall()]
            return column_name in columns
    except Exception as e:
        logger.error(f"Error checking column existence: {e}")
        return False

def add_auth_columns():
    """Add authentication columns to users table"""
    print("Adding authentication columns to users table...")
    
    try:
        with engine.connect() as conn:
            # Add is_active column
            if not check_column_exists('users', 'is_active'):
                conn.execute(text("""
                    ALTER TABLE users 
                    ADD COLUMN is_active BOOLEAN DEFAULT TRUE NOT NULL
                """))
                print("✅ Added is_active column")
            else:
                print("⚠️  is_active column already exists")
            
            # Add role column
            if not check_column_exists('users', 'role'):
                conn.execute(text("""
                    ALTER TABLE users 
                    ADD COLUMN role VARCHAR DEFAULT 'user' NOT NULL
                """))
                print("✅ Added role column")
            else:
                print("⚠️  role column already exists")
            
            # Add last_login column
            if not check_column_exists('users', 'last_login'):
                conn.execute(text("""
                    ALTER TABLE users 
                    ADD COLUMN last_login DATETIME
                """))
                print("✅ Added last_login column")
            else:
                print("⚠️  last_login column already exists")
            
            conn.commit()
            print("✅ All authentication columns added successfully")
            
    except OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("⚠️  Some columns already exist, continuing...")
        else:
            logger.error(f"Database error during migration: {e}")
            raise
    except Exception as e:
        logger.error(f"Unexpected error during migration: {e}")
        raise

def update_existing_users():
    """Update existing users with default values"""
    print("Updating existing users with default values...")
    
    try:
        db = SessionLocal()
        
        # Update users that might have NULL values
        users = db.query(User).all()
        updated_count = 0
        
        for user in users:
            updated = False
            
            # Set default is_active if not set
            if not hasattr(user, 'is_active') or user.is_active is None:
                user.is_active = True
                updated = True
            
            # Set default role if not set
            if not hasattr(user, 'role') or not user.role:
                user.role = "user"
                updated = True
            
            # Ensure preferences exist
            if not user.preferences:
                user.preferences = {
                    "risk_tolerance": "moderate",
                    "default_analysis_depth": "standard",
                    "notification_settings": {"email": True, "webapp": True},
                    "ui_preferences": {
                        "theme": "light",
                        "default_currency": "USD",
                        "decimal_places": 2
                    }
                }
                updated = True
            
            if updated:
                updated_count += 1
        
        db.commit()
        print(f"✅ Updated {updated_count} existing users")
        
    except Exception as e:
        logger.error(f"Error updating existing users: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def create_admin_user():
    """Create an admin user if none exists"""
    print("Checking for admin user...")
    
    try:
        db = SessionLocal()
        
        # Check if admin user exists
        admin_user = db.query(User).filter(User.role == "admin").first()
        
        if not admin_user:
            # Create admin user
            import hashlib
            
            admin_email = "admin@example.com"
            admin_password = "admin123"  # Change this in production!
            hashed_password = hashlib.sha256(admin_password.encode()).hexdigest()
            
            admin_user = User(
                email=admin_email,
                hashed_password=hashed_password,
                role="admin",
                is_active=True,
                preferences={
                    "risk_tolerance": "aggressive",
                    "default_analysis_depth": "comprehensive",
                    "notification_settings": {"email": True, "webapp": True},
                    "ui_preferences": {
                        "theme": "dark",
                        "default_currency": "USD",
                        "decimal_places": 4
                    }
                }
            )
            
            db.add(admin_user)
            db.commit()
            
            print(f"✅ Created admin user: {admin_email}")
            print(f"⚠️  Default password: {admin_password} (CHANGE THIS!)")
        else:
            print(f"✅ Admin user already exists: {admin_user.email}")
    
    except Exception as e:
        logger.error(f"Error creating admin user: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def verify_migration():
    """Verify the migration was successful"""
    print("Verifying migration...")
    
    try:
        db = SessionLocal()
        
        # Check if we can query users with new fields
        users = db.query(User).all()
        
        for user in users:
            # Verify all required fields exist
            assert hasattr(user, 'is_active'), "is_active field missing"
            assert hasattr(user, 'role'), "role field missing"
            assert hasattr(user, 'last_login'), "last_login field missing"
            
            # Verify field types
            assert isinstance(user.is_active, bool), "is_active should be boolean"
            assert isinstance(user.role, str), "role should be string"
            assert user.last_login is None or hasattr(user.last_login, 'isoformat'), "last_login should be datetime or None"
        
        print(f"✅ Migration verified successfully for {len(users)} users")
        
        # Test new methods
        if users:
            test_user = users[0]
            display_name = test_user.display_name
            user_dict = test_user.to_dict()
            print(f"✅ New methods working - display_name: {display_name}")
        
    except Exception as e:
        logger.error(f"Migration verification failed: {e}")
        raise
    finally:
        db.close()

def main():
    """Run the complete migration"""
    print("🔄 User Authentication Migration")
    print("=" * 50)
    
    try:
        # Step 1: Add new columns
        add_auth_columns()
        
        # Step 2: Update existing users
        update_existing_users()
        
        # Step 3: Create admin user
        create_admin_user()
        
        # Step 4: Verify migration
        verify_migration()
        
        print("\n" + "=" * 50)
        print("✅ User authentication migration completed successfully!")
        print("\nNext steps:")
        print("1. Update your User model with the enhanced version")
        print("2. Test authentication with: python test_authentication.py") 
        print("3. Start the API and test login at /auth/login")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        print("Please check the logs and fix any issues before retrying.")
        return False
    
    return True

if __name__ == "__main__":
    main()