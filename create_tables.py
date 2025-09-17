# create_tables.py
import sys
import os

# Add the current directory to Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db.session import engine
from db.models import Base

def create_tables():
    """Create all tables including the new conversation tables"""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")
    
    # Verify tables were created
    import sqlite3
    conn = sqlite3.connect('financial_platform.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"✅ Found {len(tables)} tables:", [table[0] for table in tables])
    conn.close()

if __name__ == "__main__":
    create_tables()