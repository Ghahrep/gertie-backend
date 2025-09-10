# setup_database.py - Initialize Database with Sample Data

"""
Run this script to set up your clean database with sample data for testing.
Usage: python setup_database.py
"""

import sys
import os
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db.session import SessionLocal, init_db, create_tables
from db.models import User, Asset, Portfolio, Holding, Alert, create_sample_portfolio_data

def create_sample_user():
    """Create a sample user for testing"""
    db = SessionLocal()
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == "demo@financial-platform.com").first()
        if existing_user:
            print("Demo user already exists")
            return existing_user
        
        # Create demo user
        demo_user = User(
            email="demo@financial-platform.com",
            hashed_password=generate_password_hash("demo123"),  # Simple password for demo
            preferences={
                "risk_tolerance": "moderate",
                "default_analysis_depth": "standard",
                "notification_settings": {"email": True, "webapp": True}
            }
        )
        
        db.add(demo_user)
        db.commit()
        db.refresh(demo_user)
        
        print(f"Created demo user: {demo_user.email} (ID: {demo_user.id})")
        return demo_user
        
    except Exception as e:
        print(f"Error creating user: {e}")
        db.rollback()
        return None
    finally:
        db.close()

def populate_assets():
    """Populate the database with sample assets"""
    db = SessionLocal()
    try:
        # Check if assets already exist
        existing_assets = db.query(Asset).count()
        if existing_assets > 0:
            print(f"Assets already exist ({existing_assets} assets)")
            return
        
        sample_data = create_sample_portfolio_data(1)  # user_id doesn't matter for assets
        
        for asset_data in sample_data["assets"]:
            asset = Asset(
                ticker=asset_data["ticker"],
                name=asset_data["name"],
                asset_type=asset_data["asset_type"],
                sector=asset_data["sector"],
                current_price=asset_data["current_price"],
                last_updated=datetime.utcnow()
            )
            db.add(asset)
        
        db.commit()
        print(f"Created {len(sample_data['assets'])} sample assets")
        
    except Exception as e:
        print(f"Error creating assets: {e}")
        db.rollback()
    finally:
        db.close()

def create_sample_portfolios(user_id: int):
    """Create sample portfolios for the user"""
    db = SessionLocal()
    try:
        # Check if portfolios already exist for this user
        existing_portfolios = db.query(Portfolio).filter(Portfolio.user_id == user_id).count()
        if existing_portfolios > 0:
            print(f"Portfolios already exist for user {user_id}")
            return
        
        sample_data = create_sample_portfolio_data(user_id)
        
        # Create portfolios
        portfolios = {}
        for portfolio_data in sample_data["portfolios"]:
            portfolio = Portfolio(
                user_id=portfolio_data["user_id"],
                name=portfolio_data["name"],
                description=portfolio_data["description"]
            )
            db.add(portfolio)
            db.flush()  # Get the ID without committing
            portfolios[portfolio.name] = portfolio
        
        # Create holdings
        for holding_data in sample_data["sample_holdings"]:
            portfolio = portfolios[holding_data["portfolio_name"]]
            asset = db.query(Asset).filter(Asset.ticker == holding_data["ticker"]).first()
            
            if asset:
                cost_basis = holding_data["shares"] * holding_data["purchase_price"]
                holding = Holding(
                    portfolio_id=portfolio.id,
                    asset_id=asset.id,
                    shares=holding_data["shares"],
                    purchase_price=holding_data["purchase_price"],
                    cost_basis=cost_basis,
                    purchase_date=datetime.utcnow() - timedelta(days=30)  # 30 days ago
                )
                db.add(holding)
        
        db.commit()
        print(f"Created {len(sample_data['portfolios'])} portfolios with holdings")
        
    except Exception as e:
        print(f"Error creating portfolios: {e}")
        db.rollback()
    finally:
        db.close()

def create_sample_alerts(user_id: int):
    """Create sample alerts for the user"""
    db = SessionLocal()
    try:
        # Get user's first portfolio
        portfolio = db.query(Portfolio).filter(Portfolio.user_id == user_id).first()
        if not portfolio:
            print("No portfolio found for alerts")
            return
        
        # Sample alerts
        alerts_data = [
            {
                "alert_type": "price_change",
                "condition": {
                    "ticker": "AAPL",
                    "operator": ">",
                    "value": 180.0,
                    "description": "AAPL price above $180"
                }
            },
            {
                "alert_type": "portfolio_value",
                "condition": {
                    "metric": "total_value",
                    "operator": "<",
                    "value": 100000,
                    "description": "Portfolio value below $100,000"
                }
            },
            {
                "alert_type": "risk_threshold",
                "condition": {
                    "metric": "volatility",
                    "operator": ">",
                    "value": 20.0,
                    "description": "Portfolio volatility above 20%"
                }
            }
        ]
        
        for alert_data in alerts_data:
            alert = Alert(
                user_id=user_id,
                portfolio_id=portfolio.id,
                alert_type=alert_data["alert_type"],
                condition=alert_data["condition"],
                notification_channels=["webapp", "email"]
            )
            db.add(alert)
        
        db.commit()
        print(f"Created {len(alerts_data)} sample alerts")
        
    except Exception as e:
        print(f"Error creating alerts: {e}")
        db.rollback()
    finally:
        db.close()

def main():
    """Main setup function"""
    print("Setting up Financial Platform Clean Database...")
    print("=" * 50)
    
    # Initialize database tables
    print("1. Creating database tables...")
    create_tables()
    
    # Create sample user
    print("2. Creating demo user...")
    demo_user = create_sample_user()
    if not demo_user:
        print("Failed to create demo user. Exiting.")
        return
    
    # Populate assets
    print("3. Creating sample assets...")
    populate_assets()
    
    # Create sample portfolios
    print("4. Creating sample portfolios...")
    create_sample_portfolios(demo_user.id)
    
    # Create sample alerts
    print("5. Creating sample alerts...")
    create_sample_alerts(demo_user.id)
    
    print("=" * 50)
    print("Database setup complete!")
    print("\nDemo Login Credentials:")
    print("Email: demo@financial-platform.com")
    print("Password: demo123")
    print("\nDatabase file: financial_platform.db")
    print("\nTo explore the data:")
    print("1. Install a SQLite browser or use command line")
    print("2. Open financial_platform.db")
    print("3. Browse the 5 tables: users, assets, portfolios, holdings, alerts")

if __name__ == "__main__":
    # Install werkzeug if not already installed (for password hashing)
    try:
        from werkzeug.security import generate_password_hash
    except ImportError:
        print("Installing werkzeug for password hashing...")
        os.system("pip install werkzeug")
        from werkzeug.security import generate_password_hash
    
    main()