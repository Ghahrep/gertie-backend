# add_sample_data.py - Run this in your backend directory
from sqlalchemy.orm import Session
from db.session import SessionLocal
from db.models import User, Asset, Portfolio, Holding, create_sample_portfolio_data
from datetime import datetime

def add_sample_data():
    db = SessionLocal()
    
    try:
        # Find the admin user (ID: 2 based on your debug output)
        admin_user = db.query(User).filter(User.email == "admin@example.com").first()
        if not admin_user:
            print("Admin user not found!")
            return
        
        print(f"Found admin user: {admin_user.email} (ID: {admin_user.id})")
        
        # Check if admin already has portfolios
        existing_portfolios = db.query(Portfolio).filter(Portfolio.user_id == admin_user.id).all()
        if existing_portfolios:
            print(f"Admin user already has {len(existing_portfolios)} portfolios:")
            for p in existing_portfolios:
                print(f"  - {p.name}: {len(p.holdings)} holdings")
            return
        
        # Get sample data for admin user
        sample_data = create_sample_portfolio_data(admin_user.id)
        
        # Add assets first
        print("Adding assets...")
        for asset_data in sample_data["assets"]:
            existing_asset = db.query(Asset).filter(Asset.ticker == asset_data["ticker"]).first()
            if not existing_asset:
                asset = Asset(
                    ticker=asset_data["ticker"],
                    name=asset_data["name"],
                    asset_type=asset_data["asset_type"],
                    sector=asset_data["sector"],
                    current_price=asset_data["current_price"],
                    last_updated=datetime.utcnow()
                )
                db.add(asset)
                print(f"  Added asset: {asset_data['ticker']} - {asset_data['name']}")
            else:
                # Update current price if asset exists
                existing_asset.current_price = asset_data["current_price"]
                existing_asset.last_updated = datetime.utcnow()
                print(f"  Updated asset: {asset_data['ticker']} price to ${asset_data['current_price']}")
        
        db.commit()
        print("Assets committed to database")
        
        # Add portfolios
        print("\nAdding portfolios...")
        portfolios_created = {}
        for portfolio_data in sample_data["portfolios"]:
            portfolio = Portfolio(
                user_id=portfolio_data["user_id"],
                name=portfolio_data["name"],
                description=portfolio_data["description"],
                currency="USD",
                is_active=True,
                created_at=datetime.utcnow()
            )
            db.add(portfolio)
            db.flush()  # Get the ID
            portfolios_created[portfolio.name] = portfolio
            print(f"  Added portfolio: {portfolio.name} (ID: {portfolio.id})")
        
        db.commit()
        print("Portfolios committed to database")
        
        # Add holdings
        print("\nAdding holdings...")
        for holding_data in sample_data["sample_holdings"]:
            portfolio = portfolios_created[holding_data["portfolio_name"]]
            asset = db.query(Asset).filter(Asset.ticker == holding_data["ticker"]).first()
            
            if portfolio and asset:
                cost_basis = holding_data["shares"] * holding_data["purchase_price"]
                holding = Holding(
                    portfolio_id=portfolio.id,
                    asset_id=asset.id,
                    shares=holding_data["shares"],
                    purchase_price=holding_data["purchase_price"],
                    cost_basis=cost_basis,
                    purchase_date=datetime.utcnow()
                )
                db.add(holding)
                current_value = holding_data["shares"] * asset.current_price
                print(f"  Added: {holding_data['shares']} shares of {holding_data['ticker']} "
                      f"(Cost: ${cost_basis:,.0f}, Current: ${current_value:,.0f})")
        
        db.commit()
        print("Holdings committed to database")
        
        # Verify and show summary
        print("\n" + "="*50)
        print("SUCCESS! Sample data added for admin user")
        print("="*50)
        
        portfolios = db.query(Portfolio).filter(Portfolio.user_id == admin_user.id).all()
        for portfolio in portfolios:
            total_value = sum(h.current_value() for h in portfolio.holdings)
            total_cost = sum(h.cost_basis or 0 for h in portfolio.holdings)
            pnl = total_value - total_cost
            
            print(f"\n📊 {portfolio.name}:")
            print(f"   Holdings: {len(portfolio.holdings)} positions")
            print(f"   Total Value: ${total_value:,.0f}")
            print(f"   Total Cost: ${total_cost:,.0f}")
            print(f"   P&L: ${pnl:,.0f} ({(pnl/total_cost*100):+.1f}%)")
            
            for holding in portfolio.holdings:
                pnl_pct = holding.unrealized_pnl_percent()
                print(f"     • {holding.asset.ticker}: {holding.shares} shares "
                      f"(${holding.current_value():,.0f}, {pnl_pct:+.1f}%)")
        
        print(f"\n🎯 Total Portfolio Value: ${sum(sum(h.current_value() for h in p.holdings) for p in portfolios):,.0f}")
        print("\nNow refresh your frontend to see the portfolio data!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    add_sample_data()