# verify_data.py

from db.session import SessionLocal
from db.models import User, Portfolio, Asset, Holding, calculate_portfolio_summary

print("Connecting to the database...")
db = SessionLocal()

try:
    # Get demo user
    user = db.query(User).first()
    if user:
        print(f'Successfully found Demo User: {user.email}')
    else:
        print('Could not find a user. Is the database seeded with data?')

    # Get portfolios with summaries
    portfolios = db.query(Portfolio).all()
    if portfolios:
        print(f'Found {len(portfolios)} portfolios.')
        for portfolio in portfolios:
            summary = calculate_portfolio_summary(portfolio)
            print(f'\n-> Portfolio: {summary["name"]}:')
            print(f'   Total Value: ${summary["total_value"]:,.2f}')
            print(f'   Holdings Count: {summary["holdings_count"]}')
            print(f'   Profit/Loss: ${summary["total_pnl"]:,.2f} ({summary["total_pnl_percent"]:.2f}%)')
    else:
        print("No portfolios found in the database.")

finally:
    print("\nClosing database connection.")
    db.close()