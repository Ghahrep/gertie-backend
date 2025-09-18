# Quick fix for sample data creation
# File: create_sample_data.py

import os
from sqlalchemy import create_engine, text

def create_sample_data():
    """Create sample data for testing the new AI tables"""
    
    database_url = os.getenv("DATABASE_URL", "sqlite:///./financial_platform.db")
    engine = create_engine(database_url)
    
    print("Creating sample data for advanced AI tables...")
    
    with engine.connect() as conn:
        try:
            # Get first user
            result = conn.execute(text("SELECT id FROM users LIMIT 1"))
            user = result.fetchone()
            
            if user:
                user_id = user[0]
                print(f"Using user ID: {user_id}")
                
                # Create sample user profile (using string formatting to avoid parameter issues)
                conn.execute(text(f"""
                    INSERT OR REPLACE INTO user_profiles 
                    (user_id, expertise_level, complexity_preference, total_conversations, agent_satisfaction_scores)
                    VALUES ({user_id}, 'intermediate', 0.7, 5, '{{"quantitative_analyst": 0.85, "portfolio_manager": 0.92}}')
                """))
                
                # Create sample system metrics
                conn.execute(text("""
                    INSERT INTO system_metrics 
                    (metric_type, metric_name, value, context)
                    VALUES ('system_health', 'database_migration_success', 1.0, '{"timestamp": "2025-01-01", "version": "2.0"}')
                """))
                
                # Create a sample conversation turn
                conn.execute(text(f"""
                    INSERT INTO conversation_turns
                    (user_id, conversation_id, user_query, agent_response, specialist, confidence)
                    VALUES ({user_id}, 'sample_conv_001', 'What is my portfolio performance?', 'Your portfolio has gained 12.5% this year...', 'portfolio_manager', 0.95)
                """))
                
                conn.commit()
                print("✓ Sample data created successfully!")
                print("  - User profile with learning preferences")
                print("  - System health metrics")
                print("  - Sample conversation turn")
                
                # Verify the data
                result = conn.execute(text("SELECT COUNT(*) FROM user_profiles WHERE user_id = ?"), (user_id,))
                profile_count = result.fetchone()[0]
                
                result = conn.execute(text("SELECT COUNT(*) FROM system_metrics"))
                metrics_count = result.fetchone()[0]
                
                result = conn.execute(text("SELECT COUNT(*) FROM conversation_turns WHERE user_id = ?"), (user_id,))
                turns_count = result.fetchone()[0]
                
                print(f"\nData verification:")
                print(f"  User profiles: {profile_count}")
                print(f"  System metrics: {metrics_count}")
                print(f"  Conversation turns: {turns_count}")
                
            else:
                print("No users found in database")
                
        except Exception as e:
            print(f"Error creating sample data: {str(e)}")

if __name__ == "__main__":
    create_sample_data()