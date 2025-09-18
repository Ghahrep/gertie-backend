# Direct Database Migration Script (No Alembic Required)
# File: migrate_database.py

"""
Direct database migration script to add advanced AI models
Works with existing SQLAlchemy setup without requiring Alembic
"""

import os
import sys
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Add your project root to the path so we can import models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_database_engine():
    """Get database engine from environment or use SQLite default"""
    database_url = os.getenv("DATABASE_URL", "sqlite:///./financial_platform.db")
    print(f"Connecting to database: {database_url}")
    return create_engine(database_url)

def check_existing_tables(engine):
    """Check which tables already exist"""
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    print(f"Existing tables: {existing_tables}")
    return existing_tables

def create_advanced_ai_tables(engine):
    """Create all advanced AI tables using raw SQL"""
    
    # Table creation SQL statements
    tables_sql = {
        'conversation_turns': """
        CREATE TABLE IF NOT EXISTS conversation_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            conversation_id VARCHAR(100) NOT NULL,
            user_query TEXT NOT NULL,
            agent_response TEXT NOT NULL,
            specialist VARCHAR(50) NOT NULL,
            confidence FLOAT DEFAULT 0.0,
            risk_score INTEGER DEFAULT 50,
            collaboration_involved BOOLEAN DEFAULT FALSE,
            secondary_specialists JSON DEFAULT '[]',
            portfolio_context JSON,
            semantic_embedding JSON,
            user_satisfaction FLOAT,
            user_engagement_time INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """,
        
        'user_profiles': """
        CREATE TABLE IF NOT EXISTS user_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL UNIQUE,
            expertise_level VARCHAR(20) DEFAULT 'beginner',
            complexity_preference FLOAT DEFAULT 0.5,
            collaboration_preference FLOAT DEFAULT 0.5,
            agent_satisfaction_scores JSON DEFAULT '{}',
            total_conversations INTEGER DEFAULT 0,
            total_turns INTEGER DEFAULT 0,
            avg_session_length FLOAT DEFAULT 0.0,
            preferred_response_length VARCHAR(20) DEFAULT 'medium',
            preferred_detail_level VARCHAR(20) DEFAULT 'standard',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """,
        
        'proactive_insights': """
        CREATE TABLE IF NOT EXISTS proactive_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            portfolio_id INTEGER,
            insight_id VARCHAR(100) NOT NULL UNIQUE,
            insight_type VARCHAR(50) NOT NULL,
            priority VARCHAR(20) NOT NULL,
            title VARCHAR(200) NOT NULL,
            description TEXT NOT NULL,
            recommendations JSON DEFAULT '[]',
            conversation_starters JSON DEFAULT '[]',
            data JSON DEFAULT '{}',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME,
            is_active BOOLEAN DEFAULT TRUE,
            view_count INTEGER DEFAULT 0,
            click_count INTEGER DEFAULT 0,
            dismiss_count INTEGER DEFAULT 0,
            action_taken BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
        )
        """,
        
        'insight_engagements': """
        CREATE TABLE IF NOT EXISTS insight_engagements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            insight_id INTEGER NOT NULL,
            engagement_type VARCHAR(50) NOT NULL,
            engagement_data JSON DEFAULT '{}',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (insight_id) REFERENCES proactive_insights(id)
        )
        """,
        
        'portfolio_snapshots': """
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL,
            total_value FLOAT NOT NULL,
            holdings_data JSON NOT NULL,
            allocation_weights JSON DEFAULT '{}',
            risk_score FLOAT DEFAULT 0.0,
            concentration_ratio FLOAT DEFAULT 0.0,
            volatility_estimate FLOAT DEFAULT 0.0,
            market_conditions JSON DEFAULT '{}',
            snapshot_type VARCHAR(50) DEFAULT 'scheduled',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
        )
        """,
        
        'enhanced_conversations': """
        CREATE TABLE IF NOT EXISTS enhanced_conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            portfolio_id INTEGER NOT NULL,
            conversation_id VARCHAR(100) NOT NULL UNIQUE,
            session_id VARCHAR(100),
            user_query TEXT NOT NULL,
            agent_response TEXT NOT NULL,
            specialist VARCHAR(50) NOT NULL,
            routing_confidence FLOAT DEFAULT 0.0,
            analysis_confidence FLOAT DEFAULT 0.0,
            collaboration_used BOOLEAN DEFAULT FALSE,
            specialists_consulted JSON DEFAULT '[]',
            tools_used JSON DEFAULT '[]',
            portfolio_context JSON DEFAULT '{}',
            proactive_insights_count INTEGER DEFAULT 0,
            related_insights_count INTEGER DEFAULT 0,
            response_time_ms INTEGER DEFAULT 0,
            tokens_used INTEGER DEFAULT 0,
            user_rating FLOAT,
            user_feedback_text TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
        )
        """,
        
        'system_metrics': """
        CREATE TABLE IF NOT EXISTS system_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_type VARCHAR(50) NOT NULL,
            metric_name VARCHAR(100) NOT NULL,
            value FLOAT NOT NULL,
            count INTEGER DEFAULT 1,
            context JSON DEFAULT '{}',
            user_id INTEGER,
            recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    }
    
    # Index creation SQL
    indexes_sql = {
        'conversation_turns': [
            "CREATE INDEX IF NOT EXISTS idx_conversation_turns_user_id ON conversation_turns(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_conversation_turns_conversation_id ON conversation_turns(conversation_id)",
            "CREATE INDEX IF NOT EXISTS idx_conversation_turns_specialist ON conversation_turns(specialist)",
            "CREATE INDEX IF NOT EXISTS idx_conversation_turns_created_at ON conversation_turns(created_at)"
        ],
        'user_profiles': [
            "CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_profiles_expertise_level ON user_profiles(expertise_level)"
        ],
        'proactive_insights': [
            "CREATE INDEX IF NOT EXISTS idx_proactive_insights_user_id ON proactive_insights(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_proactive_insights_portfolio_id ON proactive_insights(portfolio_id)",
            "CREATE INDEX IF NOT EXISTS idx_proactive_insights_insight_type ON proactive_insights(insight_type)",
            "CREATE INDEX IF NOT EXISTS idx_proactive_insights_priority ON proactive_insights(priority)",
            "CREATE INDEX IF NOT EXISTS idx_proactive_insights_is_active ON proactive_insights(is_active)"
        ],
        'insight_engagements': [
            "CREATE INDEX IF NOT EXISTS idx_insight_engagements_user_id ON insight_engagements(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_insight_engagements_insight_id ON insight_engagements(insight_id)",
            "CREATE INDEX IF NOT EXISTS idx_insight_engagements_engagement_type ON insight_engagements(engagement_type)"
        ],
        'portfolio_snapshots': [
            "CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_portfolio_id ON portfolio_snapshots(portfolio_id)",
            "CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_created_at ON portfolio_snapshots(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_snapshot_type ON portfolio_snapshots(snapshot_type)"
        ],
        'enhanced_conversations': [
            "CREATE INDEX IF NOT EXISTS idx_enhanced_conversations_user_id ON enhanced_conversations(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_enhanced_conversations_portfolio_id ON enhanced_conversations(portfolio_id)",
            "CREATE INDEX IF NOT EXISTS idx_enhanced_conversations_conversation_id ON enhanced_conversations(conversation_id)",
            "CREATE INDEX IF NOT EXISTS idx_enhanced_conversations_specialist ON enhanced_conversations(specialist)"
        ],
        'system_metrics': [
            "CREATE INDEX IF NOT EXISTS idx_system_metrics_metric_type ON system_metrics(metric_type)",
            "CREATE INDEX IF NOT EXISTS idx_system_metrics_metric_name ON system_metrics(metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_system_metrics_recorded_at ON system_metrics(recorded_at)",
            "CREATE INDEX IF NOT EXISTS idx_system_metrics_user_id ON system_metrics(user_id)"
        ]
    }
    
    print("Creating advanced AI tables...")
    
    with engine.connect() as conn:
        # Create tables
        for table_name, sql in tables_sql.items():
            try:
                print(f"  Creating table: {table_name}")
                conn.execute(text(sql))
                conn.commit()
                print(f"  ✓ Table {table_name} created successfully")
            except Exception as e:
                print(f"  ✗ Error creating table {table_name}: {str(e)}")
                
        # Create indexes
        for table_name, index_list in indexes_sql.items():
            print(f"  Creating indexes for {table_name}")
            for index_sql in index_list:
                try:
                    conn.execute(text(index_sql))
                    conn.commit()
                except Exception as e:
                    print(f"    Warning: Index creation failed: {str(e)}")
    
    print("Advanced AI tables creation completed!")

def verify_migration(engine):
    """Verify that all tables were created successfully"""
    print("\nVerifying migration...")
    
    required_tables = [
        'conversation_turns',
        'user_profiles', 
        'proactive_insights',
        'insight_engagements',
        'portfolio_snapshots',
        'enhanced_conversations',
        'system_metrics'
    ]
    
    existing_tables = check_existing_tables(engine)
    
    success_count = 0
    for table in required_tables:
        if table in existing_tables:
            print(f"  ✓ {table}")
            success_count += 1
        else:
            print(f"  ✗ {table} - MISSING")
    
    print(f"\nMigration verification: {success_count}/{len(required_tables)} tables created")
    return success_count == len(required_tables)

def test_basic_operations(engine):
    """Test basic CRUD operations on new tables"""
    print("\nTesting basic operations...")
    
    with engine.connect() as conn:
        try:
            # Test user_profiles table
            conn.execute(text("""
                INSERT OR IGNORE INTO user_profiles (user_id, expertise_level) 
                VALUES (999, 'test')
            """))
            
            result = conn.execute(text("SELECT * FROM user_profiles WHERE user_id = 999"))
            if result.fetchone():
                print("  ✓ user_profiles: INSERT/SELECT working")
                conn.execute(text("DELETE FROM user_profiles WHERE user_id = 999"))
                conn.commit()
            
            # Test system_metrics table
            conn.execute(text("""
                INSERT INTO system_metrics (metric_type, metric_name, value) 
                VALUES ('test', 'migration_test', 1.0)
            """))
            
            result = conn.execute(text("SELECT * FROM system_metrics WHERE metric_type = 'test'"))
            if result.fetchone():
                print("  ✓ system_metrics: INSERT/SELECT working")
                conn.execute(text("DELETE FROM system_metrics WHERE metric_type = 'test'"))
                conn.commit()
                
            print("  ✓ Basic operations test passed")
            
        except Exception as e:
            print(f"  ✗ Basic operations test failed: {str(e)}")

def create_sample_data(engine):
    """Create some sample data for testing"""
    print("\nCreating sample data...")
    
    with engine.connect() as conn:
        try:
            # Check if we have any users first
            result = conn.execute(text("SELECT id FROM users LIMIT 1"))
            user = result.fetchone()
            
            if user:
                user_id = user[0]
                print(f"  Found user ID: {user_id}")
                
                # Create sample user profile
                conn.execute(text("""
                    INSERT OR REPLACE INTO user_profiles 
                    (user_id, expertise_level, complexity_preference, total_conversations)
                    VALUES (?, 'intermediate', 0.7, 5)
                """), (user_id,))
                
                # Create sample system metrics
                conn.execute(text("""
                    INSERT INTO system_metrics 
                    (metric_type, metric_name, value, context)
                    VALUES ('system_health', 'database_migration_success', 1.0, '{"timestamp": "' || datetime('now') || '"}')
                """))
                
                conn.commit()
                print("  ✓ Sample data created successfully")
                
            else:
                print("  ! No users found - skipping sample data creation")
                print("    You can create sample data after adding users to the system")
                
        except Exception as e:
            print(f"  ✗ Sample data creation failed: {str(e)}")

def main():
    """Main migration function"""
    print("=" * 60)
    print("AI Investment Committee - Database Migration")
    print("Adding Advanced AI Models (Week 1 & 2 Capabilities)")
    print("=" * 60)
    
    try:
        # Get database engine
        engine = get_database_engine()
        
        # Check existing tables
        print("\nStep 1: Checking existing database structure...")
        existing_tables = check_existing_tables(engine)
        
        if 'users' not in existing_tables:
            print("WARNING: 'users' table not found!")
            print("Please make sure your base database schema is set up first.")
            return False
        
        # Create new tables
        print("\nStep 2: Creating advanced AI tables...")
        create_advanced_ai_tables(engine)
        
        # Verify migration
        print("\nStep 3: Verifying migration...")
        if verify_migration(engine):
            print("✓ Migration completed successfully!")
        else:
            print("✗ Migration verification failed!")
            return False
        
        # Test basic operations
        print("\nStep 4: Testing basic operations...")
        test_basic_operations(engine)
        
        # Create sample data
        print("\nStep 5: Creating sample data...")
        create_sample_data(engine)
        
        print("\n" + "=" * 60)
        print("MIGRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Your database now supports:")
        print("✓ Semantic conversation memory (Week 1)")
        print("✓ User learning profiles (Week 1)")
        print("✓ Proactive insights engine (Week 2)")
        print("✓ Portfolio drift detection (Week 2)")
        print("✓ Advanced analytics and metrics")
        print("\nNext steps:")
        print("1. Update your models.py file with the enhanced version")
        print("2. Test your Week 1 & 2 AI capabilities")
        print("3. Verify chat endpoints work with new models")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Migration failed with error: {str(e)}")
        print("Please check your database connection and try again.")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)