# migrations/add_conversation_tables.py
"""Add conversation history tables

Revision ID: add_conversation_tables
Revises: [your_previous_migration]
Create Date: 2025-09-12

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = 'add_conversation_tables'
down_revision = None  # Replace with your latest migration
branch_labels = None
depends_on = None

def upgrade():
    # Create chat_conversations table
    op.create_table('chat_conversations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('portfolio_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('last_message_at', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('summary', sa.String(length=500), nullable=True),
        sa.Column('total_messages', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['portfolio_id'], ['portfolios.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_chat_conversations_id'), 'chat_conversations', ['id'], unique=False)

    # Create chat_messages table
    op.create_table('chat_messages',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('conversation_id', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(length=20), nullable=False),
        sa.Column('content', sa.String(length=10000), nullable=False),
        sa.Column('specialist_id', sa.String(length=50), nullable=True),
        sa.Column('routing_confidence', sa.Float(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('portfolio_context', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['conversation_id'], ['chat_conversations.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_chat_messages_id'), 'chat_messages', ['id'], unique=False)

def downgrade():
    op.drop_index(op.f('ix_chat_messages_id'), table_name='chat_messages')
    op.drop_table('chat_messages')
    op.drop_index(op.f('ix_chat_conversations_id'), table_name='chat_conversations')
    op.drop_table('chat_conversations')