# agents/__init__.py
"""
Investment Committee Agents Package
"""

try:
    from .enhanced_investment_committee_manager import EnhancedInvestmentCommitteeManager
    # Export the working manager
    committee_manager = EnhancedInvestmentCommitteeManager
except ImportError:
    # Fallback if manager can't be imported
    committee_manager = None

__version__ = "1.0.0"