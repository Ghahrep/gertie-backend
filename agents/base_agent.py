# agents/base_agent.py
"""
Base Agent Class for Enhanced Investment Committee
================================================

Provides common functionality for all enhanced agents.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

class BaseAgent:
    """Base class for all investment committee agents"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.agent_type = agent_name.lower().replace(" ", "_")
        self.logger = logging.getLogger(f"agents.{self.agent_type}")
        self.capabilities = []
        self.expertise_areas = []
        
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate standard error response"""
        return {
            "specialist": self.agent_type,
            "analysis_type": "error",
            "content": f"**{self.agent_name} - Technical Issue**\n\n{error_message}\n\nPlease try again or rephrase your question.",
            "analysis": {
                "riskScore": 50,
                "recommendation": "Try again",
                "confidence": 0,
                "specialist": self.agent_name
            },
            "error": error_message
        }
    
    async def analyze_query(self, query: str, portfolio_data: Dict, context: Dict) -> Dict:
        """Base analyze_query method - should be overridden by subclasses"""
        return {
            "specialist": self.agent_type,
            "analysis_type": "base",
            "content": f"**{self.agent_name}**\n\nBase agent response - this should be overridden by the enhanced agent.",
            "analysis": {
                "riskScore": 50,
                "recommendation": "Use enhanced agent implementation",
                "confidence": 50,
                "specialist": self.agent_name
            }
        }
    
    def get_capabilities(self) -> List[str]:
        """Return agent capabilities"""
        return self.capabilities
    
    def get_expertise_areas(self) -> List[str]:
        """Return agent expertise areas"""
        return self.expertise_areas