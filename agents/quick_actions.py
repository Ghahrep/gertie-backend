# agents/quick_actions.py - Quick Actions System for Investment Committee

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class QuickAction:
    """Quick action definition"""
    id: str
    label: str
    specialist: str
    category: str
    description: str
    priority: int = 1
    context_requirements: Optional[Dict[str, Any]] = None

class QuickActionsManager:
    """Manages contextual quick actions for the Investment Committee"""
    
    def __init__(self):
        self.actions = self._initialize_actions()
    
    def _initialize_actions(self) -> Dict[str, QuickAction]:
        """Initialize all available quick actions"""
        actions = {}
        
        # Risk Management Actions (Quantitative Analyst)
        risk_actions = [
            QuickAction(
                id="calculate_var",
                label="Calculate Value at Risk",
                specialist="quantitative_analyst",
                category="risk",
                description="Calculate VaR for different confidence levels"
            ),
            QuickAction(
                id="stress_test",
                label="Run Stress Test",
                specialist="quantitative_analyst", 
                category="risk",
                description="Test portfolio against market scenarios",
                priority=2
            ),
            QuickAction(
                id="correlation_analysis",
                label="Correlation Matrix",
                specialist="quantitative_analyst",
                category="risk", 
                description="Analyze correlations between holdings"
            ),
            QuickAction(
                id="volatility_forecast",
                label="Volatility Forecast",
                specialist="quantitative_analyst",
                category="risk",
                description="GARCH volatility forecasting"
            )
        ]
        
        # Portfolio Management Actions
        portfolio_actions = [
            QuickAction(
                id="rebalancing_check",
                label="Check Rebalancing",
                specialist="portfolio_manager",
                category="portfolio",
                description="Analyze portfolio allocation vs targets"
            ),
            QuickAction(
                id="optimization",
                label="Portfolio Optimization",
                specialist="portfolio_manager",
                category="portfolio",
                description="Optimize allocation for risk/return"
            ),
            QuickAction(
                id="performance_review",
                label="Performance Review",
                specialist="portfolio_manager",
                category="portfolio",
                description="Detailed performance attribution analysis"
            ),
            QuickAction(
                id="trade_ideas",
                label="Generate Trade Ideas",
                specialist="portfolio_manager",
                category="portfolio",
                description="AI-generated trade recommendations"
            )
        ]
        
        # Strategic Actions (CIO)
        strategic_actions = [
            QuickAction(
                id="strategic_review",
                label="Strategic Review",
                specialist="cio",
                category="strategy",
                description="High-level portfolio strategy assessment"
            ),
            QuickAction(
                id="market_outlook",
                label="Market Outlook",
                specialist="cio",
                category="strategy",
                description="Current market regime and outlook analysis"
            ),
            QuickAction(
                id="asset_allocation",
                label="Asset Allocation Review",
                specialist="cio",
                category="strategy",
                description="Strategic asset allocation analysis"
            ),
            QuickAction(
                id="sector_analysis",
                label="Sector Analysis",
                specialist="cio",
                category="strategy",
                description="Sector allocation and opportunity analysis"
            )
        ]
        
        # Behavioral Actions (Behavioral Coach)
        behavioral_actions = [
            QuickAction(
                id="bias_assessment",
                label="Bias Assessment",
                specialist="behavioral_coach",
                category="behavioral",
                description="Identify potential behavioral biases"
            ),
            QuickAction(
                id="emotional_check",
                label="Emotional Check-in",
                specialist="behavioral_coach",
                category="behavioral",
                description="Assess emotional state and decision-making"
            ),
            QuickAction(
                id="decision_framework",
                label="Decision Framework",
                specialist="behavioral_coach",
                category="behavioral",
                description="Structured decision-making guidance"
            ),
            QuickAction(
                id="discipline_review",
                label="Discipline Review",
                specialist="behavioral_coach",
                category="behavioral",
                description="Review adherence to investment discipline"
            )
        ]
        
        # Combine all actions
        all_actions = risk_actions + portfolio_actions + strategic_actions + behavioral_actions
        
        for action in all_actions:
            actions[action.id] = action
            
        return actions
    
    def get_contextual_actions(
        self, 
        portfolio_context: Dict[str, Any],
        conversation_history: List[Dict[str, str]] = None,
        max_actions: int = 6
    ) -> List[Dict[str, Any]]:
        """Get contextually relevant quick actions based on portfolio state"""
        
        relevant_actions = []
        
        try:
            # Parse portfolio context
            total_value = self._parse_value(portfolio_context.get("totalValue", "0"))
            risk_level = portfolio_context.get("riskLevel", "MODERATE")
            holdings_count = len(portfolio_context.get("holdings", []))
            
            # Risk-based recommendations
            if risk_level == "HIGH":
                relevant_actions.extend([
                    self.actions["stress_test"],
                    self.actions["calculate_var"],
                    self.actions["emotional_check"]
                ])
            elif risk_level == "LOW":
                relevant_actions.extend([
                    self.actions["optimization"],
                    self.actions["trade_ideas"],
                    self.actions["strategic_review"]
                ])
            else:  # MODERATE
                relevant_actions.extend([
                    self.actions["rebalancing_check"],
                    self.actions["correlation_analysis"],
                    self.actions["performance_review"]
                ])
            
            # Portfolio size based recommendations
            if total_value > 1000000:  # $1M+
                relevant_actions.append(self.actions["asset_allocation"])
                relevant_actions.append(self.actions["sector_analysis"])
            
            if holdings_count > 10:
                relevant_actions.append(self.actions["correlation_analysis"])
            elif holdings_count < 5:
                relevant_actions.append(self.actions["optimization"])
            
            # Conversation history based recommendations
            if conversation_history:
                recent_content = " ".join([msg.get("content", "") for msg in conversation_history[-3:]])
                
                if any(word in recent_content.lower() for word in ["worried", "concerned", "nervous", "anxious"]):
                    relevant_actions.insert(0, self.actions["emotional_check"])
                    relevant_actions.insert(1, self.actions["bias_assessment"])
                
                if any(word in recent_content.lower() for word in ["sell", "selling", "market crash", "afraid"]):
                    relevant_actions.insert(0, self.actions["discipline_review"])
                    relevant_actions.insert(1, self.actions["stress_test"])
                
                if any(word in recent_content.lower() for word in ["buy", "buying", "opportunity", "invest more"]):
                    relevant_actions.insert(0, self.actions["decision_framework"])
                    relevant_actions.insert(1, self.actions["market_outlook"])
            
            # Default actions if none selected
            if not relevant_actions:
                relevant_actions = [
                    self.actions["performance_review"],
                    self.actions["rebalancing_check"],
                    self.actions["market_outlook"],
                    self.actions["bias_assessment"]
                ]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_actions = []
            for action in relevant_actions:
                if action.id not in seen:
                    seen.add(action.id)
                    unique_actions.append(action)
            
            # Limit to max_actions
            selected_actions = unique_actions[:max_actions]
            
            # Convert to API format
            return [self._action_to_dict(action) for action in selected_actions]
            
        except Exception as e:
            # Fallback to default actions
            return [
                self._action_to_dict(self.actions["performance_review"]),
                self._action_to_dict(self.actions["calculate_var"]),
                self._action_to_dict(self.actions["market_outlook"]),
                self._action_to_dict(self.actions["bias_assessment"])
            ]
    
    def get_all_actions_by_category(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all actions organized by category"""
        categories = {}
        
        for action in self.actions.values():
            category = action.category
            if category not in categories:
                categories[category] = []
            categories[category].append(self._action_to_dict(action))
        
        return categories
    
    def get_action_query(self, action_id: str, portfolio_context: Dict[str, Any]) -> str:
        """Generate appropriate query for a quick action"""
        action = self.actions.get(action_id)
        if not action:
            return "Please provide analysis."
        
        portfolio_id = portfolio_context.get("portfolio_id", "")
        
        query_templates = {
            "calculate_var": "Calculate the Value at Risk (VaR) for my portfolio at 95% and 99% confidence levels.",
            "stress_test": "Run a comprehensive stress test on my portfolio against major market scenarios.",
            "correlation_analysis": "Show me the correlation matrix between my holdings and identify concentration risks.",
            "volatility_forecast": "Provide a volatility forecast for my portfolio using GARCH modeling.",
            "rebalancing_check": "Check if my portfolio needs rebalancing and suggest optimal allocations.",
            "optimization": "Optimize my portfolio allocation for the best risk-adjusted returns.",
            "performance_review": "Provide a detailed performance review with attribution analysis.",
            "trade_ideas": "Generate AI-based trade ideas to improve my portfolio.",
            "strategic_review": "Conduct a high-level strategic review of my investment approach.",
            "market_outlook": "What's the current market outlook and how should it affect my strategy?",
            "asset_allocation": "Review my strategic asset allocation across different asset classes.",
            "sector_analysis": "Analyze my sector allocation and identify opportunities.",
            "bias_assessment": "Help me identify any behavioral biases in my investment decisions.",
            "emotional_check": "Let's do an emotional check-in about my current investment mindset.",
            "decision_framework": "Guide me through a structured framework for making this investment decision.",
            "discipline_review": "Review whether I'm sticking to my investment discipline and long-term plan."
        }
        
        return query_templates.get(action_id, action.description)
    
    def _action_to_dict(self, action: QuickAction) -> Dict[str, Any]:
        """Convert QuickAction to dictionary for API response"""
        return {
            "id": action.id,
            "label": action.label,
            "specialist": action.specialist,
            "category": action.category,
            "description": action.description,
            "priority": action.priority
        }
    
    def _parse_value(self, value_str: str) -> float:
        """Parse portfolio value string to float"""
        try:
            # Remove $ and commas
            clean_str = value_str.replace("$", "").replace(",", "")
            return float(clean_str)
        except:
            return 0.0