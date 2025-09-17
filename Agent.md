# Enhanced Investment Committee Agents

## Overview

The Enhanced Investment Committee system provides institutional-grade investment analysis through specialized AI agents that integrate with real backend financial tools. Each agent specializes in specific areas of investment management and uses advanced quantitative models for analysis.

## Architecture

```
Enhanced Investment Committee
├── Enhanced Manager (routing & orchestration)
├── Enhanced Quantitative Analyst (risk analysis)
├── Enhanced CIO Agent (strategic allocation)
├── Enhanced Portfolio Manager (optimization)
├── Enhanced Behavioral Coach (psychology)
└── Backend Tools Integration
    ├── Risk Tools (VaR, stress testing)
    ├── Regime Tools (HMM, volatility detection)
    ├── Strategy Tools (momentum, optimization)
    └── Behavioral Tools (bias detection)
```

## Current Agents

### 1. Enhanced Quantitative Analyst (`EnhancedChatQuantitativeAnalyst`)
**File:** `agents/enhanced_chat_quantitative_analyst.py`
**Specialization:** Risk analysis and quantitative modeling

**Capabilities:**
- Real VaR/CVaR calculation using portfolio returns
- Monte Carlo stress testing with fat-tail modeling
- Regime-conditional risk analysis
- Advanced correlation and factor analysis
- Time-varying risk assessment

**Backend Tools Used:**
- `tools.risk_tools.calculate_risk_metrics`
- `tools.risk_tools.calculate_regime_conditional_risk`
- `tools.risk_tools.advanced_monte_carlo_stress_test`

### 2. Enhanced CIO Agent (`EnhancedChatCIOAgent`)
**File:** `agents/enhanced_chat_cio_agent.py`
**Specialization:** Strategic asset allocation and market outlook

**Capabilities:**
- HMM regime detection and analysis
- Strategic asset allocation optimization
- Market outlook with volatility regime assessment
- Long-term strategic planning
- Regime-aware target allocation

**Backend Tools Used:**
- `tools.regime_tools.detect_hmm_regimes`
- `tools.regime_tools.detect_volatility_regimes`

### 3. Enhanced Portfolio Manager (`EnhancedChatPortfolioManager`)
**File:** `agents/enhanced_chat_portfolio_manager.py`
**Specialization:** Tactical portfolio management and optimization

**Capabilities:**
- Dynamic risk budgeting and optimization
- Real trade generation from strategy analysis
- Equal Risk Contribution portfolio construction
- Advanced rebalancing with transaction cost analysis
- Systematic trade execution planning

**Backend Tools Used:**
- `tools.risk_tools.calculate_dynamic_risk_budgets`
- `tools.strategy_tools.design_risk_adjusted_momentum_strategy`

### 4. Enhanced Behavioral Coach (`EnhancedChatBehavioralCoach`)
**File:** `agents/enhanced_chat_behavioral_coach.py`
**Specialization:** Investment psychology and bias detection

**Capabilities:**
- Chat history bias analysis using NLP
- Market sentiment detection algorithms
- Behavioral risk impact quantification
- Evidence-based coaching strategies
- Emotional state assessment

**Backend Tools Used:**
- `tools.behavioral_tools.analyze_chat_for_biases`
- `tools.behavioral_tools.detect_market_sentiment`

## System Components

### Enhanced Manager (`EnhancedInvestmentCommitteeManager`)
**File:** `agents/enhanced_investment_committee_manager.py`

Routes queries to appropriate specialists and provides:
- Intelligent query routing with confidence scoring
- Cross-specialist recommendations
- Performance tracking and analytics
- Enhanced conversation memory
- Backend tool performance monitoring

### Data Helpers (`AgentDataHelper`)
**File:** `agents/data_helpers.py`

Provides data conversion utilities:
- Portfolio data extraction and formatting
- Backend tool integration wrappers
- Risk analysis integration
- Convenience functions for common operations

### Base Agent (`BaseAgent`)
**File:** `agents/base_agent.py`

Common functionality for all agents:
- Error response formatting
- Logging setup
- Base analyze_query method
- Capability tracking

## Adding New Agents

### Step 1: Create Agent File

Create a new file in the `agents/` directory following the naming convention:
```
agents/enhanced_chat_[agent_name].py
```

### Step 2: Agent Implementation

```python
# agents/enhanced_chat_new_agent.py
from agents.base_agent import BaseAgent
from agents.data_helpers import AgentDataHelper
from typing import Dict, List, Any, Optional

class EnhancedChatNewAgent(BaseAgent):
    """Enhanced New Agent with Backend Integration"""
    
    def __init__(self):
        super().__init__("New Agent")
        self.expertise_areas = ["area1", "area2", "area3"]
    
    async def analyze_query(self, query: str, portfolio_data: Dict, context: Dict) -> Dict:
        """Main entry point for analysis"""
        query_lower = query.lower()
        
        # Route to appropriate analysis methods
        if any(word in query_lower for word in ["keyword1", "keyword2"]):
            return await self._enhanced_analysis_method_1(portfolio_data, context)
        else:
            return await self._enhanced_general_analysis(portfolio_data, context)
    
    async def _enhanced_analysis_method_1(self, portfolio_data: Dict, context: Dict) -> Dict:
        """Specific analysis method using backend tools"""
        try:
            # Use backend tools for analysis
            from tools.new_tool_module import new_analysis_function
            
            # Extract portfolio data
            portfolio_returns = AgentDataHelper.extract_portfolio_returns(portfolio_data)
            
            # Call backend analysis
            analysis_results = new_analysis_function(portfolio_returns)
            
            # Process results and format response
            return {
                "specialist": "new_agent",
                "analysis_type": "analysis_method_1",
                "content": f"**New Agent Analysis**\n\nResults: {analysis_results}",
                "analysis": {
                    "riskScore": 50,
                    "recommendation": "Recommendation based on analysis",
                    "confidence": 90,
                    "specialist": "New Agent"
                },
                "data": {
                    "key_metric": analysis_results.get('metric', 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return self._error_response("Analysis failed")
```

### Step 3: Add to Manager

Update `agents/enhanced_investment_committee_manager.py`:

```python
# Add import at top
try:
    from agents.enhanced_chat_new_agent import EnhancedChatNewAgent
except ImportError as e:
    logging.error(f"Failed to import EnhancedChatNewAgent: {e}")
    EnhancedChatNewAgent = None

# Add to __init__ method
if EnhancedChatNewAgent:
    try:
        self.specialists["new_agent"] = EnhancedChatNewAgent()
        logger.info("✅ Enhanced New Agent initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Enhanced New Agent: {e}")

# Add routing patterns
self.routing_patterns["new_agent"] = {
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "phrases": ["specific phrase", "another phrase"]
}
```

### Step 4: Backend Tool Integration

If creating new backend tools:

1. **Create tool module:**
```python
# tools/new_tool_module.py
import numpy as np
import pandas as pd

def new_analysis_function(data):
    """New analysis function using institutional models"""
    # Implement analysis logic
    results = {
        'metric': calculate_metric(data),
        'confidence': 0.95,
        'analysis_type': 'new_analysis'
    }
    return results
```

2. **Add to data helpers:**
```python
# In agents/data_helpers.py
def get_new_analysis(portfolio_data: Dict) -> Dict:
    """Get new analysis for agents"""
    try:
        from tools.new_tool_module import new_analysis_function
        portfolio_returns = AgentDataHelper.extract_portfolio_returns(portfolio_data)
        return new_analysis_function(portfolio_returns)
    except Exception as e:
        logger.error(f"New analysis failed: {e}")
        return {'error': str(e)}
```

### Step 5: Update Specialist Info

Add to the specialist definitions in the manager:

```python
"new_agent": {
    "id": "new_agent",
    "name": "New Agent",
    "description": "Description of new agent capabilities",
    "capabilities": [
        "Capability 1",
        "Capability 2"
    ],
    "backend_tools": ["new_analysis_function", "other_tool"]
}
```

### Step 6: Testing

Run integration tests to verify the new agent:

```bash
python test_practical_integration.py
```

## Backend Tool Development

### Creating New Backend Tools

1. **Tool Module Structure:**
```python
# tools/new_tool.py
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

def institutional_analysis_function(data: pd.Series, **kwargs) -> Dict[str, Any]:
    """
    Institutional-grade analysis function
    
    Parameters:
    -----------
    data : pd.Series
        Input data for analysis
    
    Returns:
    --------
    Dict : Analysis results with metrics and metadata
    """
    try:
        # Implementation using advanced models
        results = {
            'primary_metric': calculate_primary_metric(data),
            'secondary_metrics': calculate_secondary_metrics(data),
            'confidence': calculate_confidence(data),
            'model_used': 'institutional_model_v1',
            'analysis_timestamp': pd.Timestamp.now()
        }
        return results
    except Exception as e:
        return {'error': str(e)}
```

2. **Integration Patterns:**
- Use consistent return formats with error handling
- Include confidence scores and metadata
- Support both DataFrame and Series inputs
- Provide fallback behavior for edge cases

### Tool Categories

**Risk Tools (`tools/risk_tools.py`):**
- VaR/CVaR calculation
- Stress testing
- Correlation analysis
- Time-varying risk models

**Regime Tools (`tools/regime_tools.py`):**
- HMM regime detection
- Volatility regime analysis
- Regime transition models

**Strategy Tools (`tools/strategy_tools.py`):**
- Momentum strategies
- Mean reversion strategies
- Risk-adjusted optimization

**Behavioral Tools (`tools/behavioral_tools.py`):**
- NLP bias detection
- Sentiment analysis
- Behavioral pattern recognition

## Configuration

### Environment Variables
```bash
# Optional configuration
AGENT_LOG_LEVEL=INFO
BACKEND_TOOLS_TIMEOUT=30
MAX_PORTFOLIO_SIZE=1000
```

### Data Helpers Configuration
The system automatically adapts to available backend tools and provides fallback behavior when tools are unavailable.

## API Usage

### Basic Query Routing
```python
from agents.enhanced_investment_committee_manager import EnhancedInvestmentCommitteeManager

manager = EnhancedInvestmentCommitteeManager()

portfolio_context = {
    "total_value": 150000,
    "holdings": [
        {"ticker": "AAPL", "value": 45000, "symbol": "AAPL"},
        {"ticker": "MSFT", "value": 37500, "symbol": "MSFT"}
    ],
    "daily_change": "+1.2%"
}

response = await manager.route_query(
    query="What's my portfolio risk?",
    portfolio_context=portfolio_context,
    preferred_specialist="quantitative_analyst"
)
```

### Response Format
```python
{
    "specialist_used": "quantitative_analyst",
    "analysis_type": "var_analysis", 
    "content": "Formatted analysis content",
    "analysis": {
        "riskScore": 65,
        "recommendation": "Optimize risk-return profile",
        "confidence": 92,
        "specialist": "Quantitative Analyst"
    },
    "data": {
        "var_95_pct": 2.5,
        "annual_volatility": 18.5
    },
    "backend_integration": true,
    "execution_time": 0.234
}
```

## Troubleshooting

### Common Issues

1. **Import Errors:**
   - Ensure all files use absolute imports (`agents.module` not `.module`)
   - Check that base_agent.py exists
   - Verify all required dependencies are installed

2. **Backend Tool Failures:**
   - Check that backend tools are properly installed
   - Verify tool functions return expected format
   - Use fallback behavior for graceful degradation

3. **Agent Not Found:**
   - Verify agent is added to manager's specialists dictionary
   - Check routing patterns are configured
   - Ensure class name matches expected format

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Monitoring

The system tracks:
- Backend tool execution times
- Agent response confidence levels
- Success/failure rates
- Cross-specialist suggestion patterns

Access performance data:
```python
performance = manager.get_backend_performance_summary()
```

## Security Considerations

- All backend tools should validate input data
- Portfolio data should be sanitized before processing
- Agent responses should not expose sensitive system details
- Use appropriate error handling to prevent information leakage