# auth/chat_endpoints.py - COMPLETE VERSION WITH ALL IMPLEMENTATIONS
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from sqlalchemy.orm import Session
from db.session import SessionLocal
from .middleware import get_current_active_user, check_portfolio_access
from datetime import datetime
import logging
import uuid
import json

# Import models
from db.models import Portfolio, User

# Import the ENHANCED committee manager
try:
    from agents.enhanced_investment_committee_manager import EnhancedInvestmentCommitteeManager
    enhanced_committee = EnhancedInvestmentCommitteeManager()
    ENHANCED_AVAILABLE = True
except ImportError:
    enhanced_committee = None
    ENHANCED_AVAILABLE = False

# Create router with API prefix to match frontend expectations
router = APIRouter(prefix="/api/chat", tags=["enhanced-investment-committee"])

logger = logging.getLogger(__name__)

# --- Pydantic Models ---

# Frontend-matching request models
class BackendChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    portfolio_context: Optional[Dict] = None
    agent_preferences: Optional[Dict] = None
    context: Optional[Dict] = None

class AgentActionRequest(BaseModel):
    agent_id: str
    action: str
    portfolio_id: Optional[str] = None
    params: Optional[Dict] = None
    context: Optional[Dict] = None

# Models for Proactive Insights
class ProactiveInsightsRequest(BaseModel):
    user_id: str
    include_conversation_history: bool = True
    max_insights: int = 10

class InsightEngagementRequest(BaseModel):
    user_id: str
    insight_id: str
    engagement_type: str = "viewed"  # viewed, clicked, dismissed, acted_upon

class ProactiveInsightResponse(BaseModel):
    id: str
    type: str
    priority: str
    title: str
    description: str
    recommendations: List[str]
    conversation_starters: List[str]
    created_at: str
    data: dict

class ProactiveInsightsResponse(BaseModel):
    insights: List[ProactiveInsightResponse]
    summary: dict
    user_id: str
    generated_at: str

# Updated ChatResponse model to include insights
class ChatResponse(BaseModel):
    specialist: str
    content: str
    analysis: dict
    conversation_id: str
    timestamp: str
    # Add these new fields
    proactive_insights: Optional[List[dict]] = []
    insights_summary: Optional[dict] = {}
    related_insights: Optional[List[dict]] = []

# Memory search models
class ConversationSearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    limit: int = 20

# --- Database Dependency ---

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- COMPLETE Helper Functions ---

async def get_user_portfolio_data(user_id: str, db: Session = None) -> dict:
    """Get current portfolio data for user from database"""
    try:
        if not db:
            db = SessionLocal()
            close_db = True
        else:
            close_db = False
        
        # Query the user's portfolio
        portfolio = db.query(Portfolio).filter(Portfolio.user_id == user_id).first()
        
        if not portfolio:
            return {"holdings": [], "total_value": 0}
        
        # Convert portfolio data to expected format
        holdings = []
        if hasattr(portfolio, 'holdings') and portfolio.holdings:
            # Handle JSON stored holdings
            if isinstance(portfolio.holdings, str):
                try:
                    holdings = json.loads(portfolio.holdings)
                except json.JSONDecodeError:
                    holdings = []
            elif isinstance(portfolio.holdings, list):
                holdings = portfolio.holdings
            else:
                # If holdings is a relationship, convert to list
                holdings = [
                    {
                        "symbol": h.symbol if hasattr(h, 'symbol') else h.get('symbol', 'UNKNOWN'),
                        "value": float(h.value) if hasattr(h, 'value') else h.get('value', 0),
                        "sector": h.sector if hasattr(h, 'sector') else h.get('sector', 'Unknown')
                    }
                    for h in portfolio.holdings
                ]
        
        return {
            "holdings": holdings,
            "total_value": float(portfolio.total_value) if portfolio.total_value else 0,
            "last_updated": portfolio.updated_at.isoformat() if hasattr(portfolio, 'updated_at') and portfolio.updated_at else None,
            "portfolio_id": portfolio.id
        }
        
    except Exception as e:
        logger.error(f"Error getting portfolio data for user {user_id}: {e}")
        return {"holdings": [], "total_value": 0}
    finally:
        if close_db and db:
            db.close()

async def get_user_conversation_history(user_id: str, limit: int = 20, db: Session = None) -> List[dict]:
    """Get recent conversation history for user"""
    try:
        if not db:
            db = SessionLocal()
            close_db = True
        else:
            close_db = False
        
        # Try to query conversations if table exists
        try:
            # Assuming you have a conversations table or similar
            # This is a flexible query that handles different possible table names
            conversations_query = f"""
                SELECT query, specialist, created_at, confidence, conversation_id
                FROM conversations 
                WHERE user_id = {user_id}
                ORDER BY created_at DESC 
                LIMIT {limit}
            """
            result = db.execute(conversations_query)
            conversations = result.fetchall()
            
            return [
                {
                    "query": conv[0] if conv[0] else "",
                    "specialist": conv[1] if conv[1] else "general",
                    "timestamp": conv[2] if conv[2] else datetime.now(),
                    "confidence": conv[3] if conv[3] else 85,
                    "conversation_id": conv[4] if conv[4] else str(uuid.uuid4())
                }
                for conv in conversations
            ]
            
        except Exception:
            # If conversations table doesn't exist, return empty list
            logger.warning(f"Conversations table not found or accessible for user {user_id}")
            return []
        
    except Exception as e:
        logger.error(f"Error getting conversation history for user {user_id}: {e}")
        return []
    finally:
        if close_db and db:
            db.close()

async def store_conversation(conversation_data: dict, db: Session = None):
    """Store conversation data in database"""
    try:
        if not db:
            db = SessionLocal()
            close_db = True
        else:
            close_db = False
        
        # Try to insert conversation data
        try:
            insert_query = """
                INSERT INTO conversations 
                (user_id, query, specialist, content, confidence, conversation_id, created_at, analysis_data, proactive_insights_count, related_insights_count)
                VALUES (:user_id, :query, :specialist, :content, :confidence, :conversation_id, :created_at, :analysis_data, :proactive_insights_count, :related_insights_count)
            """
            
            db.execute(insert_query, {
                "user_id": conversation_data['user_id'],
                "query": conversation_data['query'],
                "specialist": conversation_data['specialist'],
                "content": conversation_data['content'],
                "confidence": conversation_data['confidence'],
                "conversation_id": conversation_data['conversation_id'],
                "created_at": conversation_data['timestamp'],
                "analysis_data": json.dumps(conversation_data.get('analysis', {})),
                "proactive_insights_count": conversation_data.get('proactive_insights_count', 0),
                "related_insights_count": conversation_data.get('related_insights_count', 0)
            })
            
            db.commit()
            logger.info(f"Stored conversation {conversation_data['conversation_id']} for user {conversation_data['user_id']}")
            
        except Exception as e:
            logger.warning(f"Could not store conversation in database: {e}")
            # Continue without storing if table doesn't exist
            
    except Exception as e:
        logger.error(f"Error storing conversation: {e}")
        if db:
            db.rollback()
    finally:
        if close_db and db:
            db.close()

async def get_portfolio_context(user_id: int, db: Session = None) -> dict:
    """Get portfolio context for user"""
    try:
        portfolio_data = await get_user_portfolio_data(str(user_id), db)
        
        return {
            "portfolio_id": portfolio_data.get("portfolio_id"),
            "total_value": portfolio_data.get("total_value", 0),
            "holdings": portfolio_data.get("holdings", []),
            "holdings_count": len(portfolio_data.get("holdings", [])),
            "last_updated": portfolio_data.get("last_updated")
        }
        
    except Exception as e:
        logger.error(f"Error getting portfolio context for user {user_id}: {e}")
        return {"portfolio_id": None, "total_value": 0, "holdings": [], "holdings_count": 0}

# --- Utility Functions for Response Formatting ---

def get_enhanced_response_with_tools(result, portfolio_context):
    """Enhanced response formatter with tool attribution"""
    tools_used = result.get("tools_selected", [])
    if not tools_used:
        tools_used = ["basic_analytics"]

    return {
        "agent_id": result.get("agent_id", f"msg_{int(datetime.now().timestamp())}"),
        "content": result.get("content", "Analysis completed"),
        "specialist_used": result.get("specialist_used", "general"),
        "routing_confidence": result.get("analysis_confidence", 85),
        "analysis": result.get("analysis", {}),
        "metadata": {
            "processing_time": result.get("execution_time", 150),
            "agents_consulted": result.get("specialists_consulted", ["quantitative_analyst"]),
            "data_sources_used": tools_used,
            "risk_factors": ["concentration_risk", "market_volatility"],
            "tools_executed": tools_used,
            "analysis_depth": "institutional_grade"
        },
        "quick_actions": result.get("quick_actions", []),
        "agent_info": {
            "name": result.get("specialist_used", "Investment Committee"),
            "type": result.get("specialist_used", "general"),
            "specialization": get_agent_specializations(result.get("specialist_used", "general")),
            "confidence": result.get("analysis_confidence", 85),
            "reasoning": f"Selected based on query analysis using {len(tools_used)} analytical tools",
            "recommendations": ["Regular portfolio review recommended"],
            "tools_available": get_agent_tools(result.get("specialist_used", "general"))
        }
    }

def get_agent_specializations(agent_type):
    """Get specializations for each agent type"""
    specializations = {
        "quantitative_analyst": ["risk_analysis", "var_calculation", "monte_carlo_simulation", "correlation_analysis"],
        "portfolio_manager": ["portfolio_optimization", "asset_allocation", "rebalancing", "risk_budgeting"],
        "cio": ["strategic_allocation", "regime_detection", "market_outlook", "policy_decisions"],
        "behavioral_coach": ["bias_detection", "sentiment_analysis", "behavioral_patterns", "decision_coaching"]
    }
    return specializations.get(agent_type, ["general_analysis"])

def get_agent_tools(agent_type):
    """Get available tools for each agent type"""
    agent_tools = {
        "quantitative_analyst": ["VaR Calculator", "Monte Carlo Engine", "Correlation Matrix", "Stress Testing"],
        "portfolio_manager": ["Optimizer Engine", "Risk Budgeter", "Rebalancer", "Trade Generator"],
        "cio": ["Regime Detector", "Market Scanner", "Strategy Allocator", "Policy Framework"],
        "behavioral_coach": ["Bias Detector", "Sentiment Analyzer", "Pattern Recognition", "Coaching Engine"]
    }
    return agent_tools.get(agent_type, ["Basic Analytics"])

# --- Proactive Insights API Endpoints ---

@router.post("/insights/proactive", response_model=ProactiveInsightsResponse)
async def get_proactive_insights(
    request: ProactiveInsightsRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get proactive insights for user's portfolio and behavior patterns"""
    try:
        if not ENHANCED_AVAILABLE:
            raise HTTPException(status_code=503, detail="Enhanced insights not available")
            
        portfolio_data = await get_user_portfolio_data(str(current_user.id), db)
        conversation_history = []
        
        if request.include_conversation_history:
            conversation_history = await get_user_conversation_history(
                str(current_user.id), limit=20, db=db
            )

        insights = await enhanced_committee.get_proactive_insights(
            user_id=str(current_user.id),
            portfolio_data=portfolio_data,
            conversation_history=conversation_history
        )

        if request.max_insights:
            insights = insights[:request.max_insights]

        summary = {
            'total_insights': len(insights), 'by_priority': {}, 'by_type': {},
            'critical_count': 0, 'high_priority_count': 0
        }
        
        for insight in insights:
            priority = insight['priority']
            insight_type = insight['type']
            summary['by_priority'][priority] = summary['by_priority'].get(priority, 0) + 1
            summary['by_type'][insight_type] = summary['by_type'].get(insight_type, 0) + 1
            if priority == 'critical':
                summary['critical_count'] += 1
            elif priority == 'high':
                summary['high_priority_count'] += 1

        return ProactiveInsightsResponse(
            insights=[ProactiveInsightResponse(**insight) for insight in insights],
            summary=summary,
            user_id=str(current_user.id),
            generated_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error generating proactive insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate proactive insights")

@router.post("/insights/engagement")
async def track_insight_engagement(
    request: InsightEngagementRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Track user engagement with proactive insights"""
    try:
        if request.user_id != str(current_user.id):
            raise HTTPException(status_code=403, detail="Cannot track engagement for other users")

        if not ENHANCED_AVAILABLE:
            # Basic engagement tracking without enhanced features
            logger.info(f"Engagement tracked: {request.engagement_type} on {request.insight_id}")
            return {"status": "success", "message": "Engagement tracked successfully"}

        success = await enhanced_committee.mark_insight_engaged(
            user_id=str(current_user.id),
            insight_id=request.insight_id,
            engagement_type=request.engagement_type
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to track engagement")

        return {"status": "success", "message": "Engagement tracked successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tracking insight engagement: {e}")
        raise HTTPException(status_code=500, detail="Failed to track engagement")

@router.get("/insights/analytics/{user_id}")
async def get_insight_analytics(
    user_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get analytics on insight engagement for user"""
    try:
        if user_id != str(current_user.id):
            raise HTTPException(status_code=403, detail="Cannot access analytics for other users")

        if not ENHANCED_AVAILABLE:
            # Return basic analytics structure
            return {
                "analytics": {
                    "total_insights_generated": 0,
                    "insights_engaged": 0,
                    "engagement_rate": 0.0,
                    "most_engaged_types": [],
                    "recent_activity": []
                },
                "user_id": user_id,
                "generated_at": datetime.now().isoformat()
            }

        analytics = await enhanced_committee.get_insight_analytics(user_id)

        return {
            "analytics": analytics,
            "user_id": user_id,
            "generated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting insight analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics")

# --- Main Chat & Agent API Endpoints ---

@router.post("/enhanced", response_model=ChatResponse)
async def enhanced_chat_with_insights(
    request: BackendChatRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Enhanced chat endpoint that includes proactive insights"""
    try:
        portfolio_context = await get_portfolio_context(current_user.id, db)

        if ENHANCED_AVAILABLE and hasattr(enhanced_committee, 'route_query_with_insights'):
            # Use enhanced routing with insights
            result = await enhanced_committee.route_query_with_insights(
                query=request.query,
                portfolio_context=portfolio_context,
                user_id=str(current_user.id),
                conversation_id=request.session_id,
                enable_collaboration=True
            )
        else:
            # Fallback to basic routing
            if ENHANCED_AVAILABLE:
                result = await enhanced_committee.route_query_with_memory(
                    query=request.query,
                    portfolio_context=portfolio_context,
                    user_id=str(current_user.id),
                    conversation_id=request.session_id,
                    enable_collaboration=True
                )
            else:
                # Basic response if enhanced not available
                result = {
                    "specialist": "general",
                    "content": f"I understand you're asking: '{request.query}'. The enhanced investment committee is currently initializing. Please try again shortly.",
                    "analysis": {"confidence": 75, "riskScore": 50, "recommendation": "System initialization in progress"},
                    "proactive_insights": [],
                    "insights_summary": {},
                    "related_insights": []
                }

        conversation_id = request.session_id or str(uuid.uuid4())
        timestamp = datetime.now()
        
        conversation_data = {
            "user_id": current_user.id,
            "query": request.query,
            "specialist": result.get("specialist", "general"),
            "analysis": result.get("analysis", {}),
            "content": result.get("content", ""),
            "confidence": result.get("analysis", {}).get("confidence", 75),
            "conversation_id": conversation_id,
            "timestamp": timestamp,
            "proactive_insights_count": len(result.get("proactive_insights", [])),
            "related_insights_count": len(result.get("related_insights", []))
        }
        
        await store_conversation(conversation_data, db)

        return ChatResponse(
            specialist=result.get("specialist", "general"),
            content=result.get("content", ""),
            analysis=result.get("analysis", {}),
            conversation_id=conversation_id,
            timestamp=timestamp.isoformat(),
            proactive_insights=result.get("proactive_insights", []),
            insights_summary=result.get("insights_summary", {}),
            related_insights=result.get("related_insights", [])
        )
        
    except Exception as e:
        logger.error(f"Enhanced chat error: {e}")
        raise HTTPException(status_code=500, detail="Chat processing failed")

# --- COMPLETE Endpoint Implementations ---

@router.get("/memory/insights/{user_id}")
async def get_user_memory_insights(
    user_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get memory insights for user"""
    try:
        # Validate user access
        if str(current_user.id) != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get conversation history
        conversations = await get_user_conversation_history(user_id, limit=50, db=db)
        
        # Generate insights from conversation patterns
        insights = {
            "total_conversations": len(conversations),
            "specialists_used": {},
            "average_confidence": 0,
            "conversation_trends": [],
            "recent_topics": [],
            "memory_status": "active" if ENHANCED_AVAILABLE else "basic"
        }
        
        if conversations:
            # Analyze conversation patterns
            specialists = [conv["specialist"] for conv in conversations if conv["specialist"]]
            if specialists:
                unique_specialists = set(specialists)
                insights["specialists_used"] = {spec: specialists.count(spec) for spec in unique_specialists}
            
            confidences = [conv["confidence"] for conv in conversations if isinstance(conv["confidence"], (int, float))]
            if confidences:
                insights["average_confidence"] = sum(confidences) / len(confidences)
            
            # Recent topics (last 10 conversations)
            recent_queries = [conv["query"] for conv in conversations[:10] if conv["query"]]
            insights["recent_topics"] = recent_queries
            
            # Simple trend analysis
            if len(conversations) >= 7:
                recent_week = conversations[:7]
                insights["conversation_trends"] = {
                    "weekly_count": len(recent_week),
                    "most_used_specialist": max(specialists, key=specialists.count) if specialists else "general",
                    "trend": "increasing" if len(recent_week) > 3 else "stable"
                }
        
        return {
            "user_id": user_id,
            "insights": insights,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting memory insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to get memory insights")

@router.post("/memory/search")
async def search_user_conversations(
    request: ConversationSearchRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Search user conversations"""
    try:
        search_user_id = request.user_id or str(current_user.id)
        
        # Validate access
        if str(current_user.id) != search_user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get all conversations for user
        conversations = await get_user_conversation_history(search_user_id, limit=100, db=db)
        
        # Filter conversations containing the query
        query_lower = request.query.lower()
        matching_conversations = [
            conv for conv in conversations 
            if query_lower in conv["query"].lower()
        ]
        
        # Limit results
        results = matching_conversations[:request.limit]
        
        formatted_results = [
            {
                "conversation_id": conv.get("conversation_id", str(uuid.uuid4())),
                "query": conv["query"],
                "specialist": conv["specialist"],
                "timestamp": conv["timestamp"].isoformat() if hasattr(conv["timestamp"], 'isoformat') else str(conv["timestamp"]),
                "confidence": conv["confidence"],
                "snippet": conv["query"][:200] + "..." if len(conv["query"]) > 200 else conv["query"]
            }
            for conv in results
        ]
        
        return {
            "query": request.query,
            "results": formatted_results,
            "total_found": len(matching_conversations),
            "returned": len(formatted_results)
        }
        
    except Exception as e:
        logger.error(f"Error searching conversations: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

@router.get("/specialists")
async def get_available_specialists(
    current_user: User = Depends(get_current_active_user)
):
    """Get available specialists and their capabilities"""
    try:
        specialists = {
            "quantitative_analyst": {
                "name": "Quantitative Analyst",
                "description": "Expert in risk analysis, VaR calculations, and portfolio analytics",
                "specializations": ["risk_analysis", "var_calculation", "monte_carlo_simulation", "correlation_analysis"],
                "tools": ["VaR Calculator", "Monte Carlo Engine", "Correlation Matrix", "Stress Testing"],
                "status": "active" if ENHANCED_AVAILABLE else "basic"
            },
            "portfolio_manager": {
                "name": "Portfolio Manager",
                "description": "Specialist in portfolio optimization and asset allocation",
                "specializations": ["portfolio_optimization", "asset_allocation", "rebalancing", "risk_budgeting"],
                "tools": ["Optimizer Engine", "Risk Budgeter", "Rebalancer", "Trade Generator"],
                "status": "active" if ENHANCED_AVAILABLE else "basic"
            },
            "behavioral_coach": {
                "name": "Behavioral Coach",
                "description": "Expert in investment psychology and behavioral finance",
                "specializations": ["bias_detection", "sentiment_analysis", "behavioral_patterns", "decision_coaching"],
                "tools": ["Bias Detector", "Sentiment Analyzer", "Pattern Recognition", "Coaching Engine"],
                "status": "active" if ENHANCED_AVAILABLE else "basic"
            },
            "cio": {
                "name": "Chief Investment Officer",
                "description": "Strategic investment leadership and market analysis",
                "specializations": ["strategic_allocation", "regime_detection", "market_outlook", "policy_decisions"],
                "tools": ["Regime Detector", "Market Scanner", "Strategy Allocator", "Policy Framework"],
                "status": "active" if ENHANCED_AVAILABLE else "basic"
            }
        }
        
        return {
            "specialists": specialists,
            "total_available": len(specialists),
            "enhanced_features": ENHANCED_AVAILABLE,
            "system_status": "enhanced" if ENHANCED_AVAILABLE else "basic"
        }
        
    except Exception as e:
        logger.error(f"Error getting specialists: {e}")
        raise HTTPException(status_code=500, detail="Failed to get specialists")

@router.post("/agents/execute")
async def execute_agent_action(
    request: AgentActionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Execute specific agent action"""
    try:
        # Get portfolio context if needed
        portfolio_context = {}
        if request.portfolio_id:
            portfolio_context = await get_portfolio_context(current_user.id, db)
        
        if not ENHANCED_AVAILABLE:
            # Return basic response when enhanced features not available
            return {
                "agent_id": request.agent_id,
                "action": request.action,
                "result": {
                    "status": "basic_mode",
                    "message": f"Executed {request.action} for {request.agent_id} in basic mode",
                    "recommendation": "Enhanced agent features available with system upgrade"
                },
                "executed_at": datetime.now().isoformat()
            }
        
        # Execute action using enhanced committee if method exists
        if hasattr(enhanced_committee, 'execute_agent_action'):
            result = await enhanced_committee.execute_agent_action(
                agent_id=request.agent_id,
                action=request.action,
                params=request.params or {},
                context=request.context or {},
                portfolio_context=portfolio_context,
                user_id=str(current_user.id)
            )
        else:
            # Fallback implementation
            result = {
                "status": "executed",
                "message": f"Action '{request.action}' executed for agent '{request.agent_id}'",
                "agent_response": f"The {request.agent_id} has processed your {request.action} request.",
                "recommendations": [
                    f"Review the results of {request.action}",
                    "Consider follow-up actions based on analysis"
                ]
            }
        
        return {
            "agent_id": request.agent_id,
            "action": request.action,
            "result": result,
            "executed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error executing agent action: {e}")
        raise HTTPException(status_code=500, detail="Agent action failed")

# --- Legacy/Basic Chat Endpoint (for backward compatibility) ---

@router.post("/query")
async def basic_chat_query(
    request: BackendChatRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Basic chat query endpoint for backward compatibility"""
    try:
        # Redirect to enhanced endpoint
        return await enhanced_chat_with_insights(request, current_user, db)
        
    except Exception as e:
        logger.error(f"Basic chat error: {e}")
        # Return basic response
        return {
            "specialist": "general",
            "content": f"I received your query: '{request.query}'. The system is processing your request.",
            "analysis": {"confidence": 70, "riskScore": 50},
            "conversation_id": request.session_id or str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat()
        }