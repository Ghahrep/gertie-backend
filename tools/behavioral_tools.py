# tools/behavioral_tools.py - Clean Behavioral Analysis Tools
"""
Behavioral Finance Analysis Tools - Clean Architecture
=====================================================

Tools for analyzing user behavior patterns and generating AI-powered summaries.
Stripped of agent dependencies for direct function calls.
"""

import json
from typing import Dict, Any, List, Optional
import anthropic
from datetime import datetime

def analyze_chat_for_biases(chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Analyzes a user's chat history to detect potential behavioral biases.
    Looks for patterns like frequent switching, chasing performance, or fear-based language.
    
    Args:
        chat_history: List of chat messages with 'role' and 'content' keys
        
    Returns:
        Dict with bias analysis results
    """
    try:
        biases_found = {}
        user_messages = [msg['content'].lower() for msg in chat_history if msg.get('role') == 'user']
        
        if not user_messages:
            return {
                "success": True, 
                "summary": "No user messages found to analyze.",
                "biases_detected": {},
                "analysis_timestamp": datetime.now().isoformat()
            }
        
        # 1. Loss Aversion / Panic Selling Detection
        loss_aversion_keywords = ['panic', 'sell everything', 'market crash', 'get out', 'afraid of losing']
        if any(any(kw in msg for kw in loss_aversion_keywords) for msg in user_messages):
            biases_found['Loss Aversion'] = {
                "finding": "Detected language related to panic or selling based on fear.",
                "suggestion": "Consider sticking to your long-term plan. Emotional decisions during downturns can often hurt performance.",
                "severity": "High"
            }

        # 2. Herding / FOMO Detection
        herding_keywords = ['everyone is buying', 'hot stock', 'get in on', 'don\'t want to miss out', 'fomo']
        if any(any(kw in msg for kw in herding_keywords) for msg in user_messages):
            biases_found['Herding Behavior (FOMO)'] = {
                "finding": "Detected language suggesting a desire to follow the crowd or chase popular trends.",
                "suggestion": "Ensure investment decisions are based on your own research and strategy, not just popularity.",
                "severity": "Medium"
            }

        # 3. Overconfidence / Frequent Rebalancing Detection
        rebalance_queries = [msg for msg in user_messages if 'rebalance' in msg]
        if len(rebalance_queries) > 2:
            biases_found['Over-trading / Overconfidence'] = {
                "finding": "Noticed multiple requests for rebalancing in a short period.",
                "suggestion": "Frequent trading can increase costs and may not always lead to better results. Ensure each change aligns with your long-term goals.",
                "severity": "Medium"
            }
        
        # 4. Anchoring Bias Detection
        anchoring_keywords = ['bought at', 'paid', 'was worth', 'used to be']
        if any(any(kw in msg for kw in anchoring_keywords) for msg in user_messages):
            biases_found['Anchoring Bias'] = {
                "finding": "References to past prices or purchase prices detected.",
                "suggestion": "Focus on future prospects rather than past purchase prices when making investment decisions.",
                "severity": "Low"
            }
            
        if not biases_found:
            return {
                "success": True, 
                "summary": "No strong behavioral biases were detected in the recent conversation.",
                "biases_detected": {},
                "message_count": len(user_messages),
                "analysis_timestamp": datetime.now().isoformat()
            }

        return {
            "success": True, 
            "biases_detected": biases_found,
            "message_count": len(user_messages),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error analyzing chat history: {str(e)}",
            "biases_detected": {},
            "analysis_timestamp": datetime.now().isoformat()
        }


def summarize_analysis_results(
    analysis_type: str, 
    data: Dict[str, Any], 
    anthropic_api_key: Optional[str] = None
) -> str:
    """
    Uses Claude 3.5 Sonnet to generate a human-readable, narrative summary
    of a structured JSON analysis result.
    
    Args:
        analysis_type: Type of analysis (e.g., "risk", "portfolio_optimization")
        data: Dictionary containing analysis results
        anthropic_api_key: API key for Anthropic (optional, can be set via env var)
        
    Returns:
        Human-readable summary string
    """
    if not anthropic_api_key:
        import os
        anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
    if not anthropic_api_key:
        return "Error: ANTHROPIC_API_KEY not found. Please set as environment variable or pass as parameter."
    
    try:
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        data_json_string = json.dumps(data, indent=2, default=str)
        
        system_prompt = (
            "You are an expert financial analyst. Your task is to write a concise, professional, and "
            "insightful summary of a financial analysis report. Do not just list the numbers; "
            "provide a narrative interpretation that highlights key insights and actionable takeaways."
        )
        
        user_prompt = f"""
        Here is the financial analysis report for you to summarize:

        Analysis Type: {analysis_type}

        JSON Data:
        {data_json_string}

        Based on the data above, write a fluid, one-paragraph summary for the user. 
        Focus on the most important takeaways. For a risk report, highlight the overall 
        risk-adjusted return (Sharpe Ratio) and the potential downside (Max Drawdown or CVaR).
        For portfolio optimization, focus on expected improvements and key trade recommendations.
        """

        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        summary = message.content[0].text if message.content else "No summary could be generated."
        return summary
        
    except anthropic.APIError as e:
        error_details = {
            "error_type": type(e).__name__,
            "status_code": getattr(e, 'status_code', 'Unknown'),
            "timestamp": datetime.now().isoformat()
        }
        return f"API Error: Failed to generate summary due to Anthropic API issue. Status: {error_details['status_code']}"
        
    except Exception as e:
        return f"Unexpected error occurred while generating summary: {str(e)}"


def detect_market_sentiment(chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Analyzes chat history to detect overall market sentiment and mood.
    
    Args:
        chat_history: List of chat messages
        
    Returns:
        Dict with sentiment analysis results
    """
    try:
        user_messages = [msg['content'].lower() for msg in chat_history if msg.get('role') == 'user']
        
        if not user_messages:
            return {"success": True, "sentiment": "neutral", "confidence": 0.0}
        
        # Positive sentiment keywords
        positive_keywords = ['bullish', 'optimistic', 'buy', 'growth', 'opportunity', 'confident', 'strong']
        
        # Negative sentiment keywords  
        negative_keywords = ['bearish', 'pessimistic', 'sell', 'decline', 'worried', 'crash', 'fear']
        
        # Uncertainty keywords
        uncertainty_keywords = ['unsure', 'confused', 'volatile', 'uncertain', 'mixed signals']
        
        positive_count = sum(1 for msg in user_messages for kw in positive_keywords if kw in msg)
        negative_count = sum(1 for msg in user_messages for kw in negative_keywords if kw in msg)
        uncertainty_count = sum(1 for msg in user_messages for kw in uncertainty_keywords if kw in msg)
        
        total_sentiment_words = positive_count + negative_count + uncertainty_count
        
        if total_sentiment_words == 0:
            sentiment = "neutral"
            confidence = 0.0
        elif positive_count > negative_count and positive_count > uncertainty_count:
            sentiment = "positive"
            confidence = positive_count / total_sentiment_words
        elif negative_count > positive_count and negative_count > uncertainty_count:
            sentiment = "negative"
            confidence = negative_count / total_sentiment_words
        elif uncertainty_count > positive_count and uncertainty_count > negative_count:
            sentiment = "uncertain"
            confidence = uncertainty_count / total_sentiment_words
        else:
            sentiment = "mixed"
            confidence = 0.5
            
        return {
            "success": True,
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "sentiment_breakdown": {
                "positive_signals": positive_count,
                "negative_signals": negative_count,
                "uncertainty_signals": uncertainty_count
            },
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error detecting sentiment: {str(e)}",
            "sentiment": "neutral",
            "confidence": 0.0
        }