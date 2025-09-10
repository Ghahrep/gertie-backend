# services/financial_analysis.py - Core Financial Analysis Service
"""
Financial Analysis Service - Clean Architecture
==============================================

Replaces complex orchestrator with simple service class.
Direct integration of 24 working financial tools.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import time
import logging
from dataclasses import dataclass

# Import working tools
from tools.risk_tools import (
    calculate_risk_metrics,
    calculate_drawdowns,
    fit_garch_forecast,
    calculate_correlation_matrix,
    calculate_beta,
    apply_market_shock
)

from tools.behavioral_tools import (
    analyze_chat_for_biases,
    summarize_analysis_results,
    detect_market_sentiment
)

from tools.fractal_tools import (
    calculate_hurst,
    calculate_multifractal_spectrum
)

from tools.regime_tools import (
    detect_hmm_regimes,
    detect_volatility_regimes,
    analyze_regime_persistence
)

from tools.strategy_tools import (
    design_momentum_strategy,
    design_mean_reversion_strategy
)

# Database imports
from db.session import SessionLocal
from db.models import Portfolio, Holding, Asset

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Standardized analysis result container"""
    success: bool
    analysis_type: str
    data: Dict[str, Any]
    execution_time: float
    error: Optional[str] = None


class MarketDataProvider:
    """Integrated market data provider with caching"""
    
    def __init__(self, cache_ttl: int = 300):  # 5-minute cache
        self.cache = {}
        self.cache_ttl = cache_ttl
        
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache:
            return False
        return time.time() - self.cache[key]['timestamp'] < self.cache_ttl
    
    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Get current market prices with caching"""
        cache_key = f"prices_{','.join(sorted(tickers))}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            prices = {}
            for ticker in tickers:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d", interval="1m")
                if not hist.empty:
                    prices[ticker] = float(hist['Close'].iloc[-1])
                else:
                    logger.warning(f"No price data for {ticker}")
                    prices[ticker] = 0.0
            
            self.cache[cache_key] = {
                'data': prices,
                'timestamp': time.time()
            }
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            return {ticker: 0.0 for ticker in tickers}
    
    def get_historical_data(self, tickers: List[str], period: str = "1y") -> pd.DataFrame:
        """Get historical price data"""
        cache_key = f"history_{','.join(sorted(tickers))}_{period}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            data = yf.download(tickers, period=period, auto_adjust=True)
            if len(tickers) == 1:
                # yfinance returns Series for single ticker, convert to DataFrame
                data = pd.DataFrame(data)
                data.columns = [f"{col}_{tickers[0]}" for col in data.columns]
            
            self.cache[cache_key] = {
                'data': data,
                'timestamp': time.time()
            }
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def get_portfolio_returns(self, portfolio_id: int, period: str = "1y") -> pd.Series:
        """Generate portfolio return series from database holdings"""
        try:
            db = SessionLocal()
            portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
            
            if not portfolio or not portfolio.holdings:
                return pd.Series()
            
            # Get tickers and weights
            tickers = []
            weights = []
            total_value = 0
            
            for holding in portfolio.holdings:
                tickers.append(holding.asset.ticker)
                value = holding.shares * (holding.asset.current_price or 100)
                total_value += value
            
            # Calculate weights
            for holding in portfolio.holdings:
                value = holding.shares * (holding.asset.current_price or 100)
                weights.append(value / total_value if total_value > 0 else 0)
            
            db.close()
            
            # Get historical data
            hist_data = self.get_historical_data(tickers, period)
            if hist_data.empty:
                return pd.Series()
            
            # Calculate portfolio returns
            price_cols = [col for col in hist_data.columns if 'Close' in col or col == 'Close']
            if not price_cols:
                return pd.Series()
            
            returns_data = hist_data[price_cols].pct_change().dropna()
            
            # Weight the returns
            portfolio_returns = (returns_data * weights).sum(axis=1)
            return portfolio_returns
            
        except Exception as e:
            logger.error(f"Error generating portfolio returns: {e}")
            return pd.Series()


class FinancialAnalysisService:
    """Main financial analysis service - replaces complex orchestrator"""
    
    def __init__(self, market_data_provider: Optional[MarketDataProvider] = None):
        self.market_data = market_data_provider or MarketDataProvider()
        logger.info("FinancialAnalysisService initialized")
    
    def analyze_risk(self, portfolio_id: int, **kwargs) -> AnalysisResult:
        """Comprehensive risk analysis using working tools"""
        start_time = time.time()
        
        try:
            # Get portfolio returns
            returns = self.market_data.get_portfolio_returns(portfolio_id)
            if returns.empty:
                return AnalysisResult(
                    success=False,
                    analysis_type="risk",
                    data={},
                    execution_time=time.time() - start_time,
                    error="Could not retrieve portfolio data"
                )
            
            # Core risk metrics
            risk_metrics = calculate_risk_metrics(returns)
            drawdowns = calculate_drawdowns(returns)
            
            # Advanced risk analysis
            shock_results = apply_market_shock(returns)
            
            # GARCH forecasting (if enough data)
            garch_forecast = None
            if len(returns) >= 100:
                garch_forecast = fit_garch_forecast(returns)
            
            analysis_data = {
                "portfolio_id": portfolio_id,
                "analysis_date": datetime.now().isoformat(),
                "risk_metrics": risk_metrics,
                "drawdown_analysis": drawdowns,
                "shock_testing": shock_results,
                "volatility_forecast": garch_forecast,
                "data_points": len(returns),
                "summary": {
                    "overall_risk_level": self._categorize_risk_level(risk_metrics),
                    "key_concerns": self._identify_key_risks(risk_metrics, drawdowns)
                }
            }
            
            return AnalysisResult(
                success=True,
                analysis_type="risk",
                data=analysis_data,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return AnalysisResult(
                success=False,
                analysis_type="risk",
                data={},
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    def analyze_behavior(self, chat_history: List[Dict[str, str]], **kwargs) -> AnalysisResult:
        """Behavioral analysis using working behavioral tools"""
        start_time = time.time()
        
        try:
            # Bias detection
            bias_analysis = analyze_chat_for_biases(chat_history)
            
            # Sentiment analysis
            sentiment_analysis = detect_market_sentiment(chat_history)
            
            analysis_data = {
                "analysis_date": datetime.now().isoformat(),
                "bias_analysis": bias_analysis,
                "sentiment_analysis": sentiment_analysis,
                "chat_message_count": len(chat_history),
                "summary": {
                    "primary_biases": list(bias_analysis.get('biases_detected', {}).keys()),
                    "overall_sentiment": sentiment_analysis.get('sentiment', 'neutral'),
                    "confidence": sentiment_analysis.get('confidence', 0.0)
                }
            }
            
            return AnalysisResult(
                success=True,
                analysis_type="behavior",
                data=analysis_data,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Behavioral analysis failed: {e}")
            return AnalysisResult(
                success=False,
                analysis_type="behavior",
                data={},
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    def analyze_regimes(self, portfolio_id: int, **kwargs) -> AnalysisResult:
        """Market regime analysis using working regime tools"""
        start_time = time.time()
        
        try:
            # Get portfolio returns
            returns = self.market_data.get_portfolio_returns(portfolio_id)
            if returns.empty:
                return AnalysisResult(
                    success=False,
                    analysis_type="regimes",
                    data={},
                    execution_time=time.time() - start_time,
                    error="Could not retrieve portfolio data"
                )
            
            # Volatility regime detection
            vol_regimes = detect_volatility_regimes(returns)
            
            # HMM regime detection (if enough data)
            hmm_regimes = None
            if len(returns) >= 50:
                hmm_regimes = detect_hmm_regimes(returns, n_regimes=2)
            
            # Regime persistence analysis
            persistence = None
            if vol_regimes and 'regime_series' in vol_regimes:
                persistence = analyze_regime_persistence(vol_regimes['regime_series'])
            
            analysis_data = {
                "portfolio_id": portfolio_id,
                "analysis_date": datetime.now().isoformat(),
                "volatility_regimes": vol_regimes,
                "hmm_regimes": hmm_regimes,
                "regime_persistence": persistence,
                "data_points": len(returns),
                "summary": {
                    "current_regime": vol_regimes.get('current_regime', 'Unknown'),
                    "regime_stability": self._assess_regime_stability(persistence)
                }
            }
            
            return AnalysisResult(
                success=True,
                analysis_type="regimes",
                data=analysis_data,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Regime analysis failed: {e}")
            return AnalysisResult(
                success=False,
                analysis_type="regimes",
                data={},
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    def analyze_fractals(self, portfolio_id: int, **kwargs) -> AnalysisResult:
        """Fractal analysis using working fractal tools"""
        start_time = time.time()
        
        try:
            # Get portfolio returns
            returns = self.market_data.get_portfolio_returns(portfolio_id)
            if returns.empty:
                return AnalysisResult(
                    success=False,
                    analysis_type="fractals",
                    data={},
                    execution_time=time.time() - start_time,
                    error="Could not retrieve portfolio data"
                )
            
            # Hurst exponent calculation
            hurst_result = calculate_hurst(returns)
            
            # Multifractal spectrum (if enough data)
            multifractal_result = None
            if len(returns) >= 256:
                try:
                    multifractal_result = calculate_multifractal_spectrum(returns)
                except Exception as e:
                    logger.warning(f"Multifractal analysis failed: {e}")
            
            analysis_data = {
                "portfolio_id": portfolio_id,
                "analysis_date": datetime.now().isoformat(),
                "hurst_analysis": hurst_result,
                "multifractal_analysis": multifractal_result,
                "data_points": len(returns),
                "summary": {
                    "memory_characteristics": hurst_result.get('interpretation', 'Unknown'),
                    "hurst_exponent": hurst_result.get('hurst_exponent', 0.5),
                    "multifractality": multifractal_result.get('multifractality_level', 'Unknown') if multifractal_result else 'Insufficient Data'
                }
            }
            
            return AnalysisResult(
                success=True,
                analysis_type="fractals",
                data=analysis_data,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Fractal analysis failed: {e}")
            return AnalysisResult(
                success=False,
                analysis_type="fractals",
                data={},
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    def analyze(self, query: str, portfolio_id: Optional[int] = None, **kwargs) -> AnalysisResult:
        """Simple query routing to appropriate analysis method"""
        query_lower = query.lower()
        
        # Risk analysis keywords
        if any(keyword in query_lower for keyword in ['risk', 'var', 'volatility', 'drawdown', 'sharp']):
            if portfolio_id:
                return self.analyze_risk(portfolio_id, **kwargs)
            else:
                return AnalysisResult(
                    success=False,
                    analysis_type="risk",
                    data={},
                    execution_time=0.0,
                    error="Portfolio ID required for risk analysis"
                )
        
        # Behavioral analysis keywords
        elif any(keyword in query_lower for keyword in ['bias', 'behavior', 'sentiment', 'emotion']):
            chat_history = kwargs.get('chat_history', [])
            return self.analyze_behavior(chat_history)
        
        # Regime analysis keywords
        elif any(keyword in query_lower for keyword in ['regime', 'market state', 'volatility regime']):
            if portfolio_id:
                return self.analyze_regimes(portfolio_id, **kwargs)
            else:
                return AnalysisResult(
                    success=False,
                    analysis_type="regimes",
                    data={},
                    execution_time=0.0,
                    error="Portfolio ID required for regime analysis"
                )
        
        # Fractal analysis keywords
        elif any(keyword in query_lower for keyword in ['fractal', 'hurst', 'memory', 'multifractal']):
            if portfolio_id:
                return self.analyze_fractals(portfolio_id, **kwargs)
            else:
                return AnalysisResult(
                    success=False,
                    analysis_type="fractals",
                    data={},
                    execution_time=0.0,
                    error="Portfolio ID required for fractal analysis"
                )
        
        # Default to risk analysis if portfolio provided
        elif portfolio_id:
            return self.analyze_risk(portfolio_id, **kwargs)
        
        else:
            return AnalysisResult(
                success=False,
                analysis_type="unknown",
                data={},
                execution_time=0.0,
                error=f"Could not determine analysis type from query: {query}"
            )
    
    def _categorize_risk_level(self, risk_metrics: Dict[str, Any]) -> str:
        """Categorize overall risk level"""
        if not risk_metrics:
            return "Unknown"
        
        try:
            sharpe = risk_metrics.get('risk_adjusted_ratios', {}).get('sharpe_ratio', 0)
            vol = risk_metrics.get('performance_stats', {}).get('annualized_volatility_pct', 0)
            
            if sharpe > 1.0 and vol < 15:
                return "Low"
            elif sharpe > 0.5 and vol < 25:
                return "Medium"
            else:
                return "High"
        except:
            return "Unknown"
    
    def _identify_key_risks(self, risk_metrics: Dict[str, Any], drawdowns: Dict[str, Any]) -> List[str]:
        """Identify key risk concerns"""
        concerns = []
        
        try:
            if risk_metrics:
                vol = risk_metrics.get('performance_stats', {}).get('annualized_volatility_pct', 0)
                if vol > 30:
                    concerns.append("High volatility")
                
                sharpe = risk_metrics.get('risk_adjusted_ratios', {}).get('sharpe_ratio', 0)
                if sharpe < 0:
                    concerns.append("Negative risk-adjusted returns")
            
            if drawdowns:
                max_dd = abs(drawdowns.get('max_drawdown_pct', 0))
                if max_dd > 20:
                    concerns.append("Large historical drawdowns")
        except:
            pass
        
        return concerns if concerns else ["No major concerns identified"]
    
    def _assess_regime_stability(self, persistence: Dict[str, Any]) -> str:
        """Assess regime stability"""
        if not persistence:
            return "Unknown"
        
        try:
            avg_duration = persistence.get('avg_regime_duration', 0)
            if avg_duration > 30:
                return "Stable"
            elif avg_duration > 10:
                return "Moderate"
            else:
                return "Unstable"
        except:
            return "Unknown"