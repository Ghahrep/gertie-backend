# services/financial_analysis.py - Extended with Portfolio Signature Integration
"""
Financial Analysis Service - Clean Architecture with Portfolio Signature Support
=============================================================================

Extends existing service with portfolio signature generation functionality.
Direct integration of 24 working financial tools with normalization layer.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import time
import logging
from dataclasses import dataclass

# Import working tools (keep existing imports)
from tools.risk_tools import (
    calculate_risk_metrics,
    calculate_drawdowns,
    fit_garch_forecast,
    calculate_correlation_matrix,
    calculate_beta,
    apply_market_shock,
    calculate_cvar,
    generate_risk_sentiment_index
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
from db.models import Portfolio, Holding, Asset, User

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Standardized analysis result container"""
    success: bool
    analysis_type: str
    data: Dict[str, Any]
    execution_time: float
    error: Optional[str] = None


@dataclass
class PortfolioSignature:
    """Portfolio signature data structure"""
    id: int
    name: str
    description: str
    value: float
    pnl: float
    pnl_percent: float
    holdings_count: int
    risk_score: int
    volatility_forecast: int
    correlation: float
    concentration: float
    complexity: float
    tail_risk: float
    diversification: float
    market_volatility: float
    stress_index: int
    risk_level: str
    alerts: List[Dict[str, str]]
    last_updated: str
    data_quality: str


class MarketDataProvider:
    """Integrated market data provider with caching (existing implementation enhanced)"""
    
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
                db.close()
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
    """Main financial analysis service - enhanced with portfolio signature generation"""
    
    def __init__(self, market_data_provider: Optional[MarketDataProvider] = None):
        self.market_data = market_data_provider or MarketDataProvider()
        logger.info("FinancialAnalysisService initialized with signature support")
    
    # ================== NEW PORTFOLIO SIGNATURE METHODS ==================
    
    def generate_portfolio_signature(self, portfolio_id: int, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Generate comprehensive portfolio risk signature
        
        This is the main method that bridges quantitative analysis to frontend visualization
        """
        start_time = time.time()
        
        try:
            # Get portfolio data from database
            portfolio_data = self._get_portfolio_data(portfolio_id)
            if not portfolio_data:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            holdings_data = self._get_holdings_data(portfolio_id)
            if not holdings_data:
                logger.warning(f"No holdings data for portfolio {portfolio_id}")
                return self._generate_empty_signature(portfolio_data, "No holdings data")
            
            # Get returns for analysis
            returns = self.market_data.get_portfolio_returns(portfolio_id)
            if returns.empty:
                logger.warning(f"No returns data for portfolio {portfolio_id}, using synthetic data")
                returns = self._generate_synthetic_returns(252)
            
            # Run all risk analyses
            risk_results = self._run_comprehensive_analysis(returns, holdings_data)
            
            # Build signature response
            signature = self._build_signature_response(portfolio_data, holdings_data, risk_results)
            
            execution_time = time.time() - start_time
            signature["execution_time"] = execution_time
            
            logger.info(f"Generated signature for portfolio {portfolio_id} in {execution_time:.2f}s")
            return signature
            
        except Exception as e:
            logger.error(f"Error generating signature for portfolio {portfolio_id}: {e}")
            execution_time = time.time() - start_time
            return self._generate_error_signature(portfolio_id, str(e), execution_time)
    
    def _get_portfolio_data(self, portfolio_id: int) -> Optional[Dict[str, Any]]:
        """Fetch basic portfolio information"""
        try:
            db = SessionLocal()
            portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
            
            if not portfolio:
                db.close()
                return None
            
            data = {
                "id": portfolio.id,
                "name": portfolio.name,
                "description": portfolio.description or "",
                "currency": portfolio.currency,
                "is_active": portfolio.is_active,
                "user_id": portfolio.user_id,
                "created_at": portfolio.created_at
            }
            
            db.close()
            return data
            
        except Exception as e:
            logger.error(f"Error fetching portfolio data: {e}")
            return None
    
    def _get_holdings_data(self, portfolio_id: int) -> List[Dict[str, Any]]:
        """Fetch current holdings with latest prices"""
        try:
            db = SessionLocal()
            portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
            
            if not portfolio or not portfolio.holdings:
                db.close()
                return []
            
            holdings_data = []
            for holding in portfolio.holdings:
                current_price = holding.asset.current_price or holding.purchase_price
                market_value = holding.shares * current_price
                cost_basis = holding.cost_basis or (holding.shares * holding.purchase_price)
                
                holdings_data.append({
                    "symbol": holding.asset.ticker,
                    "name": holding.asset.name,
                    "sector": holding.asset.sector,
                    "asset_type": holding.asset.asset_type,
                    "shares": holding.shares,
                    "current_price": current_price,
                    "purchase_price": holding.purchase_price,
                    "market_value": market_value,
                    "cost_basis": cost_basis,
                    "pnl": market_value - cost_basis,
                    "pnl_percent": ((market_value - cost_basis) / cost_basis * 100) if cost_basis > 0 else 0
                })
            
            db.close()
            return holdings_data
            
        except Exception as e:
            logger.error(f"Error fetching holdings data: {e}")
            return []
    
    def _run_comprehensive_analysis(self, returns: pd.Series, holdings_data: List[Dict]) -> Dict[str, Any]:
        """Run all risk analysis tools and return consolidated results"""
        try:
            results = {}
            
            # Core risk metrics
            try:
                risk_metrics = calculate_risk_metrics(returns.values)
                results["risk_metrics"] = risk_metrics
            except Exception as e:
                logger.warning(f"Risk metrics calculation failed: {e}")
                results["risk_metrics"] = self._get_fallback_risk_metrics()
            
            # Risk sentiment
            try:
                sentiment_index = generate_risk_sentiment_index(returns.values)
                results["sentiment_index"] = sentiment_index
            except Exception as e:
                logger.warning(f"Risk sentiment calculation failed: {e}")
                results["sentiment_index"] = 0.0
            
            # CVaR analysis
            try:
                cvar_95 = calculate_cvar(returns.values, confidence_level=0.95)
                cvar_99 = calculate_cvar(returns.values, confidence_level=0.99)
                results["cvar_metrics"] = {"cvar_95": cvar_95, "cvar_99": cvar_99}
            except Exception as e:
                logger.warning(f"CVaR calculation failed: {e}")
                results["cvar_metrics"] = {"cvar_95": -0.05, "cvar_99": -0.08}
            
            # GARCH forecasting
            try:
                if len(returns) >= 100:
                    garch_forecast = fit_garch_forecast(returns.values)
                    results["garch_forecast"] = garch_forecast
                else:
                    results["garch_forecast"] = {"current_vol": 0.02, "forecast_vol": 0.02, "trend": "stable"}
            except Exception as e:
                logger.warning(f"GARCH forecast failed: {e}")
                results["garch_forecast"] = {"current_vol": 0.02, "forecast_vol": 0.02, "trend": "stable"}
            
            # Correlation analysis
            try:
                if len(holdings_data) > 1:
                    tickers = [h["symbol"] for h in holdings_data]
                    correlation_matrix = calculate_correlation_matrix(tickers)
                    avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
                    results["correlation_metrics"] = {
                        "average_correlation": avg_correlation,
                        "matrix": correlation_matrix.tolist()
                    }
                else:
                    results["correlation_metrics"] = {"average_correlation": 0.0, "matrix": [[1.0]]}
            except Exception as e:
                logger.warning(f"Correlation analysis failed: {e}")
                results["correlation_metrics"] = {"average_correlation": 0.5, "matrix": [[1.0]]}
            
            # Multifractal analysis
            try:
                if len(returns) >= 256:
                    fractal_results = calculate_multifractal_spectrum(returns.values)
                    results["fractal_metrics"] = fractal_results
                else:
                    results["fractal_metrics"] = {"spectrum_width": 0.5, "complexity_score": 0.5}
            except Exception as e:
                logger.warning(f"Fractal analysis failed: {e}")
                results["fractal_metrics"] = {"spectrum_width": 0.5, "complexity_score": 0.5}
            
            # Concentration metrics
            try:
                total_value = sum(h["market_value"] for h in holdings_data)
                if total_value > 0:
                    weights = [h["market_value"] / total_value for h in holdings_data]
                    hhi = sum(w**2 for w in weights)
                    effective_stocks = 1.0 / hhi if hhi > 0 else 1.0
                    top_weight = max(weights) if weights else 1.0
                    
                    results["concentration_metrics"] = {
                        "herfindahl_index": hhi,
                        "effective_stocks": effective_stocks,
                        "top_holding_weight": top_weight
                    }
                else:
                    results["concentration_metrics"] = {
                        "herfindahl_index": 1.0,
                        "effective_stocks": 1.0,
                        "top_holding_weight": 1.0
                    }
            except Exception as e:
                logger.warning(f"Concentration analysis failed: {e}")
                results["concentration_metrics"] = {
                    "herfindahl_index": 0.5,
                    "effective_stocks": 2.0,
                    "top_holding_weight": 0.5
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return self._get_fallback_analysis_results()
    
    def _build_signature_response(self, portfolio_data: Dict, holdings_data: List[Dict], 
                                 risk_results: Dict) -> Dict[str, Any]:
        """Build the final signature response with normalized metrics"""
        try:
            # Calculate basic portfolio metrics
            total_value = sum(h["market_value"] for h in holdings_data)
            total_pnl = sum(h["pnl"] for h in holdings_data)
            holdings_count = len(holdings_data)
            
            # Normalize risk metrics for frontend
            risk_score = self._normalize_sentiment_to_risk_score(risk_results.get("sentiment_index", 0.0))
            volatility_forecast = self._normalize_volatility_forecast(risk_results.get("garch_forecast", {}))
            correlation = self._normalize_correlation(risk_results.get("correlation_metrics", {}).get("average_correlation", 0.5))
            concentration = self._normalize_concentration(risk_results.get("concentration_metrics", {}).get("herfindahl_index", 0.5))
            complexity = self._normalize_complexity(risk_results.get("fractal_metrics", {}).get("complexity_score", 0.5))
            tail_risk = self._normalize_tail_risk(risk_results.get("cvar_metrics", {}).get("cvar_95", -0.05))
            diversification = self._normalize_diversification(risk_results.get("concentration_metrics", {}), holdings_count)
            
            # Generate alerts
            alerts = self._generate_portfolio_alerts(risk_results, holdings_data)
            
            # Determine data quality
            data_quality = "live" if all(h.get("current_price") for h in holdings_data) else "synthetic"
            
            return {
                # Basic portfolio info
                "id": portfolio_data["id"],
                "name": portfolio_data["name"],
                "description": portfolio_data["description"],
                "value": round(total_value, 2),
                "pnl": round(total_pnl, 2),
                "pnlPercent": round((total_pnl / (total_value - total_pnl) * 100), 2) if (total_value - total_pnl) > 0 else 0,
                "holdingsCount": holdings_count,
                
                # Risk signature metrics (normalized for frontend)
                "riskScore": risk_score,
                "volatilityForecast": volatility_forecast,
                "correlation": correlation,
                "concentration": concentration,
                "complexity": complexity,
                "tailRisk": tail_risk,
                "diversification": diversification,
                
                # Additional context
                "marketVolatility": risk_results.get("garch_forecast", {}).get("current_vol", 0.02),
                "stressIndex": risk_score,
                "riskLevel": self._categorize_risk_level(risk_score),
                
                # Alerts and metadata
                "alerts": alerts,
                "lastUpdated": datetime.now().isoformat(),
                "dataQuality": data_quality
            }
            
        except Exception as e:
            logger.error(f"Error building signature response: {e}")
            return self._generate_error_signature(portfolio_data.get("id", 0), str(e), 0.0)
    
    # ================ NORMALIZATION METHODS ================
    
    def _normalize_sentiment_to_risk_score(self, sentiment_index: float) -> int:
        """Convert sentiment index (-1 to 1) to risk score (0-100)"""
        # Sentiment index: -1 (low risk) to 1 (high risk)
        normalized = max(0, min(100, (sentiment_index + 1) * 50))
        return int(normalized)
    
    def _normalize_volatility_forecast(self, garch_forecast: Dict) -> int:
        """Convert GARCH forecast to volatility intensity (0-100)"""
        current_vol = garch_forecast.get("current_vol", 0.02)
        forecast_vol = garch_forecast.get("forecast_vol", 0.02)
        
        # Average volatility scaled to 0-100
        avg_vol = (current_vol + forecast_vol) / 2
        annual_vol = avg_vol * np.sqrt(252)  # Annualize
        
        # Scale: 0% vol = 0, 50% vol = 100
        normalized = min(100, annual_vol * 100 / 0.5)
        return int(normalized)
    
    def _normalize_correlation(self, avg_correlation: float) -> float:
        """Normalize correlation (-1 to 1) to (0 to 1)"""
        return max(0, min(1, (avg_correlation + 1) / 2))
    
    def _normalize_concentration(self, hhi: float) -> float:
        """Normalize Herfindahl index to concentration score"""
        return max(0, min(1, hhi))
    
    def _normalize_complexity(self, complexity_score: float) -> float:
        """Normalize multifractal complexity score"""
        return max(0, min(1, complexity_score))
    
    def _normalize_tail_risk(self, cvar_95: float) -> float:
        """Normalize CVaR to tail risk score (0-1)"""
        # CVaR is negative, more negative = higher risk
        # Map -20% CVaR to 1.0 risk, 0% to 0.0 risk
        return max(0, min(1, abs(cvar_95) / 0.20))
    
    def _normalize_diversification(self, concentration_metrics: Dict, holdings_count: int) -> float:
        """Calculate diversification score based on effective stocks"""
        effective_stocks = concentration_metrics.get("effective_stocks", 1.0)
        max_effective = min(holdings_count, 10)  # Cap at 10
        return max(0, min(1, effective_stocks / max_effective)) if max_effective > 0 else 0
    
    def _categorize_risk_level(self, risk_score: int) -> str:
        """Categorize risk score into levels"""
        if risk_score < 30:
            return "LOW"
        elif risk_score < 70:
            return "MODERATE"
        else:
            return "HIGH"
    
    def _generate_portfolio_alerts(self, risk_results: Dict, holdings_data: List[Dict]) -> List[Dict[str, str]]:
        """Generate alerts based on risk analysis"""
        alerts = []
        
        try:
            # High concentration alert
            concentration = risk_results.get("concentration_metrics", {})
            top_weight = concentration.get("top_holding_weight", 0)
            if top_weight > 0.4:
                alerts.append({
                    "type": "concentration",
                    "severity": "medium",
                    "message": f"Single holding represents {top_weight*100:.1f}% of portfolio"
                })
            
            # High correlation alert
            correlation = risk_results.get("correlation_metrics", {})
            avg_corr = correlation.get("average_correlation", 0)
            if avg_corr > 0.8:
                alerts.append({
                    "type": "correlation",
                    "severity": "medium",
                    "message": "High correlation between holdings reduces diversification benefits"
                })
            
            # Volatility alert
            garch = risk_results.get("garch_forecast", {})
            current_vol = garch.get("current_vol", 0)
            if current_vol * np.sqrt(252) > 0.3:  # 30% annual volatility
                alerts.append({
                    "type": "volatility",
                    "severity": "high",
                    "message": "Portfolio volatility exceeds 30% annually"
                })
            
            # CVaR alert
            cvar = risk_results.get("cvar_metrics", {})
            cvar_95 = cvar.get("cvar_95", 0)
            if abs(cvar_95) > 0.15:  # 15% potential loss
                alerts.append({
                    "type": "tail_risk",
                    "severity": "high",
                    "message": f"Potential 5% worst-case loss exceeds {abs(cvar_95)*100:.1f}%"
                })
            
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
        
        return alerts
    
    # ================ FALLBACK AND ERROR HANDLING ================
    
    def _generate_synthetic_returns(self, days: int = 252) -> pd.Series:
        """Generate synthetic returns for portfolios without historical data"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        returns = np.random.normal(0.0008, 0.02, days)  # 20% annual vol, slight positive drift
        
        # Add volatility clustering
        for i in range(1, len(returns)):
            if abs(returns[i-1]) > 0.03:
                returns[i] *= 1.5
        
        return pd.Series(returns, index=dates)
    
    def _get_fallback_risk_metrics(self) -> Dict[str, Any]:
        """Fallback risk metrics when calculations fail"""
        return {
            "performance_stats": {
                "annualized_volatility_pct": 15.0,
                "total_return_pct": 8.0
            },
            "risk_adjusted_ratios": {
                "sharpe_ratio": 0.5,
                "sortino_ratio": 0.6
            }
        }
    
    def _get_fallback_analysis_results(self) -> Dict[str, Any]:
        """Fallback analysis results when comprehensive analysis fails"""
        return {
            "sentiment_index": 0.0,
            "cvar_metrics": {"cvar_95": -0.05, "cvar_99": -0.08},
            "garch_forecast": {"current_vol": 0.02, "forecast_vol": 0.02, "trend": "stable"},
            "correlation_metrics": {"average_correlation": 0.5, "matrix": [[1.0]]},
            "fractal_metrics": {"spectrum_width": 0.5, "complexity_score": 0.5},
            "concentration_metrics": {"herfindahl_index": 0.5, "effective_stocks": 2.0, "top_holding_weight": 0.5}
        }
    
    def _generate_empty_signature(self, portfolio_data: Dict, reason: str) -> Dict[str, Any]:
        """Generate empty signature for portfolios without data"""
        return {
            "id": portfolio_data["id"],
            "name": portfolio_data["name"],
            "description": portfolio_data["description"],
            "value": 0.0,
            "pnl": 0.0,
            "pnlPercent": 0.0,
            "holdingsCount": 0,
            "riskScore": 0,
            "volatilityForecast": 0,
            "correlation": 0.0,
            "concentration": 0.0,
            "complexity": 0.0,
            "tailRisk": 0.0,
            "diversification": 0.0,
            "marketVolatility": 0.0,
            "stressIndex": 0,
            "riskLevel": "LOW",
            "alerts": [{"type": "info", "severity": "low", "message": reason}],
            "lastUpdated": datetime.now().isoformat(),
            "dataQuality": "empty"
        }
    
    def _generate_error_signature(self, portfolio_id: int, error_msg: str, execution_time: float) -> Dict[str, Any]:
        """Generate error signature when analysis fails"""
        return {
            "id": portfolio_id,
            "name": "Unknown Portfolio",
            "description": "Error occurred during analysis",
            "value": 0.0,
            "pnl": 0.0,
            "pnlPercent": 0.0,
            "holdingsCount": 0,
            "riskScore": 0,
            "volatilityForecast": 0,
            "correlation": 0.0,
            "concentration": 0.0,
            "complexity": 0.0,
            "tailRisk": 0.0,
            "diversification": 0.0,
            "marketVolatility": 0.0,
            "stressIndex": 0,
            "riskLevel": "LOW",
            "alerts": [{"type": "error", "severity": "high", "message": f"Analysis failed: {error_msg}"}],
            "lastUpdated": datetime.now().isoformat(),
            "dataQuality": "error",
            "execution_time": execution_time
        }
    
    # ================ EXISTING METHODS (keep unchanged) ================
    
    def analyze_risk(self, portfolio_id: int, **kwargs) -> AnalysisResult:
        """Comprehensive risk analysis using working tools (existing method)"""
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
                    "overall_risk_level": self._categorize_risk_level_legacy(risk_metrics),
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
        """Behavioral analysis using working behavioral tools (existing method)"""
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
        """Market regime analysis using working regime tools (existing method)"""
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
        """Fractal analysis using working fractal tools (existing method)"""
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
        """Simple query routing to appropriate analysis method (existing method)"""
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
    
    def _categorize_risk_level_legacy(self, risk_metrics: Dict[str, Any]) -> str:
        """Categorize overall risk level (existing method)"""
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
        """Identify key risk concerns (existing method)"""
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
        """Assess regime stability (existing method)"""
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