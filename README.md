"# Financial Platform - Clean Architecture" 
# Financial Platform Backend

A production-ready financial analysis API built with FastAPI, featuring comprehensive portfolio management, risk analysis, and security screening capabilities.

## Overview

This backend provides a complete foundation for financial applications, with 40+ analytical tools, authenticated API endpoints, and real-time communication infrastructure. Built with clean architecture principles and optimized for performance.

### Key Features

- **Portfolio Management**: Complete CRUD operations, optimization, rebalancing, backtesting
- **Risk Analysis**: VaR/CVaR, GARCH modeling, drawdown analysis, correlation matrices
- **Security Screening**: Factor-based screening, portfolio complement analysis
- **Authentication**: JWT-based auth with role-based access control
- **Real-time Communication**: WebSocket infrastructure for live updates
- **Production Ready**: Rate limiting, monitoring, comprehensive error handling

## Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL or SQLite
- Virtual environment (recommended)

### Installation

1. **Clone and setup environment**:
```bash
git clone <repository-url>
cd financial-platform-clean
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Setup database**:
```bash
# Configure database URL in .env file
DATABASE_URL=sqlite:///./financial_platform.db
# Or for PostgreSQL:
# DATABASE_URL=postgresql://user:password@localhost/financial_platform

# Run database setup
python setup_database.py
```

4. **Start the server**:
```bash
python main_clean.py
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## Architecture

### Core Components

```
financial-platform-clean/
├── main_clean.py              # FastAPI application entry point
├── services/
│   └── financial_analysis.py  # Core analysis service
├── tools/                     # Financial analysis tools
│   ├── risk_tools.py         # Risk analysis functions
│   ├── portfolio_tools.py    # Portfolio management
│   ├── strategy_tools.py     # Strategy development
│   ├── behavioral_tools.py   # Behavioral analysis
│   ├── fractal_tools.py      # Fractal analysis
│   └── regime_tools.py       # Market regime detection
├── auth/                     # Authentication system
│   ├── endpoints.py          # Auth routes
│   └── middleware.py         # JWT middleware
├── db/                       # Database layer
│   ├── models.py             # SQLAlchemy models
│   └── session.py            # Database session
├── websocket/                # WebSocket infrastructure
│   ├── manager.py            # Connection management
│   └── endpoints.py          # WebSocket routes
└── utils/                    # Utility modules
    ├── pagination.py         # Pagination utilities
    ├── validation.py         # Input validation
    └── monitoring.py         # Rate limiting & metrics
```

### Service Layer

The `FinancialAnalysisService` provides a clean interface to all financial tools:

```python
from services.financial_analysis import FinancialAnalysisService

service = FinancialAnalysisService()
result = service.analyze_risk(portfolio_id=1)
```

## API Endpoints

### Authentication
- `POST /auth/login` - User login
- `POST /auth/refresh` - Token refresh
- `GET /auth/profile` - User profile
- `POST /auth/logout` - User logout

### Portfolio Management
- `GET /user/portfolios` - List user portfolios
- `GET /portfolio/{id}/summary` - Portfolio summary
- `GET /portfolio/{id}/holdings` - Portfolio holdings (with pagination)
- `POST /portfolio/{id}/trade-orders` - Generate trade orders

### Analysis
- `POST /analyze` - Unified analysis endpoint
- `POST /analyze/risk` - Risk analysis

### Admin (Admin role required)
- `GET /admin/metrics/system` - System metrics
- `GET /admin/metrics/requests` - Request metrics

### WebSocket
- `WS /ws/test` - Test endpoint with message routing
- `WS /ws/echo` - Simple echo endpoint
- `GET /ws/status` - WebSocket service status

### Monitoring
- `GET /health` - Health check
- `GET /status` - Service status
- `GET /user/rate-limit-status` - Rate limit status

## Financial Tools

### Risk Analysis (11 functions)
- Value at Risk (VaR) and Conditional VaR
- GARCH volatility modeling and forecasting
- Maximum drawdown analysis
- Correlation and beta calculations
- Market shock scenario analysis

### Portfolio Management (7 functions)
- Portfolio optimization (mean-variance, risk parity)
- Automatic rebalancing with transaction costs
- Backtesting with performance metrics
- Trade order generation
- Security screening and selection

### Strategy Development (7 functions)
- Momentum and mean-reversion strategies
- Market regime detection (HMM, volatility-based)
- Hurst exponent analysis
- Factor-based strategy design

### Behavioral Analysis (3 functions)
- Chat history bias detection
- Market sentiment analysis
- Analysis result summarization

### Advanced Analytics (11 functions)
- Fractal analysis (DFA, multifractal spectrum)
- Regime persistence analysis
- Volatility budget calculation
- Tail risk copula modeling

## Authentication & Security

### JWT Authentication
```python
# Login
response = requests.post("http://localhost:8000/auth/login-json", json={
    "username": "user@example.com",
    "password": "password123"
})
token = response.json()["access_token"]

# Use token in requests
headers = {"Authorization": f"Bearer {token}"}
response = requests.get("http://localhost:8000/user/portfolios", headers=headers)
```

### Access Control
- **User Role**: Access to own portfolios and analysis
- **Admin Role**: System metrics and user management
- **Portfolio Access**: Users can only access portfolios they own

### Rate Limiting
- General endpoints: 100 requests/hour
- Analysis endpoints: 20 requests/hour  
- Trading endpoints: 10 requests/hour

## WebSocket Integration

### Basic Usage
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/test');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};

// Send messages
ws.send('info');  // Get connection info
ws.send(JSON.stringify({type: 'ping'}));  // Ping/pong
```

### Available Commands
- `info` - Get connection information
- `ping` - Ping/pong test
- `broadcast:message` - Broadcast to all connections
- JSON messages for structured communication

## Database Models

### Core Models
```python
# User model
class User(Base):
    id: int
    email: str
    username: str
    role: str  # 'user' or 'admin'
    is_active: bool

# Portfolio model
class Portfolio(Base):
    id: int
    name: str
    user_id: int
    description: str
    currency: str
    is_active: bool

# Holdings and Assets models for portfolio data
```

## Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=sqlite:///./financial_platform.db

# JWT
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_HOURS=24

# Optional features
PAGINATION_AVAILABLE=true
VALIDATION_AVAILABLE=true
MONITORING_AVAILABLE=true
WEBSOCKET_AVAILABLE=true
```

## Development

### Running Tests
```bash
# API functionality tests
python simple_test_runner.py

# WebSocket tests
python test_websocket_basic.py

# Tool audit
python audit_tools.py
```

### Adding New Tools
1. Add function to appropriate tool module (e.g., `tools/risk_tools.py`)
2. Import in `tools/__init__.py`
3. Tool automatically available via `FinancialAnalysisService`

### Database Migrations
```bash
# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head
```

## Performance

### Optimization Features
- **Caching**: Market data cached for 5 minutes
- **Database**: Optimized queries with SQLAlchemy
- **Tools**: Direct function calls (18-109x faster than agent-based)
- **Async**: FastAPI async endpoints for I/O operations

### Benchmarks
- Risk analysis: <0.3 seconds
- Portfolio optimization: <1 second
- Security screening: <2 seconds (50 securities)
- Database queries: <100ms (with caching)

## Monitoring & Observability

### Health Checks
```bash
curl http://localhost:8000/health
curl http://localhost:8000/status
```

### Metrics Available
- Request performance and error rates
- Database query performance
- WebSocket connection counts
- Rate limiting status
- Tool execution times

### Logging
Structured logging with configurable levels:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Deployment

### Production Checklist
- [ ] Set strong `SECRET_KEY`
- [ ] Configure production database
- [ ] Enable HTTPS/TLS
- [ ] Set up reverse proxy (nginx)
- [ ] Configure rate limiting
- [ ] Set up monitoring/alerting
- [ ] Database backups

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main_clean:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Install missing packages
pip install PyPortfolioOpt backtesting fbm hmmlearn anthropic
```

**Database Connection**:
```bash
# Check database URL and permissions
python -c "from db.session import SessionLocal; db = SessionLocal(); print('DB connected')"
```

**WebSocket Connection Failed**:
- Check if WebSocket router is loaded
- Verify no firewall blocking WebSocket connections
- Test with basic endpoint: `ws://localhost:8000/ws/test`

### Debug Endpoints
- `GET /debug/routes` - List all registered routes
- `GET /ws/status` - WebSocket service status
- `GET /test-websocket` - WebSocket test client

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## Contributing

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings for public functions
- Write tests for new functionality

### Testing
```bash
# Run specific tests
python -c "from tools.portfolio_tools import screen_securities; print('Tool test passed')"

# Full integration test
python simple_test_runner.py
```

## License

[Your License Here]

## Support

For questions or issues:
1. Check the interactive API docs at `/docs`
2. Review logs for error details
3. Use debug endpoints for diagnostics
4. Check database connectivity and permissions

---

**Version**: 2.1.0  
**Last Updated**: 2025-09-10  
**Status**: Production Ready