# utils/monitoring.py - FastAPI Version Compatible
"""
Rate Limiting and API Monitoring Utilities - Compatible with different FastAPI versions
"""

import time
import json
import logging
import os
from typing import Dict, Any, Optional, Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta
import asyncio
from functools import wraps

from fastapi import Request, Response, HTTPException, status
# Try different middleware imports for FastAPI compatibility
try:
    from fastapi.middleware.base import BaseHTTPMiddleware
except ImportError:
    try:
        from starlette.middleware.base import BaseHTTPMiddleware
    except ImportError:
        # Fallback for very old versions
        BaseHTTPMiddleware = None

from starlette.responses import JSONResponse

# Try to import psutil, fall back gracefully if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple rate limiter implementation"""
    
    def __init__(self):
        self.buckets: Dict[str, deque] = defaultdict(deque)
        self.cleanup_interval = 300
        self.last_cleanup = time.time()
    
    def is_allowed(
        self,
        key: str,
        max_requests: int,
        window_seconds: int,
        burst_size: int = None
    ) -> tuple[bool, Dict[str, Any]]:
        """Check if request is allowed under rate limit"""
        current_time = time.time()
        
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_buckets(current_time)
        
        bucket = self.buckets[key]
        
        # Remove requests outside the current window
        window_start = current_time - window_seconds
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        
        # Check if we're under the rate limit
        requests_in_window = len(bucket)
        
        if requests_in_window >= max_requests:
            reset_time = bucket[0] + window_seconds if bucket else current_time + window_seconds
            return False, {
                'allowed': False,
                'requests_remaining': 0,
                'reset_time': reset_time,
                'retry_after': reset_time - current_time
            }
        
        # Allow the request
        bucket.append(current_time)
        requests_remaining = max_requests - requests_in_window - 1
        
        return True, {
            'allowed': True,
            'requests_remaining': requests_remaining,
            'reset_time': window_start + window_seconds,
            'retry_after': 0
        }
    
    def _cleanup_old_buckets(self, current_time: float):
        """Remove buckets that haven't been used recently"""
        cutoff_time = current_time - 3600
        keys_to_remove = []
        
        for key, bucket in self.buckets.items():
            if not bucket or bucket[-1] < cutoff_time:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.buckets[key]
        
        self.last_cleanup = current_time

# Global rate limiter instance
rate_limiter = RateLimiter()

# Define middleware classes only if BaseHTTPMiddleware is available
if BaseHTTPMiddleware:
    class RateLimitMiddleware(BaseHTTPMiddleware):
        """FastAPI middleware for rate limiting"""
        
        def __init__(self, app, default_rate_limit: int = 100, default_window: int = 3600):
            super().__init__(app)
            self.default_rate_limit = default_rate_limit
            self.default_window = default_window
            
            self.endpoint_limits = {
                '/auth/login': {'max_requests': 5, 'window_seconds': 300},
                '/analyze': {'max_requests': 20, 'window_seconds': 3600},
                '/portfolio/*/trade-orders': {'max_requests': 10, 'window_seconds': 3600},
            }
        
        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
            user_id = getattr(request.state, 'user_id', None)
            client_key = f"user_{user_id}" if user_id else f"ip_{client_ip}"
            
            endpoint_path = request.url.path
            rate_config = self._get_rate_limit_config(endpoint_path)
            
            allowed, rate_info = rate_limiter.is_allowed(
                key=client_key,
                max_requests=rate_config['max_requests'],
                window_seconds=rate_config['window_seconds']
            )
            
            if not allowed:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "retry_after": int(rate_info['retry_after']),
                    },
                    headers={
                        "Retry-After": str(int(rate_info['retry_after'])),
                        "X-RateLimit-Limit": str(rate_config['max_requests']),
                        "X-RateLimit-Remaining": "0",
                    }
                )
            
            response = await call_next(request)
            
            response.headers["X-RateLimit-Limit"] = str(rate_config['max_requests'])
            response.headers["X-RateLimit-Remaining"] = str(rate_info['requests_remaining'])
            
            return response
        
        def _get_rate_limit_config(self, endpoint_path: str) -> Dict[str, int]:
            if endpoint_path in self.endpoint_limits:
                return self.endpoint_limits[endpoint_path]
            
            for pattern, config in self.endpoint_limits.items():
                if '*' in pattern:
                    pattern_regex = pattern.replace('*', '[^/]+')
                    import re
                    if re.match(pattern_regex, endpoint_path):
                        return config
            
            return {
                'max_requests': self.default_rate_limit,
                'window_seconds': self.default_window
            }

    class MonitoringMiddleware(BaseHTTPMiddleware):
        """FastAPI middleware for request monitoring"""
        
        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            start_time = time.time()
            
            try:
                response = await call_next(request)
                duration = time.time() - start_time
                
                performance_monitor.record_request(
                    endpoint=request.url.path,
                    duration=duration,
                    status_code=response.status_code
                )
                
                response.headers["X-Response-Time"] = f"{duration:.3f}s"
                return response
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Request failed: {str(e)}")
                
                performance_monitor.record_request(
                    endpoint=request.url.path,
                    duration=duration,
                    status_code=500
                )
                raise

else:
    # Fallback classes for when middleware isn't available
    class RateLimitMiddleware:
        def __init__(self, app, **kwargs):
            logger.warning("RateLimitMiddleware not available - FastAPI version too old")
    
    class MonitoringMiddleware:
        def __init__(self, app, **kwargs):
            logger.warning("MonitoringMiddleware not available - FastAPI version too old")

class PerformanceMonitor:
    """System performance monitoring"""
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.start_time = time.time()
    
    def record_request(self, endpoint: str, duration: float, status_code: int):
        """Record request performance metrics"""
        self.metrics['requests'].append({
            'timestamp': time.time(),
            'endpoint': endpoint,
            'duration': duration,
            'status_code': status_code
        })
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        try:
            if PSUTIL_AVAILABLE:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                return {
                    'timestamp': datetime.utcnow().isoformat(),
                    'uptime_seconds': time.time() - self.start_time,
                    'cpu_percent': cpu_percent,
                    'memory': {
                        'total_gb': round(memory.total / (1024**3), 2),
                        'available_gb': round(memory.available / (1024**3), 2),
                        'percent_used': memory.percent
                    },
                    'disk': {
                        'total_gb': round(disk.total / (1024**3), 2),
                        'free_gb': round(disk.free / (1024**3), 2),
                        'percent_used': round((disk.used / disk.total) * 100, 1)
                    }
                }
            else:
                return {
                    'timestamp': datetime.utcnow().isoformat(),
                    'uptime_seconds': time.time() - self.start_time,
                    'cpu_percent': 'not_available',
                    'memory': 'not_available',
                    'disk': 'not_available',
                    'note': 'Install psutil for detailed system metrics'
                }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {'error': 'Unable to collect system metrics'}
    
    def get_request_metrics(self, last_minutes: int = 60) -> Dict[str, Any]:
        """Get request performance metrics"""
        cutoff_time = time.time() - (last_minutes * 60)
        recent_requests = [
            req for req in self.metrics['requests']
            if req['timestamp'] > cutoff_time
        ]
        
        if not recent_requests:
            return {
                'total_requests': 0,
                'average_response_time': 0,
                'error_rate': 0,
                'requests_per_minute': 0
            }
        
        total_requests = len(recent_requests)
        total_duration = sum(req['duration'] for req in recent_requests)
        average_response_time = total_duration / total_requests
        
        error_requests = sum(1 for req in recent_requests if req['status_code'] >= 400)
        error_rate = (error_requests / total_requests) * 100
        requests_per_minute = total_requests / last_minutes
        
        endpoint_stats = defaultdict(list)
        for req in recent_requests:
            endpoint_stats[req['endpoint']].append(req['duration'])
        
        endpoint_metrics = {}
        for endpoint, durations in endpoint_stats.items():
            endpoint_metrics[endpoint] = {
                'count': len(durations),
                'avg_duration': sum(durations) / len(durations),
                'max_duration': max(durations),
                'min_duration': min(durations)
            }
        
        return {
            'time_period_minutes': last_minutes,
            'total_requests': total_requests,
            'average_response_time': round(average_response_time, 3),
            'error_rate': round(error_rate, 2),
            'requests_per_minute': round(requests_per_minute, 2),
            'endpoints': endpoint_metrics
        }

# Global instance
performance_monitor = PerformanceMonitor()

def rate_limit(max_requests: int, window_seconds: int):
    """
    Decorator for endpoint-specific rate limiting
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Basic rate limiting logic would go here
            # For now, just pass through
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Export what's available
__all__ = [
    'rate_limit', 
    'performance_monitor', 
    'RateLimitMiddleware', 
    'MonitoringMiddleware',
    'rate_limiter'
]