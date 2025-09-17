# utils/cache.py - Caching Utilities for Portfolio Signatures
"""
Portfolio Signature Caching System
==================================
Provides both in-memory and Redis caching for portfolio signatures.
Optimizes performance while maintaining data freshness.
"""

import json
import asyncio
from typing import Optional, Dict, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
import logging

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Cache configuration settings"""
    default_ttl: int = 300  # 5 minutes
    max_memory_entries: int = 1000
    redis_url: Optional[str] = None
    enable_redis: bool = True
    enable_memory: bool = True

class MemoryCache:
    """In-memory cache with TTL support"""
    
    def __init__(self, max_entries: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_entries = max_entries
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached value if not expired"""
        async with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            if datetime.now() > entry['expires_at']:
                del self.cache[key]
                return None
            
            entry['last_accessed'] = datetime.now()
            return entry['data']
    
    async def set(self, key: str, value: Dict[str, Any], ttl: int = 300):
        """Set cached value with TTL"""
        async with self._lock:
            # Evict old entries if at capacity
            if len(self.cache) >= self.max_entries:
                await self._evict_lru()
            
            self.cache[key] = {
                'data': value,
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(seconds=ttl),
                'last_accessed': datetime.now()
            }
    
    async def delete(self, key: str):
        """Remove key from cache"""
        async with self._lock:
            self.cache.pop(key, None)
    
    async def clear(self):
        """Clear all cache entries"""
        async with self._lock:
            self.cache.clear()
    
    async def _evict_lru(self):
        """Evict least recently used entries"""
        if not self.cache:
            return
        
        # Sort by last_accessed and remove oldest 10%
        sorted_keys = sorted(
            self.cache.keys(),
            key=lambda k: self.cache[k]['last_accessed']
        )
        
        evict_count = max(1, len(sorted_keys) // 10)
        for key in sorted_keys[:evict_count]:
            del self.cache[key]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self._lock:
            total_entries = len(self.cache)
            expired_entries = sum(
                1 for entry in self.cache.values()
                if datetime.now() > entry['expires_at']
            )
            
            return {
                'total_entries': total_entries,
                'expired_entries': expired_entries,
                'memory_usage_mb': len(str(self.cache)) / (1024 * 1024),
                'hit_rate': None  # Calculated externally
            }

class RedisCache:
    """Redis-based cache for distributed caching"""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self._connection_pool = None
    
    async def connect(self):
        """Initialize Redis connection"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, falling back to memory cache")
            return False
        
        try:
            self._connection_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=20,
                decode_responses=True
            )
            self.redis_client = redis.Redis(connection_pool=self._connection_pool)
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis cache connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached value from Redis"""
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Dict[str, Any], ttl: int = 300):
        """Set cached value in Redis with TTL"""
        if not self.redis_client:
            return
        
        try:
            cached_data = json.dumps(value, default=str)
            await self.redis_client.setex(key, ttl, cached_data)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    async def delete(self, key: str):
        """Remove key from Redis"""
        if not self.redis_client:
            return
        
        try:
            await self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
    
    async def clear_pattern(self, pattern: str = "portfolio_signature:*"):
        """Clear keys matching pattern"""
        if not self.redis_client:
            return
        
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis clear pattern error: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        if not self.redis_client:
            return {}
        
        try:
            info = await self.redis_client.info('memory')
            return {
                'used_memory_mb': info.get('used_memory', 0) / (1024 * 1024),
                'connected_clients': info.get('connected_clients', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {}
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
        if self._connection_pool:
            await self._connection_pool.disconnect()

class PortfolioSignatureCache:
    """Multi-tier cache for portfolio signatures"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.memory_cache = MemoryCache(config.max_memory_entries) if config.enable_memory else None
        self.redis_cache = RedisCache(config.redis_url) if config.enable_redis and config.redis_url else None
        
        # Statistics
        self.stats = {
            'requests': 0,
            'memory_hits': 0,
            'redis_hits': 0,
            'misses': 0
        }
    
    async def initialize(self):
        """Initialize cache connections"""
        if self.redis_cache:
            redis_connected = await self.redis_cache.connect()
            if not redis_connected:
                self.redis_cache = None
    
    def _generate_key(self, portfolio_id: int, force_refresh: bool = False) -> str:
        """Generate cache key for portfolio signature"""
        base_key = f"portfolio_signature:{portfolio_id}"
        if force_refresh:
            # Add timestamp to bypass cache for force refresh
            timestamp = int(datetime.now().timestamp())
            base_key += f":force:{timestamp}"
        return base_key
    
    async def get_signature(self, portfolio_id: int, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Get portfolio signature from cache"""
        if force_refresh:
            return None
        
        self.stats['requests'] += 1
        cache_key = self._generate_key(portfolio_id)
        
        # Try memory cache first
        if self.memory_cache:
            result = await self.memory_cache.get(cache_key)
            if result:
                self.stats['memory_hits'] += 1
                return result
        
        # Try Redis cache
        if self.redis_cache:
            result = await self.redis_cache.get(cache_key)
            if result:
                self.stats['redis_hits'] += 1
                
                # Backfill memory cache
                if self.memory_cache:
                    await self.memory_cache.set(cache_key, result, self.config.default_ttl)
                
                return result
        
        self.stats['misses'] += 1
        return None
    
    async def set_signature(self, portfolio_id: int, signature_data: Dict[str, Any], ttl: Optional[int] = None):
        """Set portfolio signature in cache"""
        cache_key = self._generate_key(portfolio_id)
        ttl = ttl or self.config.default_ttl
        
        # Add caching metadata
        cached_signature = {
            **signature_data,
            'cached_at': datetime.now().isoformat(),
            'cache_ttl': ttl
        }
        
        # Set in memory cache
        if self.memory_cache:
            await self.memory_cache.set(cache_key, cached_signature, ttl)
        
        # Set in Redis cache
        if self.redis_cache:
            await self.redis_cache.set(cache_key, cached_signature, ttl)
    
    async def invalidate_portfolio(self, portfolio_id: int):
        """Invalidate cache for specific portfolio"""
        cache_key = self._generate_key(portfolio_id)
        
        if self.memory_cache:
            await self.memory_cache.delete(cache_key)
        
        if self.redis_cache:
            await self.redis_cache.delete(cache_key)
    
    async def clear_all(self):
        """Clear all cached signatures"""
        if self.memory_cache:
            await self.memory_cache.clear()
        
        if self.redis_cache:
            await self.redis_cache.clear_pattern("portfolio_signature:*")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        memory_stats = {}
        redis_stats = {}
        
        if self.memory_cache:
            memory_stats = await self.memory_cache.get_stats()
        
        if self.redis_cache:
            redis_stats = await self.redis_cache.get_stats()
        
        # Calculate hit rates
        total_requests = self.stats['requests']
        hit_rate = 0.0
        if total_requests > 0:
            total_hits = self.stats['memory_hits'] + self.stats['redis_hits']
            hit_rate = total_hits / total_requests
        
        return {
            'requests': self.stats,
            'hit_rate': hit_rate,
            'memory': memory_stats,
            'redis': redis_stats,
            'cache_layers': {
                'memory_enabled': self.memory_cache is not None,
                'redis_enabled': self.redis_cache is not None
            }
        }
    
    async def close(self):
        """Close cache connections"""
        if self.redis_cache:
            await self.redis_cache.close()

# Global cache instance (initialized in main.py)
portfolio_cache: Optional[PortfolioSignatureCache] = None

def get_cache() -> Optional[PortfolioSignatureCache]:
    """Get global cache instance"""
    return portfolio_cache

async def initialize_cache(config: CacheConfig) -> PortfolioSignatureCache:
    """Initialize global cache instance"""
    global portfolio_cache
    portfolio_cache = PortfolioSignatureCache(config)
    await portfolio_cache.initialize()
    return portfolio_cache