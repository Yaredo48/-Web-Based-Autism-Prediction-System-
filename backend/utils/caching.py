# backend/utils/caching.py
import redis
import pickle
from functools import wraps
from typing import Any, Callable
import logging
from config.config import settings

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self):
        self.redis_client = None
        self.enabled = settings.REDIS_URL is not None
        
        if self.enabled:
            try:
                self.redis_client = redis.Redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=False
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache connected")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.enabled = False
    
    def cache_result(self, key: str, value: Any, ttl: int = 3600):
        """Cache a result"""
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            serialized = pickle.dumps(value)
            self.redis_client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def get_cached(self, key: str) -> Any:
        """Get cached result"""
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            cached = self.redis_client.get(key)
            if cached:
                return pickle.loads(cached)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    def invalidate(self, key: str):
        """Invalidate cache entry"""
        if self.enabled and self.redis_client:
            self.redis_client.delete(key)

# Cache decorator
def cache_prediction(ttl: int = 300):
    """Decorator to cache prediction results"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_manager = CacheManager()
            
            # Create cache key from input features
            features = kwargs.get('features', {})
            cache_key = f"prediction:{hash(str(sorted(features.items())))}"
            
            # Try to get from cache
            cached_result = cache_manager.get_cached(cache_key)
            if cached_result:
                cached_result['cached'] = True
                return cached_result
            
            # Compute result
            result = await func(*args, **kwargs)
            
            # Cache the result
            cache_manager.cache_result(cache_key, result, ttl)
            result['cached'] = False
            
            return result
        return wrapper
    return decorator