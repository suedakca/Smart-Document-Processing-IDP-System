from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from .db_client import DatabaseClient
from loguru import logger
import redis
import os

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)
db = DatabaseClient()

# Redis for Rate Limiting (Phase 3)
try:
    redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/1"))
except Exception as e:
    logger.warning(f"Redis connection for rate limiting failed: {str(e)}")
    redis_client = None

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Please provide X-API-KEY header."
        )
    
    key_id = db.verify_key(api_key_header)
    if not key_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or Inactive API Key."
        )
    
    # Rate Limiting Logic (Simple Fixed Window)
    if redis_client:
        limit_key = f"rate_limit:{key_id}"
        current_usage = redis_client.get(limit_key)
        
        if current_usage and int(current_usage) >= 20: # 20 requests per minute limit
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded (20 req/min). Upgrade for higher limits."
            )
        
        pipe = redis_client.pipeline()
        pipe.incr(limit_key)
        pipe.expire(limit_key, 60) # Reset every 60 seconds
        pipe.execute()

    return key_id
