# backend/api/health.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
import redis
from typing import Dict

router = APIRouter(tags=["health"])

def check_database(db: Session) -> Dict[str, str]:
    """Check database connection"""
    try:
        db.execute(text("SELECT 1"))
        return {"status": "healthy", "message": "Database connected"}
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}

def check_redis() -> Dict[str, str]:
    """Check Redis connection"""
    try:
        r = redis.Redis.from_url(settings.REDIS_URL)
        r.ping()
        return {"status": "healthy", "message": "Redis connected"}
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}

def check_model() -> Dict[str, str]:
    """Check ML model availability"""
    try:
        from ml_pipeline.models.predictor import Predictor
        predictor = Predictor.load()
        return {
            "status": "healthy", 
            "message": f"Model {predictor.metadata.get('model_version')} loaded"
        }
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}

@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Comprehensive health check"""
    checks = {
        "api": {"status": "healthy", "message": "API is running"},
        "database": check_database(db),
        "redis": check_redis(),
        "model": check_model()
    }
    
    # Overall status
    all_healthy = all(c["status"] == "healthy" for c in checks.values())
    status_code = 200 if all_healthy else 503
    
    return {
        "status": "healthy" if all_healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks
    }

@router.get("/ready")
async def readiness_probe():
    """Readiness probe for Kubernetes"""
    # Check if all services are ready to accept traffic
    return {"status": "ready"}

@router.get("/live")
async def liveness_probe():
    """Liveness probe for Kubernetes"""
    # Check if application is alive
    return {"status": "alive"}