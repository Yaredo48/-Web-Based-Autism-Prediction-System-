# config/production.py
import os
from .config import Settings

class ProductionSettings(Settings):
    """Production settings"""
    
    # Override development settings
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    
    # Production database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://prod_user:secure_password@prod-db:5432/autism_prod"
    )
    
    # Redis cache
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    
    # CORS
    ALLOWED_ORIGINS: list = [
        "https://yourdomain.com",
        "https://app.yourdomain.com"
    ]
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Model retraining
    AUTO_RETRAIN_DAYS: int = 30
    AUTO_RETRAIN_SAMPLES: int = 1000
    
    class Config:
        env_file = ".env.production"

# Switch settings based on environment
import os
env = os.getenv("ENVIRONMENT", "development")

if env == "production":
    settings = ProductionSettings()
else:
    settings = Settings()