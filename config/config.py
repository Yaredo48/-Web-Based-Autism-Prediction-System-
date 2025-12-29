import os
from pathlib import Path
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Project
    PROJECT_NAME: str = "Autism Prediction System"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "A web-based system for predicting autism spectrum disorder using machine learning."
    DEBUG: bool = os.getenv("DEBUG", False)
    
    # API
    API_V1_STR: str = "/api/v1"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://user:password@localhost:5432/autism_db"
    )
    
    # ML Model
    MODEL_PATH: Path = Path("ml_pipeline/models/best_model.pkl")
    FEATURES_PATH: Path = Path("ml_pipeline/features/feature_list.json")
    
    class Config:
        env_file = ".env"

settings = Settings()

# .env file
DEBUG=True
DATABASE_URL=postgresql://user:password@localhost:5432/autism_dev
