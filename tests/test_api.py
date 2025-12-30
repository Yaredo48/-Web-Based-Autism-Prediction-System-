# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "version" in response.json()

def test_predict_endpoint():
    # Valid request
    valid_data = {
        "age": 25,
        "gender": "m",
        "ethnicity": "White",
        "jaundice": False,
        "autism_history": True,
        "assessment_scores": {
            "A1": 1, "A2": 0, "A3": 1, "A4": 0, "A5": 1,
            "A6": 0, "A7": 1, "A8": 0, "A9": 1, "A10": 0
        }
    }
    
    response = client.post("/predict/", json=valid_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence" in response.json()
    
    # Invalid request - missing field
    invalid_data = valid_data.copy()
    del invalid_data["age"]
    
    response = client.post("/predict/", json=invalid_data)
    assert response.status_code == 422  # Validation error
    
    # Invalid request - wrong data type
    invalid_data = valid_data.copy()
    invalid_data["age"] = "twenty-five"
    
    response = client.post("/predict/", json=invalid_data)
    assert response.status_code == 422

# tests/test_ml_pipeline.py
import numpy as np
import pandas as pd
from ml_pipeline.features.preprocessing import AutismPreprocessor
from ml_pipeline.models.trainer import ModelTrainer

def test_preprocessor():
    """Test data preprocessing"""
    # Create sample data
    X = pd.DataFrame({
        'age': [25, 30, 35],
        'gender': ['m', 'f', 'm'],
        'ethnicity': ['White', 'Asian', 'Middle Eastern'],
        'jaundice': [True, False, True],
        'autism_history': [False, True, False],
        'score_A1': [1, 0, 1],
        'score_A2': [0, 1, 0],
        'score_A3': [1, 0, 1],
        'score_A4': [0, 1, 0],
        'score_A5': [1, 0, 1],
        'score_A6': [0, 1, 0],
        'score_A7': [1, 0, 1],
        'score_A8': [0, 1, 0],
        'score_A9': [1, 0, 1],
        'score_A10': [0, 1, 0]
    })
    
    y = np.array([1, 0, 1])
    
    # Test preprocessing
    preprocessor = AutismPreprocessor()
    X_processed = preprocessor.fit_transform(X)
    
    assert X_processed.shape[0] == 3  # Same number of samples
    assert X_processed.shape[1] > 5   # More features after encoding
    
def test_model_training():
    """Test model training with synthetic data"""
    # Create synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 15)
    y = np.random.randint(0, 2, 100)
    
    # Test trainer
    trainer = ModelTrainer()
    results = trainer.train(X, y)
    
    assert len(results) > 0
    assert 'random_forest' in results
    assert 'xgboost' in results
    assert results['random_forest']['best_score'] > 0

# Generate API documentation
# backend/docs.py
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Autism Prediction System API",
        version="1.0.0",
        description="""
        # Autism Prediction System API
        
        ## Overview
        This API provides autism prediction based on behavioral assessment scores.
        
        ## Important Notes
        - This is a screening tool, not a diagnostic tool
        - Always consult with healthcare professionals
        - Prediction confidence below 60% should be treated with caution
        
        ## Authentication
        Currently open for development. Will implement JWT tokens in production.
        
        ## Rate Limiting
        100 requests per minute per IP address.
        
        ## Error Codes
        - 400: Bad Request
        - 422: Validation Error
        - 429: Too Many Requests
        - 500: Internal Server Error
        """,
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi