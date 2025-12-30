# backend/api/endpoints/predict.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import logging

from ml_pipeline.models.predictor import Predictor
from backend.database.database import SessionLocal, get_db
from backend.database import crud, models, schemas

router = APIRouter(prefix="/predict", tags=["prediction"])
logger = logging.getLogger(__name__)

# Initialize predictor
predictor = Predictor.load()

# Request/Response schemas
class PredictionRequest(BaseModel):
    age: int = Field(..., ge=2, le=100, description="Age of the individual")
    gender: str = Field(..., regex="^(m|f)$", description="Gender: 'm' or 'f'")
    ethnicity: str = Field(..., description="Ethnicity")
    jaundice: bool = Field(..., description="Born with jaundice")
    autism_history: bool = Field(..., description="Family history of autism")
    assessment_scores: Dict[str, int] = Field(
        ...,
        description="Assessment scores A1-A10 (0 or 1)",
        example={"A1": 1, "A2": 0, "A3": 1, "A4": 0, "A5": 1,
                "A6": 0, "A7": 1, "A8": 0, "A9": 1, "A10": 0}
    )
    
    class Config:
        schema_extra = {
            "example": {
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
        }

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted class")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    explanation: Optional[Dict[str, Any]] = Field(None, description="Model explanation")
    warning: Optional[str] = Field(None, description="Any warnings")
    model_version: str = Field(..., description="Model version used")

@router.post("/", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    session_id: Optional[str] = None,
    db: SessionLocal = Depends(get_db)
):
    """
    Make autism prediction based on input features
    """
    try:
        # Prepare features for prediction
        features = request.dict()
        
        # Convert assessment scores to individual features
        assessment_scores = features.pop('assessment_scores')
        for i in range(1, 11):
            features[f'score_A{i}'] = assessment_scores.get(f'A{i}', 0)
        
        # Make prediction
        prediction_result = predictor.predict(features)
        
        # Log prediction to database
        prediction_record = crud.create_prediction(
            db=db,
            session_id=session_id,
            input_data=features,
            prediction_result=prediction_result
        )
        
        # Add prediction ID to response
        prediction_result["prediction_id"] = prediction_record.id
        
        return prediction_result
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/history")
async def get_prediction_history(
    session_id: str,
    limit: int = 10,
    db: SessionLocal = Depends(get_db)
):
    """
    Get prediction history for a session
    """
    predictions = crud.get_predictions_by_session(db, session_id, limit)
    return predictions

@router.get("/model-info")
async def get_model_info():
    """
    Get information about the current model
    """
    return predictor.get_model_info()