# backend/database/crud.py
from sqlalchemy.orm import Session
from . import models, schemas
from typing import List, Optional
import uuid
from datetime import datetime

def create_prediction(
    db: Session,
    session_id: Optional[str],
    input_data: dict,
    prediction_result: dict
) -> models.Prediction:
    """
    Create a new prediction record
    """
    if not session_id:
        session_id = str(uuid.uuid4())
    
    db_prediction = models.Prediction(
        session_id=session_id,
        age=input_data.get('age'),
        gender=input_data.get('gender'),
        ethnicity=input_data.get('ethnicity'),
        jaundice=input_data.get('jaundice'),
        autism_history=input_data.get('autism_history'),
        assessment_score={
            f'A{i}': input_data.get(f'score_A{i}', 0)
            for i in range(1, 11)
        },
        prediction_result=prediction_result.get('prediction'),
        confidence_score=prediction_result.get('confidence'),
        model_version=prediction_result.get('model_version')
    )
    
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    
    return db_prediction

def get_predictions_by_session(
    db: Session,
    session_id: str,
    limit: int = 10
) -> List[models.Prediction]:
    """
    Get predictions for a session
    """
    return db.query(models.Prediction)\
        .filter(models.Prediction.session_id == session_id)\
        .order_by(models.Prediction.created_at.desc())\
        .limit(limit)\
        .all()

def create_model_performance(
    db: Session,
    model_version: str,
    accuracy: float,
    precision: float,
    recall: float,
    f1_score: float,
    confusion_matrix: dict
) -> models.ModelPerformance:
    """
    Record model performance metrics
    """
    db_performance = models.ModelPerformance(
        model_version=model_version,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        confusion_matrix=confusion_matrix
    )
    
    db.add(db_performance)
    db.commit()
    db.refresh(db_performance)
    
    return db_performance

def get_latest_model_performance(
    db: Session
) -> Optional[models.ModelPerformance]:
    """
    Get the latest model performance metrics
    """
    return db.query(models.ModelPerformance)\
        .order_by(models.ModelPerformance.training_date.desc())\
        .first()
        
        
    
        
    