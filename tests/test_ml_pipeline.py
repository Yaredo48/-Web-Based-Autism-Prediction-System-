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