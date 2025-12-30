# ml_pipeline/models/trainer.py
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

class ModelTrainer:
    def __init__(self, model_dir: Path = Path("ml_pipeline/models/")):
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.best_model = None
        self.best_score = 0
        
    def train(self, X, y, test_size=0.2, random_state=42):
        """Train multiple models and select the best"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Define models
        models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'xgboost': XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
        # Hyperparameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'xgboost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        }
        
        results = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Training {model_name}...")
            
            # Grid search
            grid_search = GridSearchCV(
                model,
                param_grids[model_name],
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Evaluate
            y_pred = grid_search.predict(X_test)
            y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
            
            # Store results
            results[model_name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'test_accuracy': grid_search.score(X_test, y_test),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'model': grid_search.best_estimator_
            }
            
            # Update best model
            if grid_search.best_score_ > self.best_score:
                self.best_score = grid_search.best_score_
                self.best_model = grid_search.best_estimator_
                self.best_model_name = model_name
        
        # Save best model
        self._save_best_model()
        
        # Save results
        self._save_results(results, X_test, y_test)
        
        return results
    
    def _save_best_model(self):
        """Save the best model to disk"""
        import joblib
        model_path = self.model_dir / "best_model.pkl"
        joblib.dump(self.best_model, model_path)
        self.logger.info(f"Best model saved to {model_path}")
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'best_score': float(self.best_score),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(self.model_dir / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _save_results(self, results, X_test, y_test):
        """Save training results"""
        # Save feature importance for tree-based models
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = dict(zip(
                range(len(self.best_model.feature_importances_)),
                self.best_model.feature_importances_.tolist()
            ))
            
            with open(self.model_dir / "feature_importance.json", "w") as f:
                json.dump(feature_importance, f, indent=2)