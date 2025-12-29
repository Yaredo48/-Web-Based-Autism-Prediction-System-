from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

class AutismPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_features = ['age', 'score_A1', 'score_A2', 'score_A3', 
                                'score_A4', 'score_A5', 'score_A6', 'score_A7',
                                'score_A8', 'score_A9', 'score_A10']
        self.categorical_features = ['gender', 'ethnicity', 'jaundice', 'autism_history']
        
        # Define preprocessing pipelines
        self.numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        self.categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        
        self.preprocessor = ColumnTransformer([
            ('num', self.numeric_pipeline, self.numeric_features),
            ('cat', self.categorical_pipeline, self.categorical_features)
        ])
        
        self.feature_names=[]
        
        def fit(self, X, y=None):
            X_processed=self.preprocessor.fit_transform(X)
            # Get feature names after OneHotEncoding
            cat_feature_names = self.preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(self.categorical_features)
            self.feature_names = self.numeric_features + list(cat_feature_names)
            return self
        
        def transform(self, X):
            return self.preprocessor.transform(X)
        
        def _get_feature_names(self, X):
            """Extract feature names after preprocessing"""
            # Numeric features
            self.feature_names = self.numeric_features.copy()
            # Categorical features
            for col in self.categorical_features:
                unique_vals = X[col].dropna().unique()
                for val in sorted(unique_vals):
                    self.feature_names.append(f"{col}_{val}")
            return self.feature_names
        
        def save(self, path):
            '''Save preprocessor'''
            import joblib
            joblib.dump(self, path)
        
        @classmethod
        def load(cls, path):
            '''Load preprocessor'''
            import joblib
            return joblib.load(path)
        
        