"""
    feature_engineering.py : Contient des fonctions pour l'ingénierie des fonctionnalités, comme le scaling et l'encodage des variables catégorielles.
Fonctions : create_feature_pipeline(), transform_features()


    Returns:
        _type_: _description_
"""
# exemple 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

def create_feature_pipeline(numeric_features, categorical_features):
    """Create a preprocessing pipeline for feature engineering."""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def transform_features(preprocessor, X):
    """Apply the preprocessing pipeline to the features."""
    return preprocessor.fit_transform(X)
