"""
model.py : Contient des fonctions pour entraîner différents modèles de machine learning et les sauvegarder.
Fonctions : train_random_forest(), train_logistic_regression(), train_svm(), save_model(), load_model()
"""

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pipelines import get_logistic_regression_pipeline, get_random_forest_pipeline

def train_model(pipeline, X, y, model_name):
    # Division des données en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraînement du modèle
    pipeline.fit(X_train, y_train)
    
    # Prédictions sur les données de validation
    y_pred = pipeline.predict(X_val)
    
    # Évaluation du modèle
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Accuracy for {model_name}: {accuracy}")
    
    # Sauvegarde du modèle entraîné
    joblib.dump(pipeline, f'../models/{model_name}/model.pkl')
    
    return pipeline, accuracy

def train_logistic_regression(X, y):
    pipeline = get_logistic_regression_pipeline()
    return train_model(pipeline, X, y, 'logistic_regression')

def train_random_forest(X, y):
    pipeline = get_random_forest_pipeline()
    return train_model(pipeline, X, y, 'random_forest')