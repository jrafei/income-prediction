import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Générer des données factices
np.random.seed(42)
X = np.random.rand(100, 1)
y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.5

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Liste des modèles à entraîner et leurs paramètres
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1)
}

# Entraîner chaque modèle et enregistrer les résultats avec MLflow
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Entraîner le modèle
        model.fit(X_train, y_train)
        
        # Faire des prédictions
        predictions = model.predict(X_test)
        
        # Calculer l'erreur quadratique moyenne
        mse = mean_squared_error(y_test, predictions)
        
        # Enregistrer les paramètres et les métriques dans MLflow
        mlflow.log_param("model_type", model_name)
        if hasattr(model, "alpha"):
            mlflow.log_param("alpha", model.alpha)
        mlflow.log_metric("mse", mse)
        
        # Enregistrer le modèle
        mlflow.sklearn.log_model(model, "model")

        print(f"{model_name} enregistré avec MSE : {mse}")

print("Tous les modèles ont été enregistrés avec leurs métriques.")
