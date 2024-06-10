from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
import sys
import os
sys.path.append(os.path.abspath('../src'))
from pipelines import get_logistic_regression_pipeline, get_random_forest_pipeline

def optimize_hyperparameters_RandSearch(model_name, X, y):
    if model_name == 'logistic_regression':
        pipeline = get_logistic_regression_pipeline()
        param_distributions = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__solver': [ 'saga'] #'liblinear
        }
    elif model_name == 'random_forest':
        pipeline = get_random_forest_pipeline()
        param_distributions = {
            'classifier__n_estimators': randint(10, 200),
            'classifier__max_depth': randint(1, 20)
        }
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=10, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
    random_search.fit(X, y)
    
    print(f"Best parameters for {model_name}: {random_search.best_params_}")
    print(f"Best score for {model_name}: {random_search.best_score_}")

    return random_search.best_estimator_


def optimize_hyperparameters_GridSearch(model_name, X, y, categorical_features):
    if model_name == 'logistic_regression':
        pipeline = get_logistic_regression_pipeline(categorical_features)
        param_distributions = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__solver': [ 'saga'] #'liblinear
        }
    elif model_name == 'random_forest':
        pipeline = get_random_forest_pipeline()
        param_distributions = {
            'classifier__n_estimators': randint(10, 200),
            'classifier__max_depth': randint(1, 20)
        }
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    # Configurer GridSearchCV avec la métrique F1 score
    grid = GridSearchCV(pipeline, param_distributions, verbose=False, n_jobs=1, return_train_score=True, scoring='f1')

    # Entraîner le modèle avec GridSearchCV
    grid.fit(X, y)
    
    print(f"Best parameters for {model_name}: {grid.best_params_}")
    print(f"Best score for {model_name}: {grid.best_score_}")

    return grid.best_estimator_

