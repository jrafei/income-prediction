from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def create_pipeline(model, categorical_features):

   # Imputation et Normalisation
    catgoricalPipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
    )

    preprocessor = make_column_transformer(
        (catgoricalPipeline, categorical_features)
    )

    # Création du pipeline final incluant le prétraitement et le modèle
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return pipeline

def get_logistic_regression_pipeline(categorical_features):
    model = LogisticRegression()
    return create_pipeline(model, categorical_features)

def get_random_forest_pipeline( categorical_features):
    model = RandomForestClassifier()
    return create_pipeline(model, categorical_features)
