import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def fix_target(data) :
    """
    Corrige la variable cible 'income' pour qu'elle prenne des valeurs binaires.
    
    Args:
        df (pd.DataFrame): DataFrame initiale contenant la variable cible 'income'.

    Returns:
        pd.DataFrame: DataFrame avec la variable cible corrigée.
    """
    data['income'] = data['income'].str.replace('<=50K.', '<=50K')
    data['income'] = data['income'].str.replace('>50K.', '>50K')
    data['>50K'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)
    data = data.drop(columns=['income'])
    return data


def remove_inutile_column(df) :
    df = df.drop("education", axis=1)
    return df

def impute_missing_cat_values(df,cat_features, strategy):
    """
    Remarque : df est le dataframe test ou train (et pas le dataframe complet risque de fuite de données)
    """
    imput_cat = SimpleImputer(missing_values=np.nan, strategy=strategy)
    for feature in cat_features :
        df[feature] = imput_cat.fit_transform(df[feature].values.reshape(-1,1)).ravel()
    
    return df

def standardize(df_train,df_test, cont_features):
    scaleStd = StandardScaler()
    for feature in cont_features:
        # Normaliser les données d'entrainement
        df_train[feature] = scaleStd.fit_transform(df_train[feature].values.reshape(-1,1))
        df_train[feature] = df_train[feature].ravel()
        # Normaliser les données test
        df_test[feature] = scaleStd.transform(df_test[feature].values.reshape(-1,1))
        df_test[feature] = df_test[feature].ravel()
        
    return df_train, df_test

def seperate_train_test(df,random_state):
    """
    Sépare le DataFrame en un ensemble d'entraînement et un ensemble de test.
    
    Args:
        df (pd.DataFrame): Le DataFrame à diviser.
        random_state (int): La graine aléatoire pour la reproductibilité.
    
    Returns:
        tuple: Un tuple contenant les ensembles d'entraînement et de test.
    """
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_state, stratify=df['>50K'])
    return df_train, df_test


def transform_to_float(df):
    df[['age', 'capital-gain', 'capital-loss', 'hours-per-week','fnlwgt']] = df[['age', 'capital-gain', 
                                                                                 'capital-loss', 'hours-per-week',
                                                                                 'fnlwgt']].astype(float)
    return df

def get_column_min_max(df, column_name):
    """
    Renvoie les valeurs minimum et maximum d'une colonne spécifique dans un DataFrame.
    
    Parameters:
        df (pd.DataFrame): Le DataFrame pandas.
        column_name (str): Le nom de la colonne dont on veut obtenir les valeurs min et max.
    
    Returns:
        tuple: Un tuple contenant la valeur minimum et la valeur maximum de la colonne.
    """
    min_value = df[column_name].min()
    max_value = df[column_name].max()
    return min_value, max_value


def get_cont_features(df):
    cont_features = df.select_dtypes('float64').columns
    return cont_features

def get_cat_features(df):
    cat_features = df.select_dtypes('object').columns
    # ajout de education-num dans cat_features
    cat_features = cat_features.append(pd.Index(['education-num']))
    return cat_features