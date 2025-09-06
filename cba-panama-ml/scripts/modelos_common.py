import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor, DummyClassifier

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

from sklearn.linear_model import Ridge, LogisticRegression

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error
)

import joblib
import os

# *************************************************************************************



BASE_MODEL_PATH = '../models'


def get_dataframe():
    df = pd.read_csv('../data/processed/datasets_merged_cleaned.csv')
    df['fecha'] = pd.to_datetime(df['anio'].astype(str) + '-' + df['mes'].astype(str), format='%Y-%m')
    return df

# Guarda en disco el modelo en formato joblib para su posterior uso.
def guardar_modelo(modelo, filename):
    path = os.path.join(BASE_MODEL_PATH, filename)
    
    try: os.makedirs(os.path.dirname(path))
    except: pass

    joblib.dump(modelo, path)

# carga un modelo desde disco
def cargar_modelo(filename):
    path = os.path.join(BASE_MODEL_PATH, filename)
    
    if os.path.exists(path): return joblib.load(path)
    return None

def get_ficticio_siguiente_mes(df):
    ultimo_mes = df.fecha.max()
    siguiente_mes = ultimo_mes + pd.DateOffset(months=1)
    
    ficticio_df = df[df.fecha == ultimo_mes].copy()
    ficticio_df.anio = siguiente_mes.year
    ficticio_df.mes = siguiente_mes.month

    ficticio_df['fecha'] = pd.to_datetime(ficticio_df['anio'].astype(str) + '-' + ficticio_df['mes'].astype(str), format='%Y-%m')
    return ficticio_df

# *************************************************************************************
# REGRESION

# Funciones utilities de calculo de errores.
calcular_rmse = lambda y_test, y_pred: np.sqrt(root_mean_squared_error(y_test, y_pred))
calcular_mae = lambda y_test, y_pred: mean_absolute_error(y_test, y_pred)
calcular_wmape = lambda y_test, y_pred: np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test))

# Listado de todos los modelos de regresión que se utilizarán para el análisis.
ALL_REGRESION = {
    'Baseline': DummyRegressor(strategy='mean'),
    'Ridge': Ridge(alpha=1.0, max_iter=10000),
    'RandomForest': RandomForestRegressor(random_state=42),
    'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42),
}

# Iterador, genera una tupla (cadena, X, y).
def iter_regresion_Xy(df):
    unique_cadenas = set([(a,b) for a,b in df[['cadena_id', 'cadena']].values])
    
    # Las regresiones se harán en grupos separados por cadena de supermercados, ya que se quiere evitar
    # la influencia de los precios de una cadena sobre los de otra. Usualmente una misma cadena oscila sus
    # precios de forma regular.
    for cadena_id,cadena_name in unique_cadenas:
        sub_df = df[df.cadena_id == cadena_id]
        
        # Ya fueron codificadas previamente las variables categóricas, por lo que se utilizan sus ids directamente.
        X = sub_df[['supermercado_id', 'cadena_id', 'producto_id', 'anio', 'mes']]
        y = sub_df['costo']
        
        yield cadena_name, X, y, sub_df

# *************************************************************************************
# CLASIFICACION

# Se enlistan los modelos de clasificación que se utilizarán en el análisis.
ALL_CLASIFICACION = {
    'Baseline': DummyClassifier(strategy="stratified", random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(random_state=42),
    'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42)
}

# Similar a la regresión, se crea un iterador que genera una tupla (cadena, X, y).
def iter_clasificacion_Xy(df, umbral_pct=0.05):
    unique_cadenas = set([(a,b) for a,b in df[['cadena_id', 'cadena']].values])
    
    for cadena_id, cadena_name in unique_cadenas:
        sub_df = df[df.cadena_id == cadena_id][['supermercado_id', 'cadena_id', 'producto_id', 'costo', 'anio', 'mes', 'fecha']]
        
        # Ordenar temporalmente
        sub_df = sub_df.sort_values(by=['fecha', 'producto_id', 'supermercado_id'])

        # Calcular variación porcentual de precio por producto
        # pct_change = (precio_actual - precio_anterior) / precio_anterior
        sub_df['pct_change'] = sub_df['costo'].pct_change()

        # Target: 1 si aumento > umbral, 0 en caso contrario
        sub_df['target'] = (sub_df['pct_change'] > umbral_pct).astype(int)
    
        X = sub_df[['supermercado_id', 'cadena_id', 'producto_id', 'anio', 'mes']]
        y = sub_df['target']

        yield cadena_name, X, y, sub_df
