import pandas as pd

def importancia_lineal(modelo, X_cols):
    coef = modelo.params
    return coef.sort_values(ascending=False)

def importancia_random_forest(modelo, X_cols):
    importances = modelo.feature_importances_
    return pd.Series(importances, index=X_cols).sort_values(ascending=False)