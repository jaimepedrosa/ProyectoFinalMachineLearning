import os
import sys
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from src.models.random_forest import entrenar_rf
from src.evaluation.evaluate import evaluar_modelo
from sklearn.model_selection import train_test_split
from src.models.linear_regression import entrenar_lineal
from src.models.utils import entrenar_mlp, predecir_mlp, interpretar_mlp
from src.preprocessing.transform import preprocesar_para_modelo_i, cargar_datos
from src.evaluation.interpret import importancia_lineal, importancia_random_forest
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from src.visualization.plot_metrics import plot_metric_bars, plot_feature_importances

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def seleccionar_variables_importantes(importancias, threshold=0.01):
    return [var for var, imp in importancias.items() if abs(imp) > threshold and var != "const"]

# Crear directorio 'resultados/' al inicio del script
os.makedirs("resultados", exist_ok=True)

# 1. Cargar y preprocesar datos
print("\n--- Experimento: MODELO I (con T1 y T2) ---")
df = cargar_datos("data/rendimiento_estudiantes_train.csv")
X, y = preprocesar_para_modelo_i(df)

X_cols = X.columns if isinstance(X, pd.DataFrame) else [f"var{i}" for i in range(X.shape[1])]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

resultados = []

# 2. Regresión Lineal
modelo_lr = entrenar_lineal(X_train, y_train)
y_train_pred_lr = modelo_lr.predict(sm.add_constant(X_train))
y_test_pred_lr = modelo_lr.predict(sm.add_constant(X_test))
evaluar_modelo(y_train, y_train_pred_lr, y_test, y_test_pred_lr, "Linear Regression")
resultados.append(("Linear Regression", y_train, y_train_pred_lr, y_test, y_test_pred_lr))

print("\nImportancia variables (LR):")
importancia_lr = importancia_lineal(modelo_lr, X_train.columns)
print(importancia_lr)
plot_feature_importances(importancia_lr, "Linear_Regression_I")

cols_lr = seleccionar_variables_importantes(importancia_lr.to_dict())
X_train_lr = X_train[cols_lr]
X_test_lr = X_test[cols_lr]
modelo_lr_f = entrenar_lineal(X_train_lr, y_train)
y_train_pred_lr_f = modelo_lr_f.predict(sm.add_constant(X_train_lr))
y_test_pred_lr_f = modelo_lr_f.predict(sm.add_constant(X_test_lr))
evaluar_modelo(y_train, y_train_pred_lr_f, y_test, y_test_pred_lr_f, "Linear Regression (filtrado)")
resultados.append(("Linear Regression (filtrado)", y_train, y_train_pred_lr_f, y_test, y_test_pred_lr_f))

importancia_lr_f = importancia_lineal(modelo_lr_f, X_train_lr.columns)
print("\nImportancia variables (LR filtrado):")
print(importancia_lr_f)
plot_feature_importances(importancia_lr_f, "Linear_Regression_I_FILTRADO")
pd.Series(cols_lr).to_csv("resultados/variables_lr_i_filtrado.csv", index=False)

# 3. Random Forest
modelo_rf = entrenar_rf(X_train, y_train)
y_train_pred_rf = modelo_rf.predict(X_train)
y_test_pred_rf = modelo_rf.predict(X_test)
evaluar_modelo(y_train, y_train_pred_rf, y_test, y_test_pred_rf, "Random Forest")
resultados.append(("Random Forest", y_train, y_train_pred_rf, y_test, y_test_pred_rf))

print("\nImportancia variables (RF):")
importancia_rf = importancia_random_forest(modelo_rf, X_cols)
print(importancia_rf)
plot_feature_importances(importancia_rf, "Random_Forest_I")

cols_rf = seleccionar_variables_importantes(importancia_rf)
X_train_rf = X_train[cols_rf]
X_test_rf = X_test[cols_rf]
modelo_rf_f = entrenar_rf(X_train_rf, y_train)
y_train_pred_rf_f = modelo_rf_f.predict(X_train_rf)
y_test_pred_rf_f = modelo_rf_f.predict(X_test_rf)
evaluar_modelo(y_train, y_train_pred_rf_f, y_test, y_test_pred_rf_f, "Random Forest (filtrado)")
resultados.append(("Random Forest (filtrado)", y_train, y_train_pred_rf_f, y_test, y_test_pred_rf_f))

importancia_rf_f = importancia_random_forest(modelo_rf_f, X_train_rf.columns)
print("\nImportancia variables (RF filtrado):")
print(importancia_rf_f)
plot_feature_importances(importancia_rf_f, "Random_Forest_I_FILTRADO")
pd.Series(cols_rf).to_csv("resultados/variables_rf_i_filtrado.csv", index=False)

# 4. MLP
modelo_mlp = entrenar_mlp(X_train, y_train, X_test, y_test, input_dim=X_train.shape[1])
y_train_pred_mlp = predecir_mlp(modelo_mlp, X_train)
y_test_pred_mlp = predecir_mlp(modelo_mlp, X_test)
evaluar_modelo(y_train, y_train_pred_mlp, y_test, y_test_pred_mlp, "MLP")
resultados.append(("MLP", y_train, y_train_pred_mlp, y_test, y_test_pred_mlp))

print("\nImportancia variables (MLP):")
importancia_mlp = interpretar_mlp(modelo_mlp, X_train, y_train, X_cols)
print(importancia_mlp)
plot_feature_importances(importancia_mlp, "MLP_I")

cols_mlp = seleccionar_variables_importantes(importancia_mlp.to_dict())
X_train_mlp = X_train[cols_mlp]
X_test_mlp = X_test[cols_mlp]
modelo_mlp_f = entrenar_mlp(X_train_mlp, y_train, X_test_mlp, y_test, input_dim=X_train_mlp.shape[1])
y_train_pred_mlp_f = predecir_mlp(modelo_mlp_f, X_train_mlp)
y_test_pred_mlp_f = predecir_mlp(modelo_mlp_f, X_test_mlp)
evaluar_modelo(y_train, y_train_pred_mlp_f, y_test, y_test_pred_mlp_f, "MLP (filtrado)")
resultados.append(("MLP (filtrado)", y_train, y_train_pred_mlp_f, y_test, y_test_pred_mlp_f))

if len(X_train_mlp.columns) == 0:
    print("No se han seleccionado variables importantes para MLP filtrado.")
else:
    importancia_mlp_f = interpretar_mlp(modelo_mlp_f, X_train_mlp, y_train, X_train_mlp.columns)
    print("\nImportancia variables (MLP filtrado):")
    print(importancia_mlp_f)
    plot_feature_importances(importancia_mlp_f, "MLP_I_FILTRADO")
    importancia_mlp_f.to_csv("resultados/importancia_mlp_i_filtrado.csv")
    pd.Series(cols_mlp).to_csv("resultados/variables_mlp_i_filtrado.csv", index=False)

# 5. Visualización de métricas
plot_metric_bars(resultados, "Experimento_I")

# 6. Exportar resultados a CSV
metricas = []
for nombre, y_train, y_train_pred, y_test, y_test_pred in resultados:
    metricas.append({
        "modelo": nombre,
        "mae_test": mean_absolute_error(y_test, y_test_pred),
        "rmse_test": root_mean_squared_error(y_test, y_test_pred),
        "r2_test": r2_score(y_test, y_test_pred),
        "mae_train": mean_absolute_error(y_train, y_train_pred),
        "rmse_train": root_mean_squared_error(y_train, y_train_pred),
        "r2_train": r2_score(y_train, y_train_pred)
    })
pd.DataFrame(metricas).to_csv("resultados/metricas_experimento_I.csv", index=False)
print("Métricas guardadas en resultados/metricas_experimento_I.csv")

importancia_lr.to_csv("resultados/importancia_lr_i.csv")
importancia_lr_f.to_csv("resultados/importancia_lr_i_filtrado.csv")
pd.Series(importancia_rf).to_csv("resultados/importancia_rf_i.csv")
pd.Series(importancia_rf_f).to_csv("resultados/importancia_rf_i_filtrado.csv")
importancia_mlp.to_csv("resultados/importancia_mlp_i.csv")
print("Importancias guardadas en carpeta resultados/")