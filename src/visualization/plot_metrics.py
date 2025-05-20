import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def plot_metric_bars(resultados, nombre_experimento):
    modelos = []
    maes = []
    rmses = []
    r2s = []
    mae_deviations = []  # Para almacenar la desviación de los errores absolutos (MAE)
    rmse_deviations = []  # Para almacenar la desviación de los errores cuadrados (RMSE)

    for nombre, y_train, y_train_pred, y_test, y_test_pred in resultados:
        modelos.append(nombre)
        
        # Calcular métricas
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = root_mean_squared_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

        # Calcular desviaciones para MAE y RMSE
        # Para MAE: desviación estándar de los errores absolutos
        errors_abs = np.abs(y_test - y_test_pred)
        mae_deviation = np.std(errors_abs) / np.sqrt(len(errors_abs))  # Error estándar
        mae_deviations.append(mae_deviation)

        # Para RMSE: desviación estándar de los errores cuadrados
        errors_squared = (y_test - y_test_pred) ** 2
        rmse_deviation = np.std(errors_squared) / np.sqrt(len(errors_squared))  # Error estándar
        rmse_deviations.append(rmse_deviation)

    x = np.arange(len(modelos))
    width = 0.25

    plt.figure(figsize=(12, 6))
    
    # Barras para MAE con muescas de desviación
    plt.bar(x - width, maes, width, label='MAE', color='blue')
    plt.errorbar(x - width, maes, yerr=mae_deviations, fmt='none', ecolor='black', capsize=5, capthick=2)

    # Barras para RMSE con muescas de desviación
    plt.bar(x, rmses, width, label='RMSE', color='orange')
    plt.errorbar(x, rmses, yerr=rmse_deviations, fmt='none', ecolor='black', capsize=5, capthick=2)

    # Barras para R² (sin muescas, ya que no es un error directo)
    plt.bar(x + width, r2s, width, label='R²', color='green')

    plt.ylabel("Metric Value")
    plt.title(f"Métricas del experimento: {nombre_experimento}")
    plt.xticks(x, modelos, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    filepath = f"plots/{nombre_experimento}_metrics.png"
    plt.savefig(filepath)
    print(f"Gráfica guardada en {filepath}")
    plt.close()

def plot_feature_importances(importancias, modelo_nombre, top_n=20):
    """
    Genera y guarda un gráfico de barras con las variables más importantes.
    `importancias` puede ser un dict, una Series de pandas o una lista de tuplas.
    """
    os.makedirs("plots", exist_ok=True)

    # Convertir a Series si es necesario
    if isinstance(importancias, dict):
        importancias = pd.Series(importancias)
    elif isinstance(importancias, list):
        importancias = pd.Series(dict(importancias))

    importancias = importancias.sort_values(ascending=False)
    top_features = importancias.head(top_n)

    plt.figure(figsize=(10, 6))
    top_features.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Importancia de Variables - {modelo_nombre}")
    plt.xlabel("Importancia")
    plt.tight_layout()
    path = f"plots/importancia_{modelo_nombre}.png"
    plt.savefig(path)
    plt.close()
    print(f"Gráfico de importancia guardado en {path}")