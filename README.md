# Proyecto de Predicción de Rendimiento Académico Estudiantil

Este proyecto utiliza técnicas de aprendizaje automático para predecir el rendimiento académico de estudiantes de secundaria, específicamente la calificación final del tercer trimestre (`T3`). Se exploran dos escenarios: Experimento I (incluye calificaciones previas `T1` y `T2`) y Experimento II (excluye `T1` y `T2`). Los modelos implementados incluyen Regresión Lineal, Random Forest y una Red Neuronal Multicapa (MLP).

## Estructura del Proyecto

- `data/`: Contiene el dataset (`rendimiento_estudiantes_train.csv`).
- `notebooks/`: Notebooks de Jupyter para análisis exploratorio (si los hay).
- `plots/`: Directorio donde se guardan los gráficos generados (métricas e importancia de variables).
- `resultados/`: Directorio donde se guardan los resultados (métricas e importancias en formato CSV).
- `src/`: Código fuente del proyecto.
  - `experiments/`: Scripts para ejecutar los experimentos (`run_experiment_i.py` y `run_experiment_ii.py`).
  - `models/`: Implementación de los modelos (`linear_regression.py`, `random_forest.py`, `utils.py` para MLP).
  - `preprocessing/`: Preprocesamiento de datos (`transform.py`).
  - `evaluation/`: Evaluación e interpretación de modelos (`evaluate.py`, `interpret.py`).
  - `visualization/`: Generación de gráficos (`plot_metrics.py`).
- `main.py`: Script principal para ejecutar los experimentos.
- `predict_random_forest.py`: Script para realizar predicciones usando un modelo Random Forest entrenado.
- `predicciones_finales.csv`: Archivo CSV con predicciones generadas (si se usa `predict_random_forest.py`).
- `resultados.txt`: Archivo de texto con resultados (si se genera manualmente).
- `JaimePedrosa_informePr...`: Informe del proyecto (PDF o LaTeX).

## Dependencias

Para ejecutar este proyecto, necesitas tener instalado Python 3.10 o superior y las siguientes bibliotecas. Asegúrate de tener `pip` instalado para instalar las dependencias.

### Requisitos
- Python 3.10+
- Bibliotecas Python:
  - `pandas` (para manipulación de datos)
  - `numpy` (para cálculos numéricos)
  - `matplotlib` (para visualización de gráficos)
  - `scikit-learn` (para métricas, preprocesamiento y Random Forest)
  - `statsmodels` (para Regresión Lineal)
  - `torch` (para la implementación del MLP)
  - `joblib` (para guardar y cargar modelos entrenados)

  
### Ejecución del Proyecto
1. Realizar experimentos: Abrir una terminal y ejecutar:
    python3 main.py

2. Predecir Datos Nuevos:
    python3 predict_random_forest.py