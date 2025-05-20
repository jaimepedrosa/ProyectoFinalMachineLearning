import pandas as pd
from src.models.random_forest import entrenar_rf
from src.evaluation.interpret import importancia_random_forest
from src.preprocessing.transform import preprocesar_para_modelo_i, preprocesar_para_modelo_ii, cargar_datos



def seleccionar_variables_importantes(importancias, threshold=0.01):
    return [var for var, imp in importancias.items() if abs(imp) > threshold and var != "const"]



def generar_predicciones_rf():
    # ========= CARGAR DATOS =========
    train = cargar_datos("data/rendimiento_estudiantes_train.csv")
    test = cargar_datos("data/rendimiento_estudiantes_test_vacio.csv")

    # ========= MODELO I =========
    X_i, y_i = preprocesar_para_modelo_i(train)
    X_test_i, _ = preprocesar_para_modelo_i(test)
    
    importancias_i = None
    modelo_rf_i = entrenar_rf(X_i, y_i)
    importancias_i = importancia_random_forest(modelo_rf_i, X_i.columns)
    cols_i = seleccionar_variables_importantes(importancias_i)

    modelo_rf_i_f = entrenar_rf(X_i[cols_i], y_i)
    pred_i = modelo_rf_i_f.predict(X_test_i[cols_i])

    # ========= MODELO II =========
    X_ii, y_ii = preprocesar_para_modelo_ii(train)
    X_test_ii, _ = preprocesar_para_modelo_ii(test)

    importancias_ii = None
    modelo_rf_ii = entrenar_rf(X_ii, y_ii)
    importancias_ii = importancia_random_forest(modelo_rf_ii, X_ii.columns)
    cols_ii = seleccionar_variables_importantes(importancias_ii)

    modelo_rf_ii_f = entrenar_rf(X_ii[cols_ii], y_ii)
    pred_ii = modelo_rf_ii_f.predict(X_test_ii[cols_ii])

    # ========= EXPORTAR =========
    pd.DataFrame({
        "Modelo_i": pred_i,
        "Modelo_ii": pred_ii
    }).to_csv("predicciones_finales.csv", index=False)

    print("Archivo predicciones_finales.csv generado correctamente.")


if __name__ == "__main__":
    generar_predicciones_rf()
