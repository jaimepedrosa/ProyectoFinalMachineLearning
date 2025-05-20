import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def evaluar_modelo(y_true_train, y_pred_train, y_true_test, y_pred_test, nombre):
    print(f"\n== {nombre} ==")
    for tipo, y_true, y_pred in [("Train", y_true_train, y_pred_train), ("Test", y_true_test, y_pred_test)]:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{tipo} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
