import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from src.models.mlp import MLP
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def entrenar_mlp(X_train, y_train, X_val, y_val, input_dim, epochs=100, batch_size=32, lr=0.001):
    # Convertir a tensores desde arrays numpy
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)

    # Modelo, optimizador, función de pérdida
    model = MLP(input_dim=input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluación
        model.eval()
        val_loss = 0
        val_preds = []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item()
                val_preds.extend(pred.numpy())

        if (epoch + 1) % 10 == 0:
            mae = mean_absolute_error(y_val, val_preds)
            rmse = root_mean_squared_error(y_val, val_preds)
            r2 = r2_score(y_val, val_preds)
            print(f"[MLP] Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

    return model


def predecir_mlp(model, X):
    model.eval()
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    with torch.no_grad():
        pred = model(X_tensor).numpy()
    return pred


def predict_fn(model, X_in):
    X_tensor = torch.tensor(X_in, dtype=torch.float32)
    with torch.no_grad():
        return model(X_tensor).numpy()


def interpretar_mlp(model, X, y, feature_names):
    mlp_sklearn = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1)
    mlp_sklearn.fit(X, y)

    result = permutation_importance(
        estimator=mlp_sklearn,
        X=X,
        y=y,
        n_repeats=10,
        random_state=42,
        scoring='neg_mean_absolute_error'
    )
    return pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)