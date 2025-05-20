import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def cargar_datos(path):
    return pd.read_csv(path)


def preprocesar_para_modelo_i(df):
    df = df.copy()
    y = df["T3"]
    X = df.drop(columns=["T3"])
    return _preprocesar_general(X), y


def preprocesar_para_modelo_ii(df):
    df = df.copy()
    df = df.drop(columns=["T1", "T2"])
    y = df["T3"]
    X = df.drop(columns=["T3"])
    return _preprocesar_general(X), y


def _preprocesar_general(df):
    df = df.copy()
    categ = df.select_dtypes(include="object").columns.tolist()
    num = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Pipelines con imputación
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    ct = ColumnTransformer([
        ("num", num_transformer, num),
        ("cat", cat_transformer, categ)
    ])

    X_array = ct.fit_transform(df)
    feature_names = (
        num +
        list(ct.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(categ))
    )
    return pd.DataFrame(X_array, columns=feature_names)