import statsmodels.api as sm

def entrenar_lineal(X_train, y_train):
    X_train_const = sm.add_constant(X_train)
    modelo = sm.OLS(y_train, X_train_const).fit()
    return modelo