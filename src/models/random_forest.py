from sklearn.ensemble import RandomForestRegressor

def entrenar_rf(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf