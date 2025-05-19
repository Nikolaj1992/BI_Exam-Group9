from sklearn import metrics

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    mae = metrics.mean_absolute_error(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    rmse = metrics.mean_squared_error(y_test, predictions, squared=False)
    r2 = metrics.r2_score(y_test, predictions)

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'predictions': predictions
    }