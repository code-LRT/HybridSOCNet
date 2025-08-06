import numpy as np

def evaluate(y_true, y_pred):
    error = y_true - y_pred
    mae = np.mean(np.abs(error))
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    maxe = np.max(np.abs(error))
    r2 = 1 - (np.sum(error ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return rmse,mae, maxe, r2