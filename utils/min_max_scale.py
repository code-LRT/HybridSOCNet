import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def get_scaler_from_train(train_file):
    train_data = pd.read_excel(train_file).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    return scaler

def min_max_scale(data, scaler):
    return scaler.transform(data)

def inverse_transform(predictions, true_values, scaler, label_index=-1):

    predictions = predictions.reshape(-1, 1)
    true_values = true_values.reshape(-1, 1)

    dummy_input = np.zeros((len(predictions), scaler.n_features_in_))
    dummy_input[:, label_index] = predictions[:, 0]
    pred_rescaled = scaler.inverse_transform(dummy_input)[:, label_index]

    dummy_input[:, label_index] = true_values[:, 0]
    true_rescaled = scaler.inverse_transform(dummy_input)[:, label_index]

    return true_rescaled, pred_rescaled