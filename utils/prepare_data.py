import pandas as pd
import numpy as np
from min_max_scale import get_scaler_from_train, min_max_scale

def create_sequences(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step, :-1])  
        y.append(data[i+time_step - 1, -1])  
    return np.array(X), np.array(y)

def prepare_data(train_file, test_file, val_file, time_step):

    train_Data = pd.read_excel(train_file)
    test_Data = pd.read_excel(test_file)
    val_Data = pd.read_excel(val_file)

    print('Read_in_the_train_dataset', train_Data.shape)
    print('Read_in_the_test_dataset', test_Data.shape)
    print('Read_in_the_val_dataset', val_Data.shape)

    train_Data_values = train_Data.iloc[:, :].values
    test_Data_values = test_Data.iloc[:, :].values
    val_Data_values = val_Data.iloc[:, :].values

    scaler = get_scaler_from_train(train_file)

    train_Data_scaled = min_max_scale(train_Data_values, scaler)
    test_Data_scaled = min_max_scale(test_Data_values, scaler)
    val_Data_scaled = min_max_scale(val_Data_values, scaler)

    X_train, y_train = create_sequences(train_Data_scaled, time_step)
    X_test, y_test = create_sequences(test_Data_scaled, time_step)
    X_val, y_val = create_sequences(val_Data_scaled, time_step)

    print('X_train:', X_train.shape, 'y_train:', y_train.shape)
    print('X_test:', X_test.shape, 'y_test:', y_test.shape)
    print('X_val:', X_val.shape, 'y_val:', y_val.shape)

    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)
    np.save('X_val.npy', X_val)
    np.save('y_val.npy', y_val)

    return X_train, y_train, X_test, y_test, X_val, y_val