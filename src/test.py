import torch
import pandas as pd
import numpy as np
from model import CNN_GRU
from data.load_data import load_data
from torch.utils.data import DataLoader, TensorDataset
from utils.min_max_scale import get_scaler_from_train, inverse_transform
from evaluate import evaluate

class Config:
    window_size = 80
    batch_size = 32
    feature_size = 3
    cnngru_units = 64
    cnngru_layers = 2
    cnngru_kernel = 5
    cnn_out_channels = 8
    model_save_path = "CNN_GRU.pth"

def run_test():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, X_test, y_test, _, _ = load_data(config.window_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=config.batch_size, shuffle=False)

    model = CNN_GRU(
        config.feature_size,
        config.cnn_out_channels,
        config.cnngru_units,
        config.cnngru_layers,
        dropout=0,
        kernel_size=config.cnngru_kernel
    ).to(device)

    model.load_state_dict(torch.load(config.model_save_path, map_location=device))
    model.eval()

    preds_list = []
    y_true_list = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            output = model(xb).cpu().numpy()
            preds_list.append(output)
            y_true_list.append(yb.numpy())

    preds = np.vstack(preds_list)
    y_true = np.hstack(y_true_list)

    scaler = get_scaler_from_train()
    true_rescaled, pred_rescaled = inverse_transform(preds, y_true, scaler, label_index=-1)

    rmse, mae, maxe, r2 = evaluate(true_rescaled, pred_rescaled)
    print(f"Test Results: RMSE={rmse:.4f}, MAE={mae:.4f}, MAXE={maxe:.4f}, R2={r2:.4f}")
    df = pd.DataFrame({
        "True_SOC": true_rescaled,
        "Predicted_SOC": pred_rescaled
    })
    df.to_excel("test_predictions.xlsx", index=False)
    print("Prediction results saved to test_predictions.xlsx")