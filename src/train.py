import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from model import CNN_GRU
from data.load_data import load_data

class Config:
    window_size = 80
    batch_size = 32
    epochs = 1000
    learning_rate = 0.0001
    dropout = 0.2
    feature_size = 3
    cnngru_units = 64
    cnngru_layers = 2
    cnngru_kernel = 5
    cnn_out_channels = 8
    model_save_path = "CNN_GRU.pth"

def train_model():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    start_time_all = datetime.now()

    X_train, y_train, X_test, y_test, X_val, y_val = load_data(config.window_size)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config.batch_size, shuffle=False)

    model = CNN_GRU(
        config.feature_size,
        config.cnn_out_channels,
        config.cnngru_units,
        config.cnngru_layers,
        config.dropout,
        config.cnngru_kernel
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    print("Start training...")
    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb.view(-1, 1))
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{config.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.model_save_path)

    print(f"Training complete. Total time: {(datetime.now() - start_time_all).total_seconds():.2f}s")

    # 保存loss曲线
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.svg')
    plt.show()
