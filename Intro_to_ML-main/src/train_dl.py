import copy
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

class TimeSeriesDataset(Dataset):
    """Wrap feature and target arrays into a PyTorch Dataset."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    """A simple multilayer perceptron for tabular regression."""
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

class LSTMModel(nn.Module):
    """An LSTM-based regressor for tabular time-series features."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=15):
    """Train a model with early stopping and return losses."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    best_weights = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []

    print(f"Starting training for {model.__class__.__name__}...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                running_val_loss += loss.item() * X_val.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    if best_weights:
        model.load_state_dict(best_weights)
    return model, train_losses, val_losses

def run_deep_learning():
    """Train deep learning models on engineered volatility features."""
    data_path = os.path.join("data", "processed", "sp500_clean.csv")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    drop_cols = ['Target_Vol', 'Log_Return']
    feature_cols = [c for c in df.columns if c not in drop_cols and ('Lag' in c or 'Roll' in c)]

    X = df[feature_cols].values
    y = df['Target_Vol'].values.reshape(-1, 1)

    train_mask = df.index < '2015-01-01'
    val_mask = (df.index >= '2015-01-01') & (df.index < '2020-01-01')

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)

    batch_size = 64
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled)
    val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    results = {}

    mlp = MLP(input_dim)
    mlp, _, _ = train_model(mlp, train_loader, val_loader)
    results['MLP'] = mlp

    lstm = LSTMModel(input_dim)
    lstm, _, _ = train_model(lstm, train_loader, val_loader)
    results['LSTM'] = lstm

    print("\n--- Final Results (Validation Set - Unscaled) ---")

    def eval_and_plot(model, name):
        model.eval()
        with torch.no_grad():
            preds_scaled = model(torch.tensor(X_val_scaled, dtype=torch.float32)).numpy()
            preds_real = scaler_y.inverse_transform(preds_scaled)

            mse = mean_squared_error(y_val, preds_real)
            r2 = r2_score(y_val, preds_real)
            print(f"{name}: MSE={mse:.8f}, R2={r2:.6f}")
            return preds_real

    plt.figure(figsize=(15, 6))
    val_indices = df.index[val_mask]
    plot_mask = (val_indices >= '2018-01-01') & (val_indices <= '2018-12-31')

    plt.plot(val_indices[plot_mask], y_val[plot_mask], label='Actual', color='black', alpha=0.5)

    preds_mlp = eval_and_plot(results['MLP'], "MLP")
    plt.plot(val_indices[plot_mask], preds_mlp[plot_mask], label='MLP', linestyle='--')

    preds_lstm = eval_and_plot(results['LSTM'], "LSTM")
    plt.plot(val_indices[plot_mask], preds_lstm[plot_mask], label='LSTM', linestyle='--')

    plt.title('Deep Learning Models (With Target Scaling)')
    plt.legend()
    os.makedirs("plots", exist_ok=True)
    plt.savefig('plots/model_comparison_dl_scaled.png')
    print("New comparison plot saved.")

if __name__ == "__main__":
    run_deep_learning()