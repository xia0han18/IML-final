import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def run_backtest():
    """Run backtest on all models (Classical ML + Deep Learning)."""
    print("="*60)
    print("BACKTESTING ALL MODELS (Test Set: 2020-2025)")
    print("="*60)

    
    data_path = os.path.join("data", "processed", "sp500_ml_ready.csv")
    model_dir = os.path.join("models")
    plot_dir = os.path.join("plots")
    output_dir = os.path.join("outputs")
    
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Error: Missing data file at {data_path}")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    
    test_mask = df.index >= '2020-01-01'
    df_test = df[test_mask].copy()

    
    drop_cols = ['Target_Vol', 'Log_Return']
    feature_cols = [
        c for c in df.columns
        if c not in drop_cols and ('Lag' in c or 'Roll' in c)
    ]

    X_test = df_test[feature_cols]
    y_test = df_test['Target_Vol']

    print(f"Test Set Range: {df_test.index.min().date()} to {df_test.index.max().date()}")
    print(f"Test Samples: {len(df_test)}")

    
    rmse_rows = []
    mae_rows = []
    predictions_dict = {}

    
    classical_models = [
        ("Linear_Regression", False),
        ("Ridge_Regression", False),
        ("Random_Forest", False),
        ("XGBoost", False),
        ("KNN", True),  
    ]

    scaler_path = os.path.join(model_dir, "x_scaler.pkl")
    if os.path.exists(scaler_path):
        x_scaler = joblib.load(scaler_path)
        X_test_scaled = x_scaler.transform(X_test)
    else:
        X_test_scaled = X_test.values

    print("\n--- Classical ML Models ---")
    
    for name, use_scaled in classical_models:
        model_path = os.path.join(model_dir, f"{name}.pkl")
        
        if not os.path.exists(model_path):
            print(f"  {name}: Model not found, skipping...")
            continue
        
        model = joblib.load(model_path)
        
        if use_scaled:
            preds = model.predict(X_test_scaled)
        else:
            preds = model.predict(X_test)
        
        rmse_val = rmse(y_test, preds)
        mae_val = mae(y_test, preds)
        
        rmse_rows.append({"Model": name, "RMSE_Test": rmse_val})
        mae_rows.append({"Model": name, "MAE_Test": mae_val})
        predictions_dict[name] = preds
        
        print(f"  {name}: RMSE={rmse_val:.6f}, MAE={mae_val:.6f}")


    print("\n--- Deep Learning Models ---")
    
    dl_x_scaler_path = os.path.join(model_dir, "dl_x_scaler.pkl")
    dl_y_scaler_path = os.path.join(model_dir, "dl_y_scaler.pkl")
    
    if os.path.exists(dl_x_scaler_path) and os.path.exists(dl_y_scaler_path):
        dl_x_scaler = joblib.load(dl_x_scaler_path)
        dl_y_scaler = joblib.load(dl_y_scaler_path)
        X_test_dl = dl_x_scaler.transform(X_test.values)
    else:
        print("  DL scalers not found, skipping DL models...")
        dl_x_scaler = None

    if dl_x_scaler is not None:
        input_dim = X_test.shape[1]
        
        dl_models = [
            ("MLP", MLP(input_dim)),
            ("LSTM", LSTMModel(input_dim)),
        ]
        
        for name, model in dl_models:
            model_path = os.path.join(model_dir, f"{name}.pt")
            
            if not os.path.exists(model_path):
                print(f"  {name}: Model not found, skipping...")
                continue
            
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            with torch.no_grad():
                X_tensor = torch.tensor(X_test_dl, dtype=torch.float32)
                preds_scaled = model(X_tensor).cpu().numpy()
                preds = dl_y_scaler.inverse_transform(preds_scaled).reshape(-1)
            
            rmse_val = rmse(y_test.values, preds)
            mae_val = mae(y_test.values, preds)
            
            rmse_rows.append({"Model": name, "RMSE_Test": rmse_val})
            mae_rows.append({"Model": name, "MAE_Test": mae_val})
            predictions_dict[name] = preds
            
            print(f"  {name}: RMSE={rmse_val:.6f}, MAE={mae_val:.6f}")

    rmse_df = pd.DataFrame(rmse_rows).sort_values("RMSE_Test").reset_index(drop=True)
    mae_df = pd.DataFrame(mae_rows).sort_values("MAE_Test").reset_index(drop=True)

    print("\n" + "="*50)
    print("BACKTEST RMSE (sorted)")
    print("="*50)
    print(rmse_df.to_string(index=False))

    print("\n" + "="*50)
    print("BACKTEST MAE (sorted)")
    print("="*50)
    print(mae_df.to_string(index=False))

    rmse_df.to_csv(os.path.join(output_dir, "backtest_RMSE.csv"), index=False)
    mae_df.to_csv(os.path.join(output_dir, "backtest_MAE.csv"), index=False)
    print(f"\nSaved: {output_dir}/backtest_RMSE.csv")
    print(f"Saved: {output_dir}/backtest_MAE.csv")


    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22']
    
    for idx, (name, preds) in enumerate(predictions_dict.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        ax.plot(df_test.index, y_test, label='Actual', color='black', alpha=0.5, linewidth=0.8)
        ax.plot(df_test.index, preds, label=name, color=colors[idx % len(colors)], alpha=0.8, linewidth=0.8)
        
        # Highlight COVID crash
        ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-05-01'),
                   color='red', alpha=0.1)
        
        ax.set_title(f"{name} (RMSE={rmse(y_test, preds):.6f})")
        ax.set_ylabel('Volatility')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    for idx in range(len(predictions_dict), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Backtest: All Models vs Actual Volatility (2020-2025)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "backtest_all_models.png"), dpi=150)
    plt.close()
    print(f"Saved: {plot_dir}/backtest_all_models.png")

  
    best_model = rmse_df.iloc[0]['Model']
    best_preds = predictions_dict[best_model]
    
    plt.figure(figsize=(15, 7))
    
    plt.plot(df_test.index, y_test, label='Actual Volatility',
             color='black', alpha=0.6, linewidth=1)
    plt.plot(df_test.index, best_preds, label=f'Predicted ({best_model})',
             color='blue', alpha=0.8, linewidth=1)
    
    plt.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-05-01'),
                color='red', alpha=0.1, label='COVID Crash')
    
    plt.title(f'Best Model Backtest: {best_model} (2020-2025)')
    plt.ylabel('Volatility')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(plot_dir, "backtest_best_model.png"), dpi=150)
    plt.close()
    print(f"Saved: {plot_dir}/backtest_best_model.png")

    print(f"\nBest Model (by RMSE): {best_model}")

    return rmse_df, mae_df


if __name__ == "__main__":
    rmse_results, mae_results = run_backtest()