import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(name, model, X_train, y_train, X_val, y_val):
    """
    Trains a model, prints evaluation metrics, and returns predictions.
    """
    print(f"--- Training {name} ---")
    model.fit(X_train, y_train)
    
    pred_val = model.predict(X_val)

    mse_val = mean_squared_error(y_val, pred_val)
    mae_val = mean_absolute_error(y_val, pred_val)
    r2_val = r2_score(y_val, pred_val)

    print(f"Validation MSE: {mse_val:.8f}")
    print(f"Validation MAE: {mae_val:.8f}")
    print(f"Validation R²:  {r2_val:.6f}")
    print("-" * 30)

    return pred_val

def train_classical_models():
    """Train and evaluate linear, ridge, and random forest regressors."""
    input_path = os.path.join("data", "processed", "sp500_clean.csv")
    if not os.path.exists(input_path):
        print("Data not found.")
        return
    
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    drop_cols = ['Target_Vol', 'Log_Return'] 
    feature_cols = [c for c in df.columns if c not in drop_cols and ('Lag' in c or 'Roll' in c)]

    X = df[feature_cols]
    y = df['Target_Vol']

    print(f"Features used: {feature_cols}")

    train_mask = df.index < '2015-01-01'
    val_mask = (df.index >= '2015-01-01') & (df.index < '2020-01-01')

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    model_dir = "models"
    plot_dir = "plots"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    results = {}
    models_to_train = [
        ("Linear_Regression", LinearRegression()),
        ("Ridge_Regression", Ridge(alpha=1.0)),
        ("Random_Forest", RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42))
    ]

    for name, model in models_to_train:
        results[name] = evaluate_model(name.replace("_", " "), model, X_train, y_train, X_val, y_val)

        save_path = os.path.join(model_dir, f"{name}.pkl")
        joblib.dump(model, save_path)
        print(f"Saved model to: {save_path}")

    plot_start = '2018-01-01'
    plot_end = '2018-12-31'
    val_indices = df.index[val_mask]
    plot_mask = (val_indices >= plot_start) & (val_indices <= plot_end)

    plt.figure(figsize=(15, 6))
    plt.plot(val_indices[plot_mask], y_val[plot_mask], label='Actual Volatility', color='black', alpha=0.6)

    for name, preds in results.items():
        plt.plot(val_indices[plot_mask], preds[plot_mask], label=name.replace("_", " "), linestyle='--')

    plt.title(f'Model Predictions vs Actual (Validation Slice: {plot_start} to {plot_end})')
    plt.ylabel('Volatility')
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(plot_dir, "model_comparison_classical.png"))
    print("Saved comparison plot to plots/model_comparison_classical.png")

if __name__ == "__main__":
    train_classical_models()