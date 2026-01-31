import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run_backtest():
    """Run a rolling-style backtest on the 2020-2025 test window."""
    print("--- Starting Rolling Backtest (Test Set: 2020-2025) ---")

    data_path = os.path.join("data", "processed", "sp500_clean.csv")
    model_path = os.path.join("models", "Linear_Regression.pkl")

    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("Error: Missing data or model file.")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    model = joblib.load(model_path)

    test_mask = df.index >= '2020-01-01'
    df_test = df[test_mask].copy()

    drop_cols = ['Target_Vol', 'Log_Return']
    feature_cols = [
        c
        for c in df.columns
        if c not in drop_cols and ('Lag' in c or 'Roll' in c)
    ]

    X_test = df_test[feature_cols]
    y_test = df_test['Target_Vol']

    print(f"Test Set Range: {df_test.index.min()} to {df_test.index.max()}")
    print(f"Test Samples: {len(df_test)}")

    print("Generating predictions...")
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n--- Test Set Results ---")
    print(f"MSE: {mse:.8f}")
    print(f"MAE: {mae:.8f}")
    print(f"R²:  {r2:.6f}")

    plt.figure(figsize=(15, 7))

    plt.plot(
        df_test.index,
        y_test,
        label='Actual Volatility',
        color='black',
        alpha=0.5,
        linewidth=1,
    )
    plt.plot(
        df_test.index,
        predictions,
        label='Predicted (Linear Reg)',
        color='blue',
        alpha=0.8,
        linewidth=1,
    )

    plt.axvspan(
        pd.Timestamp('2020-03-01'),
        pd.Timestamp('2020-05-01'),
        color='red',
        alpha=0.1,
        label='COVID Crash',
    )

    plt.title('Final Backtest: Forecasting S&P 500 Volatility (2020-2025)')
    plt.ylabel('Volatility (Squared Returns)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = os.path.join("plots", "final_backtest.png")
    plt.savefig(out_path)
    print(f"Saved final backtest plot to {out_path}")


if __name__ == "__main__":
    run_backtest()
