import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def run_eda():
    """
    Generates exploratory visualizations for the S&P 500 volatility dataset.
    
    Outputs:
    1. volatility_clusters.png: Visualizes the target variable over time with split regions.
    2. feature_correlations.png: Heatmap showing correlation between lags and future volatility.
    3. returns_distribution.png: Histogram of daily returns.
    """
    input_path = os.path.join("data", "processed", "sp500_clean.csv")
    if not os.path.exists(input_path):
        print(f"Error: Processed data not found at {input_path}")
        return

    print("Loading data for EDA...")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Target_Vol'], label='Realized Volatility (Squared Returns)', alpha=0.7, linewidth=0.8)
    plt.axvline(pd.Timestamp('2015-01-01'), color='r', linestyle='--', alpha=0.8)
    plt.axvline(pd.Timestamp('2020-01-01'), color='r', linestyle='--', alpha=0.8)
    plt.text(pd.Timestamp('2000-01-01'), df['Target_Vol'].max()*0.8, 'TRAINING SET', color='red', fontsize=12)
    plt.text(pd.Timestamp('2016-06-01'), df['Target_Vol'].max()*0.8, 'VAL', color='red', fontsize=12)
    plt.text(pd.Timestamp('2022-01-01'), df['Target_Vol'].max()*0.8, 'TEST', color='red', fontsize=12)

    plt.title('S&P 500 Realized Volatility & Data Splits')
    plt.ylabel('Volatility (Returns²)')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "volatility_clusters.png"))
    plt.close()
    print("Saved: volatility_clusters.png")

    feature_cols = ['Target_Vol'] + [c for c in df.columns if 'Lag' in c or 'Roll' in c]
    corr_matrix = df[feature_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "feature_correlations.png"))
    plt.close()
    print("Saved: feature_correlations.png")

    plt.figure(figsize=(10, 6))
    sns.histplot(df['Log_Return'], bins=100, kde=True, color='blue')
    plt.title('Distribution of Daily Log Returns')
    plt.xlabel('Log Return')
    plt.savefig(os.path.join(plot_dir, "returns_distribution.png"))
    plt.close()
    print("Saved: returns_distribution.png")

if __name__ == "__main__":
    run_eda()