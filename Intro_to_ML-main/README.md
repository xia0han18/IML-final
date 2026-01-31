# Introduction to Machine Learning – Assignment 3
**Xiaohan Bao (S4036298), Marcell Szőkedencsi (s5173191), Ana Abadias Alfonso (s5460824)**  
**December 4, 2025**

---

## Motivation & General Idea
Financial market volatility is a nonlinear and challenging time-series prediction problem. Given our interest in financial markets and existing background in finance, the aim of this project is to build and evaluate machine learning models to forecast short-term volatility of the S&P 500 index.  

---

## Dataset
We use historical price data of the S&P 500 index (**ticker: ^GSPC**) retrieved using Python's **yfinance** library.  
The dataset spans from 1990 to December 2025 (filtered to match VIX availability).

Included variables:
- Daily log returns
- Realized volatility (RV) as 21-day rolling standard deviation
- VIX (implied volatility index)
- Lag features (1-5 days)
- Rolling-window features (5-day and 21-day rolling means)

---

## Project Structure
```
Intro_to_ML-main/
├── data/
│   ├── raw/
│   │   └── sp500_raw.csv           # Raw data from yfinance
│   └── processed/
│       └── sp500_ml_ready.csv      # Processed data with features
├── models/
│   ├── Linear_Regression.pkl
│   ├── Ridge_Regression.pkl
│   ├── Random_Forest.pkl
│   ├── XGBoost.pkl
│   ├── KNN.pkl
│   ├── MLP.pt
│   ├── LSTM.pt
│   └── *.pkl (scalers)
├── plots/
│   ├── volatility_rv_vix.png
│   ├── volatility_clusters.png
│   ├── feature_correlations.png
│   ├── returns_distribution.png
│   ├── qq_plot.png
│   ├── model_comparison_classical.png
│   ├── backtest_all_models.png
│   └── backtest_best_model.png
├── outputs/
│   ├── model_comparison_RMSE.csv
│   ├── model_comparison_MAE.csv
│   ├── backtest_RMSE.csv
│   └── backtest_MAE.csv
├── src/
│   ├── fetch_data.py               # Download data from yfinance
│   ├── process_data.py             # Feature engineering
│   ├── eda.py                      # Exploratory data analysis
│   ├── train_classical.py          # Train classical ML models
│   ├── train_dl.py                 # Train deep learning models (LSTM)
│   ├── comparison.py               # Compare all models
│   └── backtest.py                 # Final backtesting
├── README.md
├── requirements.txt
└── report.md
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Intro_to_ML-main.git
cd Intro_to_ML-main
```

### 2. Create virtual environment (optional)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Requirements

Create a `requirements.txt` file with:
```
# Data handling
numpy
pandas

# Data fetching
yfinance

# Visualization
matplotlib
seaborn

# Machine Learning (Classical)
scikit-learn
xgboost
joblib

# Deep Learning
torch

# Statistics
scipy
```

Or install manually:
```bash
pip install numpy pandas yfinance matplotlib seaborn scikit-learn xgboost joblib torch scipy
```

---

## Usage

### Step 1: Fetch Data
Download S&P 500 and VIX data from Yahoo Finance:
```bash
cd Intro_to_ML-main
python src/fetch_data.py
```
**Output:** `data/raw/sp500_raw.csv`

### Step 2: Process Data
Engineer features (lags, rolling windows, VIX features):
```bash
python src/process_data.py
```
**Output:** `data/processed/sp500_ml_ready.csv`

### Step 3: Exploratory Data Analysis
Generate visualizations:
```bash
python src/eda.py
```
**Output:** 
- `plots/volatility_rv_vix.png`
- `plots/volatility_clusters.png`
- `plots/feature_correlations.png`
- `plots/returns_distribution.png`
- `plots/qq_plot.png`

### Step 4: Train Classical ML Models
Train Linear Regression, Ridge, Random Forest, XGBoost, KNN:
```bash
python src/train_classical.py
```
**Output:** 
- `models/*.pkl`
- `plots/model_comparison_classical.png`

### Step 5: Train Deep Learning Models
Train MLP and LSTM with hyperparameter tuning:
```bash
python src/train_dl.py
```
**Output:** 
- `models/MLP.pt`
- `models/LSTM.pt`
- `plots/dataset_split.png`
- `plots/best_batch_size.png`
- `plots/best_epoch.png`
- `plots/LSTM_train-test.png`

### Step 6: Compare All Models
Generate comparison tables:
```bash
python src/comparison.py
```
**Output:** 
- `outputs/model_comparison_RMSE.csv`
- `outputs/model_comparison_MAE.csv`

### Step 7: Backtesting
Evaluate all models on test set (2020-2025):
```bash
python src/backtest.py
```
**Output:** 
- `outputs/backtest_RMSE.csv`
- `outputs/backtest_MAE.csv`
- `plots/backtest_all_models.png`
- `plots/backtest_best_model.png`

---

## Data Splits

| Split | Period | Purpose |
|-------|--------|---------|
| Train | 1990 - 2014 | Model training |
| Validation | 2015 - 2019 | Hyperparameter tuning |
| Test | 2020 - 2025 | Final evaluation |

---

## Models

### Classical ML Models (scikit-learn)
| Model | Description |
|-------|-------------|
| Linear Regression | Baseline linear model |
| Ridge Regression | L2 regularization (alpha=1.0) |
| Random Forest | Ensemble of decision trees (n_estimators=200, max_depth=10) |
| XGBoost | Gradient boosted trees (n_estimators=300, max_depth=5) |
| KNN | K-Nearest Neighbors (n_neighbors=10, weights='distance') |

### Deep Learning Models (PyTorch)
| Model | Description |
|-------|-------------|
| MLP | Multilayer Perceptron (64 → 32 → 1 neurons) |
| LSTM | Long Short-Term Memory network (hidden_dim=64, num_layers=2) |

---

## Features

| Feature | Description |
|---------|-------------|
| `Target_Vol` | 21-day rolling standard deviation of log returns |
| `Log_Return` | Daily log return: ln(P_t / P_{t-1}) |
| `Vol_Lag_1` to `Vol_Lag_5` | Lagged volatility (1-5 days) |
| `Return_Lag_1` to `Return_Lag_5` | Lagged returns (1-5 days) |
| `Vol_Roll_Mean_5` | 5-day rolling mean of volatility |
| `Vol_Roll_Mean_21` | 21-day rolling mean of volatility |
| `VIX` | CBOE Volatility Index (implied volatility) |
| `VIX_Lag_1`, `VIX_Lag_5` | Lagged VIX values |
| `VIX_Roll_Mean_5` | 5-day rolling mean of VIX |

---

## Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| RMSE | √(Σ(y - ŷ)² / n) | Root Mean Squared Error |
| MAE | Σ\|y - ŷ\| / n | Mean Absolute Error |

---

## Results

### RMSE Comparison (Test Set)
| Model | RMSE |
|-------|------|
| MLP | 0.000602 |
| XGBoost | 0.000604 |
| Random Forest | 0.000607 |
| LSTM | 0.000618 |
| Linear Regression | 0.000621 |
| KNN | 0.000624 |
| Ridge Regression | 0.000638 |

### MAE Comparison (Test Set)
| Model | MAE |
|-------|-----|
| XGBoost | 0.000166 |
| KNN | 0.000168 |
| LSTM | 0.000168 |
| Random Forest | 0.000175 |
| MLP | 0.000178 |
| Linear Regression | 0.000185 |
| Ridge Regression | 0.000186 |

---

## Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.21 | Numerical computing |
| `pandas` | ≥1.3 | Data manipulation |
| `yfinance` | ≥0.2 | Download financial data |
| `matplotlib` | ≥3.4 | Visualization |
| `seaborn` | ≥0.11 | Statistical visualization |
| `scikit-learn` | ≥1.0 | Classical ML models |
| `xgboost` | ≥1.5 | Gradient boosting |
| `torch` | ≥1.10 | Deep learning (PyTorch) |
| `scipy` | ≥1.7 | Statistical functions |
| `joblib` | ≥1.1 | Model serialization |

---

## References

1. Yahoo Finance API via yfinance
2. scikit-learn documentation
3. PyTorch documentation
4. CBOE VIX Index methodology

