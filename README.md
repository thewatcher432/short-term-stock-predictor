# 📈 Short-Term Stock Movement Predictor

> **Science Fair Project** — Predicting whether a stock will go UP or DOWN over the next N trading days using a Long Short-Term Memory (LSTM) neural network, technical indicators, and baseline comparisons.

---

## 🧠 Overview

This project builds a complete machine learning pipeline to predict the **short-term directional movement** of U.S. stocks (AAPL, MSFT, AMZN). Instead of trying to predict the exact price (which is nearly impossible), the model asks a simpler question:

> *Will the stock close higher in 5 trading days than it does today?*

That becomes a **binary classification problem** (1 = UP, 0 = DOWN), which is well-suited for an LSTM that reads a rolling window of the past 90 days of technical features.

Three models are trained and compared head-to-head:
- **LSTM Neural Network** — the main model
- **Logistic Regression** — traditional ML baseline
- **Random Guessing** — coin-flip baseline

This lets us scientifically measure whether the neural network adds real predictive power.

---

## 📁 Project Structure

```
short-term-stock-predictor/
├── NN.py                  # Main script (data, model, train, evaluate, predict)
├── saved_model/           # Auto-created folder for saved models and scalers
│   ├── AAPL_best.h5       # Trained LSTM model
│   ├── AAPL_scaler.joblib # Fitted StandardScaler
│   └── AAPL_report.json   # Accuracy results per run
├── requirements.txt       # All dependencies
└── README.md              # This file
```

---

## ⚙️ Configuration

All key settings are at the top of `NN.py`:

| Parameter     | Default | Description                                      |
|---------------|---------|--------------------------------------------------|
| `SEQ_LEN`     | 90      | Number of past days fed into the LSTM            |
| `HORIZON`     | 5       | Days ahead to predict (1 = next day, 5 = 1 week)|
| `TEST_SIZE`   | 0.2     | Fraction of data reserved for unseen test set    |
| `GAP`         | 10      | Buffer between train and test to prevent leakage |
| `BATCH_SIZE`  | 32      | Training batch size                              |
| `EPOCHS`      | 40      | Max training epochs (early stopping active)      |
| `TICKERS`     | AAPL, MSFT, AMZN | Stocks to train/predict on           |

---

## 🔧 Features Used

For each day in the sequence window, the model sees 9 features:

| Feature          | Description                                         |
|------------------|-----------------------------------------------------|
| `return_1`       | 1-day price return (pct change)                    |
| `return_5`       | 5-day price return                                 |
| `SMA_10`         | 10-day Simple Moving Average                       |
| `SMA_21`         | 21-day Simple Moving Average                       |
| `EMA_20`         | 20-day Exponential Moving Average                  |
| `RSI_14`         | 14-day Relative Strength Index (momentum)          |
| `BB_width`       | Bollinger Band width (volatility measure)          |
| `vol_pct_change` | 1-day volume percent change                        |
| `Volume`         | Log-scaled trading volume                         |

All features are normalized using a `StandardScaler` fit **only on training data** to prevent data leakage.

---

## 🧱 Model Architecture

```
Input → (SEQ_LEN=90, features=9)
  │
  ▼
LSTM(32 units, L2 regularization=1e-3)
  │
  ▼
BatchNormalization
  │
  ▼
Dropout(0.3)
  │
  ▼
Dense(16, activation=ReLU)
  │
  ▼
Dropout(0.3)
  │
  ▼
Dense(1, activation=Sigmoid)  →  P(price goes UP)
```

- **Optimizer:** Adam (lr = 1e-4)
- **Loss:** Binary Crossentropy
- **Class Weights:** Balanced to handle UP/DOWN imbalance
- **Early Stopping:** patience=5, monitors `val_loss`, restores best weights

---

## 🚫 Preventing Data Leakage

Data leakage is the #1 mistake in financial ML. This project avoids it three ways:

1. **Chronological split** — train data is always earlier than test data, never shuffled.
2. **GAP buffer** — 10-day gap between end of training and start of test so no overlapping label windows.
3. **Scaler fit on train only** — `StandardScaler` is fit exclusively on `X_train`, then applied to `X_test` and live data.

---

## 📊 Evaluation

After training, the model is compared against two baselines on the **same unseen test set**:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model                  Test Accuracy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LSTM Neural Network    ~55–65%
  Logistic Regression    ~52–58%
  Random Guessing        ~50%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

> Stock markets are noisy. Even beating random guessing consistently is a meaningful result. A 55–60% directional accuracy is competitive with published academic results on this task.

Metrics reported:
- Accuracy
- Precision / Recall / F1-score (full `classification_report`)
- Confusion Matrix

All results are saved to `saved_model/{TICKER}_report.json` for record keeping.

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python NN.py
# Select option 1 when prompted
```

This will:
- Download 10 years of data for the ticker
- Engineer all features
- Train LSTM + Logistic Regression
- Print test results for all three models
- Save model, scaler, and JSON report

### 3. Make a prediction on today's data

```bash
python NN.py
# Select option 2 when prompted
# Enter ticker (e.g. AAPL)
```

Outputs the probability the stock will be UP in `HORIZON` days.

---

## 🔬 Experiments You Can Try

| Change This            | How to Change              | What to Observe               |
|------------------------|----------------------------|-------------------------------|
| Shorter window         | `SEQ_LEN = 70`             | Does less history help/hurt?  |
| Next-day prediction    | `HORIZON = 1`              | Harder or easier to predict?  |
| More LSTM units        | `LSTM(64, ...)`            | Does bigger model overfit?    |
| More features          | Add MACD, ATR, OBV         | Does more data improve acc?   |
| Different ticker       | `TICKER = "TSLA"`          | Is TSLA more predictable?     |

---

## 📦 Requirements

```
yfinance
numpy
pandas
scikit-learn
tensorflow
ta
joblib
matplotlib
```

---

## ⚠️ Disclaimer

This project is for **educational and science fair purposes only**. It is not financial advice. Do not use model predictions to make real investment decisions. Past patterns in stock data do not guarantee future results.

---

## 👤 Author

**The Watcher** — Science Fair 2026  
Built with Python, TensorFlow/Keras, and scikit-learn.
