# =============================================================================
# NN.py — Short-Term Stock Movement Predictor
# Science Fair 2026
#
# Predicts whether a stock price will be HIGHER or LOWER in HORIZON days.
# Compares: LSTM Neural Network vs Logistic Regression vs Random Guessing
# =============================================================================

import os
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks

import ta
from datetime import datetime, timedelta
import joblib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

SEQ_LEN    = 90      # days of history fed to LSTM (try 70–120)
HORIZON    = 5       # days ahead to predict (1=next day, 5=1 week)
TEST_SIZE  = 0.2     # fraction of data for unseen test set
GAP        = 10      # buffer between train end and test start (prevents leakage)
BATCH_SIZE = 32
EPOCHS     = 40
SEED       = 42

MODEL_DIR = "saved_model"
os.makedirs(MODEL_DIR, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "AMZN"]

np.random.seed(SEED)
tf.random.set_seed(SEED)


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

def download_data(ticker: str, years: int = 10) -> pd.DataFrame:
    """
    Download daily OHLCV data from Yahoo Finance.
    Renames 'Close' to 'Adj Close' for consistency.
    Handles MultiIndex columns from newer yfinance versions.
    """
    end   = datetime.now()
    start = end - timedelta(days=365 * years)

    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.dropna()

    # Flatten MultiIndex if present (newer yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns={"Close": "Adj Close"})
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the following features to the dataframe:
      - return_1       : 1-day percentage return
      - return_5       : 5-day percentage return
      - SMA_10         : 10-day simple moving average
      - SMA_21         : 21-day simple moving average
      - EMA_20         : 20-day exponential moving average
      - RSI_14         : 14-day Relative Strength Index
      - BB_width       : Bollinger Band width normalized by price
      - vol_pct_change : 1-day volume percentage change
      - Volume         : log(1 + volume) to reduce skew
    """
    data = df.copy()

    # Price returns
    data["return_1"] = data["Adj Close"].pct_change(1)
    data["return_5"] = data["Adj Close"].pct_change(5)

    # Moving averages
    data["SMA_10"] = data["Adj Close"].rolling(10).mean()
    data["SMA_21"] = data["Adj Close"].rolling(21).mean()
    data["EMA_20"] = data["Adj Close"].ewm(span=20).mean()

    # Momentum
    data["RSI_14"] = ta.momentum.rsi(data["Adj Close"], window=14)

    # Volatility
    bb = ta.volatility.BollingerBands(data["Adj Close"])
    data["BB_width"] = (
        bb.bollinger_hband() - bb.bollinger_lband()
    ) / data["Adj Close"]

    # Volume
    data["vol_pct_change"] = data["Volume"].pct_change(1)
    data["Volume"]         = np.log1p(data["Volume"])

    return data.dropna()


# =============================================================================
# LABEL CREATION  (single source of truth)
# =============================================================================

def create_labels(df: pd.DataFrame, horizon: int) -> pd.Series:
    """
    Binary target:
      1 = Adj Close at (t + horizon) > Adj Close at t  (price went UP)
      0 = price went DOWN or stayed same
    """
    return (
        df["Adj Close"].shift(-horizon) > df["Adj Close"]
    ).astype(int)


# =============================================================================
# DATASET BUILDER
# =============================================================================

def create_dataset(
    df: pd.DataFrame,
    features: list,
    horizon: int,
    seq_len: int
):
    """
    Builds sliding-window sequences.

    X shape: (num_samples, seq_len, num_features)
    y shape: (num_samples,)

    Each sample X[i] is the feature matrix for the window
    [i - seq_len : i], and y[i] is the direction label
    for day (i + horizon).
    """
    X, y = [], []
    labels = create_labels(df, horizon)

    for i in range(seq_len, len(df) - horizon):
        X.append(df[features].iloc[i - seq_len:i].values)
        y.append(labels.iloc[i])

    return np.array(X), np.array(y)


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

def build_lstm_model(input_shape: tuple) -> tf.keras.Model:
    """
    LSTM binary classifier.

    Architecture:
        Input  (seq_len, num_features)
        LSTM   32 units, L2 regularization
        BatchNorm
        Dropout 0.3
        Dense  16, ReLU
        Dropout 0.3
        Dense  1,  Sigmoid  ->  P(UP)

    Compiled with Adam (lr=1e-4) and binary crossentropy loss.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32, kernel_regularizer=regularizers.l2(1e-3)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def train_with_random_data():
    """
    Randomly selects a ticker and a training period (5–20 years),
    trains the full pipeline, and saves model + scaler + report.

    Returns: (ticker, model_path, scaler_path)
    """
    ticker = np.random.choice(TICKERS)
    years  = np.random.randint(5, 20)
    return train_models(ticker, years)


def train_models(ticker: str, years: int = 10):
    """
    Full training pipeline for one ticker.

    Steps:
      1. Download + add indicators
      2. Build sequences (X, y)
      3. Chronological train/test split with GAP
      4. Fit scaler on train, apply to both splits
      5. Train LSTM with early stopping + class weights
      6. Train Logistic Regression on same scaled data
      7. Evaluate all three models on unseen test set
      8. Save model, scaler, JSON report

    Returns: (ticker, model_path, scaler_path)
    """
    print(f"\nTraining on {ticker} using {years} years of data")

    # -- Data --
    raw = download_data(ticker, years)
    df  = add_technical_indicators(raw)

    feature_cols = [
        "return_1", "return_5",
        "SMA_10",   "SMA_21",
        "EMA_20",   "RSI_14",
        "BB_width", "vol_pct_change", "Volume"
    ]

    X, y = create_dataset(df, feature_cols, HORIZON, SEQ_LEN)

    # -- Chronological split with GAP --
    split     = int(len(X) * (1 - TEST_SIZE))
    train_end = max(split - GAP, 0)

    X_train, y_train = X[:train_end], y[:train_end]
    X_test,  y_test  = X[split:],     y[split:]

    print(f"Train samples : {len(X_train)}")
    print(f"Test  samples : {len(X_test)}")

    # -- Scaling (fit on train only) --
    scaler      = StandardScaler()
    X_train_2d  = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_2d)

    X_train_scaled = (
        scaler.transform(X_train_2d).reshape(X_train.shape)
    )
    X_test_scaled = (
        scaler.transform(
            X_test.reshape(-1, X_test.shape[-1])
        ).reshape(X_test.shape)
    )

    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.joblib")
    model_path  = os.path.join(MODEL_DIR, f"{ticker}_best.h5")
    joblib.dump(scaler, scaler_path)

    # -- Class weights to handle imbalance --
    cw = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(cw))

    # -- Build + train LSTM --
    model = build_lstm_model((SEQ_LEN, X.shape[-1]))

    es = callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True,
        monitor="val_loss"
    )

    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[es],
        verbose=1
    )

    model.save(model_path)

    # -- LSTM evaluation --
    lstm_preds = (
        model.predict(X_test_scaled) > 0.5
    ).astype(int).flatten()
    lstm_acc = accuracy_score(y_test, lstm_preds)

    print("\n" + "="*40)
    print("LSTM TEST RESULTS")
    print("="*40)
    print("Accuracy :", lstm_acc)
    print(classification_report(y_test, lstm_preds))
    print(confusion_matrix(y_test, lstm_preds))

    # -- Logistic Regression (same scaler, same split) --
    X_train_flat = X_train_scaled.reshape(len(X_train_scaled), -1)
    X_test_flat  = X_test_scaled.reshape(len(X_test_scaled),  -1)

    log_reg = LogisticRegression(random_state=SEED, max_iter=1000)
    log_reg.fit(X_train_flat, y_train)

    log_preds = log_reg.predict(X_test_flat)
    log_acc   = accuracy_score(y_test, log_preds)

    print("\n" + "="*40)
    print("LOGISTIC REGRESSION RESULTS")
    print("="*40)
    print("Accuracy :", log_acc)
    print(classification_report(y_test, log_preds))
    print(confusion_matrix(y_test, log_preds))

    # -- Random guessing baseline --
    rand_preds = np.random.choice([0, 1], size=len(y_test))
    rand_acc   = accuracy_score(y_test, rand_preds)

    print("\n" + "="*40)
    print("RANDOM GUESSING BASELINE")
    print("="*40)
    print("Accuracy :", rand_acc)

    # -- Training curves --
    _plot_training_history(history, ticker)

    # -- Save report --
    report = {
        "ticker"               : ticker,
        "years"                : years,
        "horizon"              : HORIZON,
        "seq_len"              : SEQ_LEN,
        "lstm_accuracy"        : float(lstm_acc),
        "logistic_accuracy"    : float(log_acc),
        "random_accuracy"      : float(rand_acc),
        "train_samples"        : int(len(X_train)),
        "test_samples"         : int(len(X_test)),
        "date"                 : datetime.now().isoformat()
    }

    report_path = os.path.join(MODEL_DIR, f"{ticker}_report.json")
    pd.Series(report).to_json(report_path)
    print(f"\nReport saved to {report_path}")

    return ticker, model_path, scaler_path


# =============================================================================
# PLOTTING
# =============================================================================

def _plot_training_history(history, ticker: str):
    """Plot and show training/validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{ticker} — Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"],     label="Train Acc")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc")
    axes[1].set_title(f"{ticker} — Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict_next_week(
    ticker: str,
    model_path: str,
    scaler_path: str
) -> dict:
    """
    Loads the saved model and scaler, downloads the most recent
    data, and returns the probability that the stock will be UP
    in HORIZON trading days.
    """
    model  = models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    raw = download_data(ticker, years=2)
    df  = add_technical_indicators(raw)

    feature_cols = [
        "return_1", "return_5",
        "SMA_10",   "SMA_21",
        "EMA_20",   "RSI_14",
        "BB_width", "vol_pct_change", "Volume"
    ]

    # Use the most recent SEQ_LEN rows
    df_recent = df.iloc[-SEQ_LEN:]
    X = df_recent[feature_cols].values

    X_scaled = scaler.transform(X).reshape(1, SEQ_LEN, len(feature_cols))

    prob_up      = float(model.predict(X_scaled)[0][0])
    direction_up = prob_up >= 0.5

    return {
        "ticker"       : ticker,
        "prob_up"      : round(prob_up, 4),
        "direction_up" : direction_up,
        "horizon_days" : HORIZON
    }


def check_prediction(
    ticker: str,
    model_path: str,
    scaler_path: str
):
    """
    Makes a prediction AND checks it against the actual price
    movement that already happened (using the last known close).
    Useful for back-checking predictions on recent dates.
    """
    result = predict_next_week(ticker, model_path, scaler_path)
    print("\nPrediction:", result)

    raw = download_data(ticker, years=2)
    df  = add_technical_indicators(raw)

    actual = df["Adj Close"].iloc[-1] > df["Adj Close"].iloc[-(HORIZON + 1)]
    print("Actual:", "UP" if actual else "DOWN")
    print(
        "Result:",
        "CORRECT" if result["direction_up"] == actual else "WRONG"
    )


# =============================================================================
# RANDOM GUESS EVALUATION
# =============================================================================

def random_guess_evaluation(
    ticker: str,
    model_path: str,
    scaler_path: str
) -> dict:
    """
    Compare model prediction vs a random coin flip on the latest data point.
    """
    print("\nEvaluating random guess vs model prediction")

    model  = models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    raw = download_data(ticker, years=2)
    df  = add_technical_indicators(raw)

    feature_cols = [
        "return_1", "return_5",
        "SMA_10",   "SMA_21",
        "EMA_20",   "RSI_14",
        "BB_width", "vol_pct_change", "Volume"
    ]

    df_recent = df.iloc[-SEQ_LEN:]
    X = df_recent[feature_cols].values
    X_scaled = scaler.transform(X).reshape(1, SEQ_LEN, len(feature_cols))

    prob             = float(model.predict(X_scaled)[0][0])
    model_prediction = prob >= 0.5
    rand_prediction  = np.random.choice([True, False])

    actual = (
        df["Adj Close"].iloc[-1] > df["Adj Close"].iloc[-(HORIZON + 1)]
    )

    print(f"Random Guess : {'UP' if rand_prediction else 'DOWN'}")
    print(f"Model Guess  : {'UP' if model_prediction else 'DOWN'}")
    print(f"Actual       : {'UP' if actual else 'DOWN'}")
    print(f"Random Correct: {rand_prediction == actual}")
    print(f"Model  Correct: {model_prediction == actual}")

    return {
        "random_correct" : bool(rand_prediction == actual),
        "model_correct"  : bool(model_prediction == actual)
    }


# =============================================================================
# LOGISTIC REGRESSION — FULL EVALUATION
# =============================================================================

def logistic_regression_evaluation(
    ticker: str,
    model_path: str,
    scaler_path: str
) -> dict:
    """
    Trains a logistic regression on recent data and compares against
    the LSTM and random guessing on the same test split.
    Uses the same scaler and GAP logic as the LSTM training.
    """
    print("\nEvaluating Logistic Regression vs LSTM vs Random")

    scaler = joblib.load(scaler_path)

    raw = download_data(ticker, years=2)
    df  = add_technical_indicators(raw)

    feature_cols = [
        "return_1", "return_5",
        "SMA_10",   "SMA_21",
        "EMA_20",   "RSI_14",
        "BB_width", "vol_pct_change", "Volume"
    ]

    df_recent = df.iloc[-(SEQ_LEN + HORIZON):]
    X = df_recent[feature_cols].values
    y = create_labels(df_recent, HORIZON).values

    X = X[:len(y)]
    X = scaler.transform(X)

    # Split
    split = int(len(X) * (1 - TEST_SIZE))
    train_end = max(split - GAP, 0)

    X_train, y_train = X[:train_end], y[:train_end]
    X_test,  y_test  = X[split:],     y[split:]

    # Logistic Regression
    log_reg = LogisticRegression(random_state=SEED, max_iter=1000)
    log_reg.fit(X_train, y_train)
    log_preds = log_reg.predict(X_test)
    log_acc   = accuracy_score(y_test, log_preds)

    print("\nLogistic Regression Results:")
    print("Accuracy:", log_acc)
    print(classification_report(y_test, log_preds))
    print(confusion_matrix(y_test, log_preds))

    # LSTM on same test set
    lstm_acc   = None
    rand_acc   = None

    # Reshape for LSTM
    num_features = len(feature_cols)
    X_test_3d = X_test.reshape(-1, 1, num_features)  # single-step sequences

    if X_test_3d.size > 0:
        model = models.load_model(model_path)
        lstm_preds = (
            model.predict(
                X_test.reshape(-1, 1, num_features)
            ) > 0.5
        ).astype(int).flatten()

        if len(lstm_preds) == len(y_test):
            lstm_acc = accuracy_score(y_test, lstm_preds)
            print("\nLSTM Model Results:")
            print("Accuracy:", lstm_acc)

        rand_preds = np.random.choice([0, 1], size=len(y_test))
        rand_acc   = accuracy_score(y_test, rand_preds)
        print("\nRandom Guessing Results:")
        print("Accuracy:", rand_acc)

    return {
        "logistic_regression_accuracy" : log_acc,
        "lstm_accuracy"                : lstm_acc,
        "random_accuracy"              : rand_acc
    }


# =============================================================================
# SINGLE-POINT LOGISTIC REGRESSION PREDICTION
# =============================================================================

def evaluate_logistic_regression_single(
    ticker: str,
    model_path: str,
    scaler_path: str
):
    """
    Trains logistic regression on all-but-the-last data point,
    then predicts that last point and checks against actual.
    """
    print("\nEvaluating Logistic Regression (single prediction)")

    scaler = joblib.load(scaler_path)

    raw = download_data(ticker, years=2)
    df  = add_technical_indicators(raw)

    feature_cols = [
        "return_1", "return_5",
        "SMA_10",   "SMA_21",
        "EMA_20",   "RSI_14",
        "BB_width", "vol_pct_change", "Volume"
    ]

    df_recent = df.iloc[-(SEQ_LEN + HORIZON):]
    X = df_recent[feature_cols].values
    y = create_labels(df_recent, HORIZON).values

    X = X[:len(y)]
    X = scaler.transform(X)

    X_single  = X[-1].reshape(1, -1)
    y_single  = y[-1]

    log_reg = LogisticRegression(random_state=SEED, max_iter=1000)
    log_reg.fit(X[:-1], y[:-1])
    log_pred = log_reg.predict(X_single)[0]

    print(f"Prediction : {'UP' if log_pred == 1 else 'DOWN'}")
    print(f"Actual     : {'UP' if y_single == 1 else 'DOWN'}")
    print(f"Result     : {'CORRECT' if log_pred == y_single else 'WRONG'}")


# =============================================================================
# MAIN MENU
# =============================================================================

if __name__ == "__main__":
    choice = input(
        "Enter 1 to train model, 2 to predict and evaluate: "
    ).strip()

    if choice == "1":
        print("Training...")
        ticker, model_path, scaler_path = train_with_random_data()

    elif choice == "2":
        ticker = input(
            f"Ticker ({', '.join(TICKERS)}): "
        ).upper().strip()

        model_path  = os.path.join(MODEL_DIR, f"{ticker}_best.h5")
        scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.joblib")

        if not os.path.exists(model_path):
            print("Model not found. Train first (option 1).")
        else:
            check_prediction(ticker, model_path, scaler_path)
            random_guess_evaluation(ticker, model_path, scaler_path)
            evaluate_logistic_regression_single(
                ticker, model_path, scaler_path
            )
    else:
        print("Invalid choice. Enter 1 or 2.")
