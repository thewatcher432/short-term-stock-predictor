App.py — Tkinter GUI wrapper for the LSTM stock predictor (imports from NN.py)

import os
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from datetime import datetime

# Matplotlib embedded in Tkinter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import all ML backend functions from NN.py
from NN import (
    download_data,
    add_technical_indicators,
    create_dataset,
    build_lstm_model,
    train_with_random_data,
    predict_next_week,
    random_guess_evaluation,
    logistic_regression_evaluation,
    evaluate_logistic_regression_single,
    MODEL_DIR,
    TICKERS,
    SEQ_LEN,
    HORIZON,
    TEST_SIZE,
    GAP,
    BATCH_SIZE,
    EPOCHS,
    SEED,
)

import tensorflow as tf
from tensorflow.keras import models, callbacks, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
import joblib


# ─────────────────────────────────────────────
# Dark theme color palette
# ─────────────────────────────────────────────
BG       = "#1e1e2e"   # main background
SURFACE  = "#2a2a3e"   # card / panel background
ACCENT   = "#7c6af7"   # purple accent
GREEN    = "#50fa7b"
RED      = "#ff5555"
FG       = "#cdd6f4"   # primary text
MUTED    = "#6c7086"   # secondary text
BORDER   = "#313244"


class StockPredictorApp(tk.Tk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.title("LSTM Stock Movement Predictor")
        self.geometry("1200x820")
        self.configure(bg=BG)
        self.resizable(True, True)

        # Track the currently selected ticker
        self.selected_ticker = tk.StringVar(value="AAPL")

        # History object returned by Keras training (used for chart)
        self.train_history = None
        self.trained_ticker = None

        self._apply_ttk_theme()
        self._build_ui()

    # ── TTK THEME ────────────────────────────────────────────────────────────

    def _apply_ttk_theme(self):
        """Apply a clean dark ttk style across all widgets."""
        style = ttk.Style(self)
        style.theme_use("clam")

        # Global settings
        style.configure(".",
            background=BG,
            foreground=FG,
            fieldbackground=SURFACE,
            troughcolor=SURFACE,
            bordercolor=BORDER,
            darkcolor=BORDER,
            lightcolor=BORDER,
            font=("Segoe UI", 10),
        )

        # Frames
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=SURFACE, relief="flat")

        # Labels
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("Card.TLabel", background=SURFACE, foreground=FG)
        style.configure("Muted.TLabel", background=SURFACE, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=BG, foreground=FG,
                        font=("Segoe UI", 14, "bold"))
        style.configure("SectionTitle.TLabel", background=SURFACE, foreground=FG,
                        font=("Segoe UI", 11, "bold"))

        # Buttons
        style.configure("Accent.TButton",
            background=ACCENT, foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padding=(12, 6),
            relief="flat",
        )
        style.map("Accent.TButton",
            background=[("active", "#6a5be0"), ("disabled", MUTED)],
            foreground=[("disabled", "#888888")],
        )

        style.configure("TButton",
            background=SURFACE, foreground=FG,
            font=("Segoe UI", 10),
            padding=(10, 5),
            relief="flat",
        )
        style.map("TButton",
            background=[("active", BORDER)],
        )

        # Entry
        style.configure("TEntry",
            fieldbackground=SURFACE,
            foreground=FG,
            insertcolor=FG,
            bordercolor=BORDER,
            padding=6,
        )

        # Combobox
        style.configure("TCombobox",
            fieldbackground=SURFACE,
            background=SURFACE,
            foreground=FG,
            arrowcolor=FG,
        )
        style.map("TCombobox",
            fieldbackground=[("readonly", SURFACE)],
            foreground=[("readonly", FG)],
        )

        # Treeview (results table)
        style.configure("Treeview",
            background=SURFACE,
            fieldbackground=SURFACE,
            foreground=FG,
            rowheight=26,
            font=("Segoe UI", 9),
        )
        style.configure("Treeview.Heading",
            background=BORDER,
            foreground=FG,
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Treeview",
            background=[("selected", ACCENT)],
            foreground=[("selected", "#ffffff")],
        )

        # Scrollbar
        style.configure("Vertical.TScrollbar",
            background=SURFACE, troughcolor=BG, arrowcolor=MUTED)

        # Separator
        style.configure("TSeparator", background=BORDER)

        # Progress bar
        style.configure("Accent.Horizontal.TProgressbar",
            troughcolor=SURFACE, background=ACCENT)

    # ── UI LAYOUT ────────────────────────────────────────────────────────────

    def _build_ui(self):
        """Construct all GUI sections."""

        # ── Top bar ──────────────────────────────────────────────────────────
        top_bar = ttk.Frame(self, style="TFrame")
        top_bar.pack(fill="x", padx=20, pady=(16, 8))

        ttk.Label(top_bar, text="📈 LSTM Stock Predictor", style="Title.TLabel").pack(side="left")

        # Status label (right-aligned)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(top_bar, textvariable=self.status_var,
                  style="Muted.TLabel", background=BG).pack(side="right", padx=8)

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=20, pady=4)

        # ── Main content: left controls + right chart ─────────────────────
        content = ttk.Frame(self, style="TFrame")
        content.pack(fill="both", expand=True, padx=20, pady=8)

        left = ttk.Frame(content, style="TFrame", width=380)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        right = ttk.Frame(content, style="TFrame")
        right.pack(side="left", fill="both", expand=True)

        self._build_left_panel(left)
        self._build_right_panel(right)

    # ── LEFT PANEL ───────────────────────────────────────────────────────────

    def _build_left_panel(self, parent):
        """Search bar, action buttons, prediction result, eval table."""

        # ── Ticker search card ───────────────────────────────────────────
        search_card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        search_card.pack(fill="x", pady=(0, 10))

        ttk.Label(search_card, text="Ticker / Symbol", style="SectionTitle.TLabel").pack(anchor="w")
        ttk.Label(search_card,
                  text="Search any US stock, ETF, or mutual fund",
                  style="Muted.TLabel").pack(anchor="w", pady=(2, 8))

        entry_row = ttk.Frame(search_card, style="Card.TFrame")
        entry_row.pack(fill="x")

        # Text entry for manual ticker input
        self.ticker_entry = ttk.Entry(entry_row, textvariable=self.selected_ticker,
                                      font=("Segoe UI", 11))
        self.ticker_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.ticker_entry.bind("<Return>", lambda _: self._on_ticker_change())

        ttk.Button(entry_row, text="Set", style="TButton",
                   command=self._on_ticker_change).pack(side="left")

        # Quick-pick known tickers
        quick_row = ttk.Frame(search_card, style="Card.TFrame")
        quick_row.pack(fill="x", pady=(8, 0))
        ttk.Label(quick_row, text="Quick picks:", style="Muted.TLabel").pack(side="left", padx=(0, 6))
        for t in ["AAPL", "MSFT", "AMZN", "SPY", "QQQ"]:
            ttk.Button(quick_row, text=t, style="TButton",
                       command=lambda tick=t: self._set_ticker(tick)).pack(side="left", padx=2)

        # Current ticker display
        self.ticker_display = tk.StringVar(value=f"Active: {self.selected_ticker.get()}")
        ttk.Label(search_card, textvariable=self.ticker_display,
                  style="Muted.TLabel").pack(anchor="w", pady=(6, 0))

        # ── Action buttons card ───────────────────────────────────────────
        btn_card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        btn_card.pack(fill="x", pady=(0, 10))

        ttk.Label(btn_card, text="Actions", style="SectionTitle.TLabel").pack(anchor="w", pady=(0, 8))

        # Train Model button
        self.train_btn = ttk.Button(btn_card, text="🧠  Train Model",
                                    style="Accent.TButton", command=self._start_training)
        self.train_btn.pack(fill="x", pady=(0, 6))

        # Progress bar shown during training
        self.progress = ttk.Progressbar(btn_card, mode="indeterminate",
                                        style="Accent.Horizontal.TProgressbar")
        self.progress.pack(fill="x", pady=(0, 4))

        # Status label below progress bar
        self.train_status = tk.StringVar(value="")
        ttk.Label(btn_card, textvariable=self.train_status,
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 8))

        ttk.Separator(btn_card, orient="horizontal").pack(fill="x", pady=6)

        # Predict button
        self.predict_btn = ttk.Button(btn_card, text="🔮  Predict Next Week",
                                      style="TButton", command=self._start_prediction)
        self.predict_btn.pack(fill="x", pady=(0, 6))

        # Evaluate button
        self.eval_btn = ttk.Button(btn_card, text="📊  Evaluate Model",
                                   style="TButton", command=self._start_evaluation)
        self.eval_btn.pack(fill="x")

        # ── Prediction result card ────────────────────────────────────────
        pred_card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        pred_card.pack(fill="x", pady=(0, 10))

        ttk.Label(pred_card, text="Prediction Result", style="SectionTitle.TLabel").pack(anchor="w", pady=(0, 8))

        result_row = ttk.Frame(pred_card, style="Card.TFrame")
        result_row.pack(fill="x")

        # Direction indicator (UP / DOWN / —)
        self.direction_var = tk.StringVar(value="—")
        self.direction_label = tk.Label(result_row, textvariable=self.direction_var,
                                        font=("Segoe UI", 28, "bold"),
                                        bg=SURFACE, fg=MUTED)
        self.direction_label.pack(side="left", padx=(0, 14))

        detail_col = ttk.Frame(result_row, style="Card.TFrame")
        detail_col.pack(side="left", fill="x", expand=True)

        self.prob_var = tk.StringVar(value="Probability: —")
        ttk.Label(detail_col, textvariable=self.prob_var,
                  style="Card.TLabel", font=("Segoe UI", 10)).pack(anchor="w")

        self.pred_ticker_var = tk.StringVar(value="Ticker: —")
        ttk.Label(detail_col, textvariable=self.pred_ticker_var,
                  style="Muted.TLabel").pack(anchor="w")

        self.pred_date_var = tk.StringVar(value="")
        ttk.Label(detail_col, textvariable=self.pred_date_var,
                  style="Muted.TLabel").pack(anchor="w")

        # ── Evaluation results table ──────────────────────────────────────
        eval_card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        eval_card.pack(fill="both", expand=True)

        ttk.Label(eval_card, text="Evaluation Results", style="SectionTitle.TLabel").pack(anchor="w", pady=(0, 8))

        cols = ("Model", "Accuracy")
        self.eval_tree = ttk.Treeview(eval_card, columns=cols, show="headings", height=6)
        for col in cols:
            self.eval_tree.heading(col, text=col)
            self.eval_tree.column(col, width=140, anchor="center")
        self.eval_tree.pack(fill="both", expand=True)

        # ── Last JSON report card ─────────────────────────────────────────
        report_card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        report_card.pack(fill="x", pady=(10, 0))

        ttk.Label(report_card, text="Last Saved Report", style="SectionTitle.TLabel").pack(anchor="w", pady=(0, 6))

        self.report_var = tk.StringVar(value="No report loaded.")
        ttk.Label(report_card, textvariable=self.report_var,
                  style="Muted.TLabel", wraplength=340, justify="left").pack(anchor="w")

    # ── RIGHT PANEL (chart) ───────────────────────────────────────────────────

    def _build_right_panel(self, parent):
        """Embedded matplotlib chart for training loss & accuracy."""
        chart_card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        chart_card.pack(fill="both", expand=True)

        ttk.Label(chart_card, text="Training Curves", style="SectionTitle.TLabel").pack(anchor="w", pady=(0, 8))

        # Create figure with dark style
        self.fig = Figure(figsize=(7, 5), facecolor=SURFACE)
        self.ax_loss = self.fig.add_subplot(211)
        self.ax_acc  = self.fig.add_subplot(212)
        self._style_axes(self.ax_loss, "Loss")
        self._style_axes(self.ax_acc,  "Accuracy")
        self.fig.tight_layout(pad=2.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_card)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

    def _style_axes(self, ax, ylabel):
        """Apply dark theme styling to a matplotlib axes."""
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
        ax.set_xlabel("Epoch", color=MUTED, fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.title.set_color(FG)

    # ── TICKER HELPERS ────────────────────────────────────────────────────────

    def _set_ticker(self, ticker):
        """Set ticker from a quick-pick button."""
        self.selected_ticker.set(ticker)
        self._on_ticker_change()

    def _on_ticker_change(self):
        """Update the active ticker display and load any saved report."""
        t = self.selected_ticker.get().strip().upper()
        self.selected_ticker.set(t)
        self.ticker_display.set(f"Active: {t}")
        self._load_report(t)

    # ── TRAINING ─────────────────────────────────────────────────────────────

    def _start_training(self):
        """Launch model training in a background thread."""
        self.train_btn.config(state="disabled")
        self.predict_btn.config(state="disabled")
        self.eval_btn.config(state="disabled")
        self.train_status.set("⏳ Downloading data & training…")
        self.status_var.set("Training…")
        self.progress.start(12)   # animate indeterminate bar

        # Run training off the main thread so UI stays responsive
        t = threading.Thread(target=self._train_thread, daemon=True)
        t.start()

    def _train_thread(self):
        """Background: calls the NN training pipeline, then updates UI."""
        try:
            # We patch train_with_random_data to also capture history.
            # Since we need the history object for charts, we run the
            # training inline here and mirror the NN.py logic.
            ticker, model_path, scaler_path, history = _train_and_capture_history(
                self.selected_ticker.get().strip().upper()
            )
            self.trained_ticker = ticker
            self.train_history   = history

            # Marshal UI updates back to the main thread
            self.after(0, self._on_training_done, ticker, model_path, scaler_path)
        except Exception as exc:
            self.after(0, self._on_training_error, str(exc))

    def _on_training_done(self, ticker, model_path, scaler_path):
        """Called on the main thread after training succeeds."""
        self.progress.stop()
        self.train_status.set(f"✅ Trained on {ticker} — model saved.")
        self.status_var.set("Idle")
        self._re_enable_buttons()
        self._plot_history(self.train_history)
        self._load_report(ticker)

    def _on_training_error(self, msg):
        """Called on the main thread if training raises an exception."""
        self.progress.stop()
        self.train_status.set("❌ Training failed.")
        self.status_var.set("Error")
        self._re_enable_buttons()
        messagebox.showerror("Training Error", msg)

    # ── PREDICTION ───────────────────────────────────────────────────────────

    def _start_prediction(self):
        """Launch prediction in a background thread."""
        ticker = self.selected_ticker.get().strip().upper()
        model_path  = f"{MODEL_DIR}/{ticker}_best.h5"
        scaler_path = f"{MODEL_DIR}/{ticker}_scaler.joblib"

        # Guard: check files exist before starting thread
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            messagebox.showerror(
                "Model Not Found",
                f"No saved model found for {ticker}.\nPlease train the model first."
            )
            return

        self._disable_buttons()
        self.status_var.set("Predicting…")

        t = threading.Thread(
            target=self._predict_thread,
            args=(ticker, model_path, scaler_path),
            daemon=True,
        )
        t.start()

    def _predict_thread(self, ticker, model_path, scaler_path):
        """Background: runs predict_next_week from NN.py."""
        try:
            result = predict_next_week(ticker, model_path, scaler_path)
            self.after(0, self._on_prediction_done, ticker, result)
        except Exception as exc:
            self.after(0, self._on_prediction_error, str(exc))

    def _on_prediction_done(self, ticker, result):
        prob_up   = result["prob_up"]
        going_up  = result["direction_up"]

        # Color-coded direction label
        direction_text  = "▲ UP"  if going_up else "▼ DOWN"
        direction_color = GREEN   if going_up else RED

        self.direction_var.set(direction_text)
        self.direction_label.config(fg=direction_color)
        self.prob_var.set(f"Probability UP: {prob_up:.1%}")
        self.pred_ticker_var.set(f"Ticker: {ticker}")
        self.pred_date_var.set(f"As of: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        self.status_var.set("Idle")
        self._re_enable_buttons()

    def _on_prediction_error(self, msg):
        self.status_var.set("Error")
        self._re_enable_buttons()
        messagebox.showerror("Prediction Error", msg)

    # ── EVALUATION ───────────────────────────────────────────────────────────

    def _start_evaluation(self):
        """Launch all 3 evaluation comparisons in a background thread."""
        ticker = self.selected_ticker.get().strip().upper()
        model_path  = f"{MODEL_DIR}/{ticker}_best.h5"
        scaler_path = f"{MODEL_DIR}/{ticker}_scaler.joblib"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            messagebox.showerror(
                "Model Not Found",
                f"No saved model found for {ticker}.\nPlease train the model first."
            )
            return

        self._disable_buttons()
        self.status_var.set("Evaluating…")

        # Clear previous table rows
        for row in self.eval_tree.get_children():
            self.eval_tree.delete(row)

        t = threading.Thread(
            target=self._eval_thread,
            args=(ticker, model_path, scaler_path),
            daemon=True,
        )
        t.start()

    def _eval_thread(self, ticker, model_path, scaler_path):
        """Background: runs all 3 evaluation functions from NN.py."""
        try:
            # Comparison: LSTM vs logistic regression vs random
            lr_results = logistic_regression_evaluation(ticker, model_path, scaler_path)

            # Single prediction check (logistic)
            evaluate_logistic_regression_single(ticker, model_path, scaler_path)

            # Random guess comparison
            rg_results = random_guess_evaluation(ticker, model_path, scaler_path)

            self.after(0, self._on_eval_done, lr_results, rg_results)
        except Exception as exc:
            self.after(0, self._on_eval_error, str(exc))

    def _on_eval_done(self, lr_results, rg_results):
        """Populate the evaluation table with accuracy figures."""
        rows = [
            ("LSTM Model",          lr_results.get("lstm_accuracy")),
            ("Logistic Regression", lr_results.get("logistic_regression_accuracy")),
            ("Random Guess",        lr_results.get("random_accuracy")),
        ]

        for model_name, acc in rows:
            display = f"{acc:.1%}" if isinstance(acc, float) else "N/A"
            self.eval_tree.insert("", "end", values=(model_name, display))

        self.status_var.set("Idle")
        self._re_enable_buttons()

    def _on_eval_error(self, msg):
        self.status_var.set("Error")
        self._re_enable_buttons()
        messagebox.showerror("Evaluation Error", msg)

    # ── CHART ────────────────────────────────────────────────────────────────

    def _plot_history(self, history):
        """Render loss & accuracy curves from the Keras History object."""
        if history is None:
            return

        h = history.history

        # Clear previous plots
        self.ax_loss.cla()
        self.ax_acc.cla()

        # Loss curves
        self.ax_loss.plot(h["loss"],     label="Train Loss", color=ACCENT, linewidth=1.5)
        self.ax_loss.plot(h["val_loss"], label="Val Loss",   color=RED,    linewidth=1.5, linestyle="--")
        self.ax_loss.legend(fontsize=8, facecolor=SURFACE, labelcolor=FG, edgecolor=BORDER)
        self._style_axes(self.ax_loss, "Loss")

        # Accuracy curves
        self.ax_acc.plot(h["accuracy"],     label="Train Acc", color=GREEN,  linewidth=1.5)
        self.ax_acc.plot(h["val_accuracy"], label="Val Acc",   color=ACCENT, linewidth=1.5, linestyle="--")
        self.ax_acc.legend(fontsize=8, facecolor=SURFACE, labelcolor=FG, edgecolor=BORDER)
        self._style_axes(self.ax_acc, "Accuracy")

        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()   # Refresh the embedded canvas

    # ── REPORT ───────────────────────────────────────────────────────────────

    def _load_report(self, ticker):
        """Load and display the last saved JSON report for a ticker."""
        report_path = f"{MODEL_DIR}/{ticker}_report.json"
        if not os.path.exists(report_path):
            self.report_var.set(f"No report for {ticker}.")
            return
        try:
            with open(report_path) as f:
                data = json.load(f)
            # Format as a compact multi-line string
            lines = [
                f"Ticker:   {data.get('ticker', ticker)}",
                f"Accuracy: {float(data.get('accuracy', 0)):.2%}",
                f"Horizon:  {data.get('horizon', HORIZON)} days",
                f"Date:     {data.get('date', '—')[:19]}",
            ]
            self.report_var.set("\n".join(lines))
        except Exception as e:
            self.report_var.set(f"Could not read report: {e}")

    # ── BUTTON HELPERS ────────────────────────────────────────────────────────

    def _disable_buttons(self):
        self.train_btn.config(state="disabled")
        self.predict_btn.config(state="disabled")
        self.eval_btn.config(state="disabled")

    def _re_enable_buttons(self):
        self.train_btn.config(state="normal")
        self.predict_btn.config(state="normal")
        self.eval_btn.config(state="normal")


# ─────────────────────────────────────────────────────────────────────────────
# Inline training helper (mirrors NN.train_with_random_data but also returns
# the Keras History so the GUI can plot loss/accuracy curves).
# ─────────────────────────────────────────────────────────────────────────────

def _train_and_capture_history(ticker):
    """
    Full training pipeline for a given ticker.
    Returns (ticker, model_path, scaler_path, history).
    The history object is used by the GUI to draw training curves.
    """
    import numpy as np
    from sklearn.utils import class_weight as cw_module

    years = np.random.randint(5, 20)

    feature_cols = [
        "return_1", "return_5",
        "SMA_10", "SMA_21",
        "EMA_20", "RSI_14",
        "BB_width", "vol_pct_change", "Volume",
    ]

    # Download & preprocess
    raw = download_data(ticker, years)
    df  = add_technical_indicators(raw)
    X, y = create_dataset(df, feature_cols)

    # Train / test split with gap to avoid leakage
    split = int(len(X) * (1 - TEST_SIZE))
    X_train, y_train = X[:split - GAP], y[:split - GAP]
    X_test,  y_test  = X[split:],       y[split:]

    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test  = scaler.transform(X_test.reshape(-1,  X_test.shape[-1])).reshape(X_test.shape)

    # Persist scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    scaler_path = f"{MODEL_DIR}/{ticker}_scaler.joblib"
    model_path  = f"{MODEL_DIR}/{ticker}_best.h5"
    joblib.dump(scaler, scaler_path)

    # Class weights to handle imbalanced labels
    cw = cw_module.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(cw))

    # Build and train LSTM
    model = build_lstm_model((SEQ_LEN, X.shape[-1]))
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ],
        verbose=0,   # suppress console spam in GUI mode
    )

    model.save(model_path)

    # Evaluate on held-out test set
    preds = (model.predict(X_test) > 0.5).astype(int)
    acc   = accuracy_score(y_test, preds)

    # Save JSON report
    report = {
        "ticker":   ticker,
        "years":    int(years),
        "accuracy": float(acc),
        "horizon":  HORIZON,
        "date":     datetime.now().isoformat(),
    }
    pd.Series(report).to_json(f"{MODEL_DIR}/{ticker}_report.json")

    return ticker, model_path, scaler_path, history


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = StockPredictorApp()
    app.mainloop()
