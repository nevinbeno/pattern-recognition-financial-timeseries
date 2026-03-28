"""
==============================================================================
Pattern Recognition for Financial Time Series Forecasting
Second Assignment - Complete Implementation
All 4 Tasks: Data Preparation, Signal Processing, CNN Model, Analysis
==============================================================================

HOW TO RUN:
    pip install yfinance numpy matplotlib scipy tensorflow scikit-learn
    python main.py

The script will:
  1. Download real stock data (TCS, INFY, RELIANCE) from Yahoo Finance
  2. Generate spectrograms via STFT
  3. Train a CNN model to predict stock prices
  4. Evaluate and visualize results
  5. Save all figures inside the figures/ folder
==============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import stft, spectrogram
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings("ignore")

# ── Try to import real data / deep learning libraries ─────────────────────
try:
    import yfinance as yf
    HAVE_YFINANCE = True
except ImportError:
    HAVE_YFINANCE = False
    print("[INFO] yfinance not found – using synthetic data.")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    HAVE_TF = True
except ImportError:
    HAVE_TF = False
    print("[INFO] TensorFlow not found – CNN section will be skipped.")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
COMPANIES    = ["TCS.NS", "INFY.NS", "RELIANCE.NS"]   # NSE tickers
LABELS       = ["TCS", "Infosys", "Reliance"]
START_DATE   = "2020-01-01"
END_DATE     = "2024-01-01"

WINDOW_LEN   = 30      # STFT window length (trading days)
HOP_SIZE     = 5       # STFT hop size
PRED_HORIZON = 5       # predict 5 days ahead
EPOCHS       = 50
BATCH_SIZE   = 16
TEST_SPLIT   = 0.2

COLORS       = ["#2563EB", "#16A34A", "#DC2626"]
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})

# ── Output directory for all figures ─────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fig_path(filename):
    """Returns full path inside the figures/ folder."""
    return os.path.join(OUTPUT_DIR, filename)

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 – DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_price(n=1000, seed=42, mu=0.0002, sigma=0.015,
                             base=1000.0):
    """
    Geometric Brownian Motion – mirrors real equity price dynamics.
    Used as fallback when yfinance is unavailable.
    """
    rng  = np.random.default_rng(seed)
    dt   = 1
    W    = rng.normal(0, 1, n)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W
    prices = base * np.exp(np.cumsum(log_returns))
    return prices


def load_data():
    """
    Returns dict: { label: np.array of closing prices }
    """
    data = {}
    if HAVE_YFINANCE:
        print("\n[Task 1] Downloading data from Yahoo Finance …")
        for ticker, label in zip(COMPANIES, LABELS):
            try:
                df = yf.download(ticker, start=START_DATE, end=END_DATE,
                                 progress=False)
                prices = df["Close"].values.flatten()
                if len(prices) < 100:
                    raise ValueError("Insufficient data")
                data[label] = prices
                print(f"  ✓ {label}: {len(prices)} trading days")
            except Exception as e:
                print(f"  ✗ {label} failed ({e}), using synthetic data")
                data[label] = generate_synthetic_price(seed=hash(label) % 999)
    else:
        print("\n[Task 1] Generating synthetic price data …")
        seeds = [7, 42, 137]
        bases = [3500, 1500, 2500]
        for label, seed, base in zip(LABELS, seeds, bases):
            data[label] = generate_synthetic_price(seed=seed, base=base)
            print(f"  ✓ {label}: {len(data[label])} synthetic days")

    # ── Normalise each series to [0, 1] ──────────────────────────────────
    scalers = {}
    normed  = {}
    for label, prices in data.items():
        sc = MinMaxScaler()
        normed[label]  = sc.fit_transform(prices.reshape(-1, 1)).flatten()
        scalers[label] = sc

    return data, normed, scalers


def plot_time_series(raw_data):
    """Task 1 Figure – Time series for all 3 companies."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Task 1 – Financial Time Series", fontsize=16, fontweight="bold")
    for ax, (label, prices), color in zip(axes, raw_data.items(), COLORS):
        ax.plot(prices, color=color, linewidth=1.2, label=label)
        ax.set_ylabel("Price (₹)", fontsize=10)
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Trading Days", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_path("fig1_time_series.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved figures/fig1_time_series.png")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 – SIGNAL PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def compute_fft(signal, label=""):
    """Compute single-sided magnitude spectrum."""
    N   = len(signal)
    yf  = np.abs(fft(signal))[:N // 2]
    xf  = fftfreq(N)[:N // 2]
    return xf, yf


def compute_stft_spectrogram(signal, win_len=WINDOW_LEN, hop=HOP_SIZE):
    """
    Manual STFT implementation matching the assignment equations.
    Returns: times (1-D), freqs (1-D), S (2-D power spectrogram).
    """
    n_cols = (len(signal) - win_len) // hop + 1
    S      = np.zeros((win_len // 2 + 1, n_cols))
    window = np.hanning(win_len)
    times  = []

    for i in range(n_cols):
        start    = i * hop
        segment  = signal[start: start + win_len] * window
        spectrum = np.fft.rfft(segment)
        S[:, i]  = np.abs(spectrum) ** 2      # |STFT|² = power spectrogram
        times.append(start + win_len // 2)

    freqs = np.fft.rfftfreq(win_len)
    return np.array(times), freqs, S


def plot_frequency_spectra(normed_data):
    """Task 2 Figure – Frequency domain for all 3 companies."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Task 2 – Frequency Spectrum (FFT)", fontsize=16,
                 fontweight="bold")
    for ax, (label, sig), color in zip(axes, normed_data.items(), COLORS):
        xf, yf = compute_fft(sig, label)
        ax.semilogy(xf[1:], yf[1:], color=color, linewidth=1.0)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Normalised Frequency")
        ax.set_ylabel("Magnitude")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path("fig2_frequency_spectra.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved figures/fig2_frequency_spectra.png")


def plot_spectrograms(normed_data):
    """Task 2 Figure – STFT Spectrogram for all 3 companies."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Task 2 – STFT Spectrograms", fontsize=16, fontweight="bold")
    for ax, (label, sig), color in zip(axes, normed_data.items(), COLORS):
        times, freqs, S = compute_stft_spectrogram(sig)
        S_db = 10 * np.log10(S + 1e-12)          # convert to dB
        img  = ax.pcolormesh(times, freqs, S_db, cmap="inferno", shading="auto")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Trading Day")
        ax.set_ylabel("Normalised Frequency")
        plt.colorbar(img, ax=ax, label="Power (dB)")
    plt.tight_layout()
    plt.savefig(fig_path("fig3_spectrograms.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved figures/fig3_spectrograms.png")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 – CNN MODEL DEVELOPMENT
# ─────────────────────────────────────────────────────────────────────────────

def build_spectrogram_dataset(normed_data):
    """
    For each company, create (X, y) pairs where:
        X[i] = spectrogram patch (freq_bins × time_frames, 1)  → CNN input
        y[i] = normalised price at time + PRED_HORIZON          → regression target
    """
    all_X, all_y = [], []
    patch_shape  = None

    for label, sig in normed_data.items():
        times, freqs, S = compute_stft_spectrogram(sig)
        n_freq = S.shape[0]
        n_time = S.shape[1]

        for t in range(n_time - PRED_HORIZON):
            # Use 10 consecutive STFT frames as the CNN input patch
            if t + 10 > n_time:
                break
            patch = S[:, t: t + 10]           # shape: (freq_bins, 10)
            patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-9)

            # Target: actual price value at future time step
            future_idx = times[min(t + PRED_HORIZON, len(times) - 1)]
            future_idx = min(int(future_idx), len(sig) - 1)
            target     = sig[future_idx]

            all_X.append(patch)
            all_y.append(target)
            patch_shape = patch.shape

    X = np.array(all_X)[..., np.newaxis]   # (N, freq_bins, 10, 1)
    y = np.array(all_y)
    print(f"  Dataset: X={X.shape}, y={y.shape}")
    return X, y, patch_shape


def build_cnn_model(input_shape):
    """
    CNN architecture for spectrogram-based regression.
    pˆ(t + Δt) = f_θ(S_t)
    """
    inp = keras.Input(shape=input_shape, name="spectrogram_input")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      name="conv1")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      name="conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                      name="conv3")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Dense head
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu", name="dense2")(x)
    out = layers.Dense(1, activation="linear", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="StockCNN")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="mse", metrics=["mae"])
    return model


def plot_cnn_architecture():
    """Task 3 Figure – CNN architecture diagram (matplotlib, no graphviz needed)."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.set_title("Task 3 – CNN Architecture for Spectrogram Regression",
                 fontsize=14, fontweight="bold", pad=15)

    blocks = [
        ("Input\nSpectrogram\n(F×T×1)",    0.5,  "#BFDBFE"),
        ("Conv2D\n32 filters\n3×3, ReLU",  2.5,  "#93C5FD"),
        ("MaxPool\n2×2",                    4.5,  "#60A5FA"),
        ("Conv2D\n64 filters\n3×3, ReLU",  6.5,  "#3B82F6"),
        ("MaxPool\n2×2",                    8.0,  "#2563EB"),
        ("Conv2D\n128 filters\n3×3, ReLU", 9.5,  "#1D4ED8"),
        ("Global Avg\nPooling",            11.0,  "#1E40AF"),
        ("Dense 128\n+ Dropout",           12.2,  "#1E3A8A"),
        ("Output\nPrice ŷ",               13.4,  "#172554"),
    ]

    prev_x = None
    for (label, x, color) in blocks:
        w, h = 1.6, 2.5
        rect = plt.Rectangle((x - w/2, 1.25), w, h,
                              facecolor=color, edgecolor="#334155",
                              linewidth=1.5, zorder=3, alpha=0.9,
                              clip_on=False)
        ax.add_patch(rect)
        ax.text(x, 2.5, label, ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold", zorder=4)
        if prev_x is not None:
            ax.annotate("", xy=(x - w/2, 2.5), xytext=(prev_x + w/2, 2.5),
                        arrowprops=dict(arrowstyle="->", color="#334155",
                                        lw=1.5), zorder=5)
        prev_x = x

    plt.tight_layout()
    plt.savefig(fig_path("fig4_cnn_architecture.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved figures/fig4_cnn_architecture.png")


def train_model(X, y):
    """Train/test split and model training."""
    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_cnn_model(X.shape[1:])
    model.summary()

    cb = [
        callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                monitor="val_loss"),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=0)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=cb, verbose=1
    )
    return model, history, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 – ANALYSIS & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history):
    """Task 4 – Training / validation loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Task 4 – Training History", fontsize=14, fontweight="bold")

    ax1.plot(history.history["loss"],     label="Train Loss",  color="#2563EB")
    ax1.plot(history.history["val_loss"], label="Val Loss",    color="#DC2626",
             linestyle="--")
    ax1.set_title("MSE Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history.history["mae"],     label="Train MAE", color="#16A34A")
    ax2.plot(history.history["val_mae"], label="Val MAE",   color="#EA580C",
             linestyle="--")
    ax2.set_title("Mean Absolute Error")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path("fig5_training_history.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved figures/fig5_training_history.png")


def plot_predictions(model, X_test, y_test):
    """Task 4 – Actual vs Predicted scatter + time plot."""
    y_pred = model.predict(X_test, verbose=0).flatten()
    mse    = mean_squared_error(y_test, y_pred)
    rmse   = np.sqrt(mse)
    mae    = np.mean(np.abs(y_test - y_pred))
    r2     = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)

    print(f"\n  ── Evaluation Metrics ──────────────────────────")
    print(f"     MSE  : {mse:.6f}")
    print(f"     RMSE : {rmse:.6f}")
    print(f"     MAE  : {mae:.6f}")
    print(f"     R²   : {r2:.4f}")
    print(f"  ────────────────────────────────────────────────\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Task 4 – Prediction Analysis   |   MSE={mse:.5f}  RMSE={rmse:.5f}  R²={r2:.3f}",
        fontsize=13, fontweight="bold"
    )

    ax1.plot(y_test[:200],  label="Actual",    color="#2563EB", linewidth=1.2)
    ax1.plot(y_pred[:200],  label="Predicted", color="#DC2626", linewidth=1.2,
             linestyle="--")
    ax1.set_title("Actual vs Predicted (first 200 samples)")
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Normalised Price")
    ax1.legend()
    ax1.grid(alpha=0.3)

    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax2.scatter(y_test, y_pred, alpha=0.4, s=10, color="#7C3AED")
    ax2.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    ax2.set_title("Actual vs Predicted (scatter)")
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path("fig6_predictions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved figures/fig6_predictions.png")
    return mse, rmse, mae, r2


def plot_feature_analysis(normed_data):
    """Task 4 – Effect of different companies (features) on spectrogram patterns."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Task 4 – Feature Analysis: Spectrogram Patterns per Company",
                 fontsize=14, fontweight="bold")

    for col, (label, sig), color in zip(range(3), normed_data.items(), COLORS):
        times, freqs, S = compute_stft_spectrogram(sig)
        S_db = 10 * np.log10(S + 1e-12)

        axes[0, col].plot(sig, color=color, linewidth=0.8)
        axes[0, col].set_title(f"{label} – Normalised Price", fontsize=11)
        axes[0, col].set_xlabel("Day")
        axes[0, col].set_ylabel("Normalised Price")
        axes[0, col].grid(alpha=0.3)

        img = axes[1, col].pcolormesh(times, freqs, S_db, cmap="magma",
                                      shading="auto")
        axes[1, col].set_title(f"{label} – Spectrogram", fontsize=11)
        axes[1, col].set_xlabel("Trading Day")
        axes[1, col].set_ylabel("Freq")
        plt.colorbar(img, ax=axes[1, col], label="dB")

    plt.tight_layout()
    plt.savefig(fig_path("fig7_feature_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved figures/fig7_feature_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("  Pattern Recognition for Financial Time Series Forecasting")
    print("=" * 66)
    print(f"\n  Figures will be saved to: {os.path.abspath(OUTPUT_DIR)}/")

    # ── Task 1 ────────────────────────────────────────────────────────────
    print("\n▶  TASK 1: Data Preparation")
    raw_data, normed_data, scalers = load_data()
    plot_time_series(raw_data)

    # ── Task 2 ────────────────────────────────────────────────────────────
    print("\n▶  TASK 2: Signal Processing")
    plot_frequency_spectra(normed_data)
    plot_spectrograms(normed_data)

    # ── Task 3 ────────────────────────────────────────────────────────────
    print("\n▶  TASK 3: CNN Model Development")
    plot_cnn_architecture()

    if HAVE_TF:
        print("\n  Building dataset from spectrograms …")
        X, y, _ = build_spectrogram_dataset(normed_data)

        print("\n  Training CNN …")
        model, history, X_test, y_test = train_model(X, y)
        plot_training_history(history)

        # ── Task 4 ────────────────────────────────────────────────────────
        print("\n▶  TASK 4: Analysis & Evaluation")
        mse, rmse, mae, r2 = plot_predictions(model, X_test, y_test)
        plot_feature_analysis(normed_data)

        model.save("stock_cnn_model.keras")
        print("  → Model saved: stock_cnn_model.keras")
    else:
        print("  [SKIP] TensorFlow not available – install it to run Tasks 3 & 4.")
        print("\n▶  TASK 4: Feature Analysis (no model needed)")
        plot_feature_analysis(normed_data)

    print(f"\n✅  All done!  Figures saved inside figures/:")
    for f in ["fig1_time_series.png", "fig2_frequency_spectra.png",
              "fig3_spectrograms.png", "fig4_cnn_architecture.png",
              "fig5_training_history.png", "fig6_predictions.png",
              "fig7_feature_analysis.png"]:
        print(f"     figures/{f}")

    print("\n" + "=" * 66)


if __name__ == "__main__":
    main()