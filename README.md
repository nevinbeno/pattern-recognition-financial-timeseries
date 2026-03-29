# Pattern Recognition for Financial Time Series Forecasting

Name: Nevin Beno. \n
University Register Number: TCR24CS052

### Second Assignment

---

## Objective

Explore how **time-frequency signal processing** and **deep learning** can be combined to predict stock prices using financial time series data from NSE-listed companies.

---

## Project Structure

```
Assignment 2/
├── Documentation/
│   ├── assignment.pdf                 # Original assignment sheet
│   └── assignment_two_report.docx     # Full written report
├── figures/                           # Auto-generated when script is run
│   ├── fig1_time_series.png
│   ├── fig2_frequency_spectra.png
│   ├── fig3_spectrograms.png
│   ├── fig4_cnn_architecture.png
│   ├── fig5_training_history.png
│   ├── fig6_predictions.png
│   └── fig7_feature_analysis.png
├── src/
│   ├── main.py                        # Main Python script (all 4 tasks)
│   └── stock_cnn_model.keras          # Saved trained CNN model (after first run)
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
└── venv/                              # Virtual environment
```

---

## Requirements

### Python Version
Python 3.8 or higher

### Setup and Activate Virtual Environment
```bash
python3 -m venv venv
```
```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `yfinance` | Download stock data from Yahoo Finance |
| `numpy` | Numerical computations |
| `matplotlib` | Plotting and figure generation |
| `scipy` | FFT and signal processing |
| `tensorflow` | CNN model training |
| `scikit-learn` | Normalisation and evaluation metrics |

---

## How to Run

```bash
cd src
python main.py
```

The script handles everything automatically:

1. Downloads real NSE stock data (TCS, Infosys, Reliance) from Yahoo Finance
2. Normalises and preprocesses the data
3. Computes FFT and STFT spectrograms
4. Trains the CNN model
5. Evaluates predictions and prints metrics to terminal
6. Saves all 7 figures to the `figures/` folder (created automatically)
7. Saves the trained model as `src/stock_cnn_model.keras`

> **Note:** If `yfinance` is unavailable, the script falls back to synthetic data generated via Geometric Brownian Motion. If `tensorflow` is unavailable, Tasks 3 & 4 (CNN training) are skipped — the signal processing figures are still generated.

---

## Methodology

### Signal Representation
Financial time series is treated as a multivariate signal:

```
X(t) = [p(t), r(t), g(t), s(t), d(t)]
```

| Symbol | Meaning |
|---|---|
| `p(t)` | Stock closing price |
| `r(t)` | Revenue |
| `g(t)` | Profit |
| `s(t)` | Market index (Sensex) |
| `d(t)` | USD-INR exchange rate |

### Time-Frequency Transformation (STFT)

Since financial time series are non-stationary, the Short-Time Fourier Transform is applied:

```
STFT(t, f) = ∫ X(τ) w(τ − t) e^{−j2πfτ} dτ
```

Power spectrogram: `S(t, f) = |STFT(t, f)|²`

**Parameters used:**

| Parameter | Value |
|---|---|
| Window Length (L) | 30 trading days (Hanning window) |
| Hop Size (H) | 5 days |
| Overlap | 25 days |
| Prediction Horizon | 5 days ahead |

### Prediction Model (CNN)

```
p̂(t + Δt) = f_θ(S_t)
```

where `f_θ` is the trained CNN and `S_t` is the spectrogram patch at time `t`.

**Architecture:**

```
Input Spectrogram (F × T × 1)
        ↓
Conv2D (32 filters, 3×3, ReLU) → BatchNorm → MaxPool (2×2)
        ↓
Conv2D (64 filters, 3×3, ReLU) → BatchNorm → MaxPool (2×2)
        ↓
Conv2D (128 filters, 3×3, ReLU) → GlobalAveragePooling2D
        ↓
Dense (128, ReLU) → Dropout (0.3) → Dense (64, ReLU)
        ↓
Output: Predicted Price (Linear)
```

---

## Data Sources

| Company | Ticker | Source |
|---|---|---|
| Tata Consultancy Services | TCS.NS | Yahoo Finance |
| Infosys | INFY.NS | Yahoo Finance |
| Reliance Industries | RELIANCE.NS | Yahoo Finance |

**Date range:** January 2020 – January 2024

Other supported sources: [NSE India](https://www.nseindia.com) · [BSE India](https://www.bseindia.com) · [Kaggle](https://www.kaggle.com)

---

## Tasks Covered

| Task | Description | Output Figure |
|---|---|---|
| Task 1 | Data collection, alignment, normalisation | `fig1_time_series.png` |
| Task 2 | FFT frequency analysis | `fig2_frequency_spectra.png` |
| Task 2 | STFT spectrogram generation | `fig3_spectrograms.png` |
| Task 3 | CNN architecture diagram | `fig4_cnn_architecture.png` |
| Task 3 | Model training history | `fig5_training_history.png` |
| Task 4 | Actual vs predicted prices | `fig6_predictions.png` |
| Task 4 | Per-company feature analysis | `fig7_feature_analysis.png` |

---

## Evaluation Metrics

| Metric | Value |
|---|---|
| MSE | 0.00162 |
| RMSE | 0.04025 |
| MAE | 0.03180 |
| R² Score | 0.9412 |

> Actual values will vary slightly each run due to CNN weight initialisation randomness and live market data.

---

## References

1. Y. Zhang and C. Aggarwal, "Stock Market Prediction Using Deep Learning," IEEE Access.
2. A. Tsantekidis et al., "Deep Learning for Financial Time Series Forecasting."
3. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," Neural Computation, 1997.
4. A. Borovykh et al., "Conditional Time Series Forecasting with CNNs."
5. A. Oppenheim and R. Schafer, Discrete-Time Signal Processing, Prentice Hall, 2009.