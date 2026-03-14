# 📈 Orima — Time Series Analyzer

A desktop application built with **Python + Tkinter** that generates or imports time-series data, visualizes **ACF / PACF** plots, and recommends the best forecasting model with plain-English explanations.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Random Data Generator** | Create synthetic time series with configurable trend, seasonality, noise, and data points |
| **CSV / Excel Import** | Load your own `.csv`, `.xlsx`, or `.xls` files and pick the column to analyse |
| **ACF & PACF Plots** | Interactive matplotlib charts embedded in the app with 95% confidence bands |
| **Model Recommendation** | Rule-based engine scores 5 models and highlights the best fit |
| **Trait Detection** | Visual indicators show detected data traits (trend, seasonality, AR/MA signatures) |
| **Dark Theme** | Modern dark UI with purple accent colours |

### Models Evaluated

- **AR** — Autoregressive
- **MA** — Moving Average
- **ARMA** — Autoregressive Moving Average
- **ARIMA** — Autoregressive Integrated Moving Average
- **SARIMA** — Seasonal ARIMA

---

## 🖥️ App Layout

```
┌─────────────────────────────────────────────────────┐
│  TOP  —  Data Generator Controls / Import Button    │
├──────────────────────────┬──────────────────────────┤
│  ACF Graph               │  PACF Graph              │
├──────────────────────────┴──────────────────────────┤
│  Model Cards  (AR │ MA │ ARMA │ ARIMA │ SARIMA)     │
│  Detected Traits  (● / ○ indicators)                │
│  Summary Text Box                                   │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Tkinter (included with standard Python)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Orima.git
cd Orima

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
python ts_analyzer.py
```

### Build as .exe (Optional)

```bash
pip install auto-py-to-exe
auto-py-to-exe
```

Then select `ts_analyzer.py` as the script and configure your build settings.

---

## 📂 Project Structure

```
Orima/
├── ts_analyzer.py    # Main application (single file)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## 🔧 How It Works

### 1. Data Input
- **Generate**: Configure data points, trend, seasonality, noise level → click "Generate Random Data"
- **Import**: Click "Import CSV / Excel" → select file → pick the value column (and optionally a date column)

### 2. ACF / PACF Analysis
The app computes the **Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)** using `statsmodels`. The green dashed lines on the plots represent the 95% confidence interval.

### 3. Model Recommendation
A rule-based scoring engine analyses the shape of the ACF/PACF plots:

| Signal | Interpretation | Model |
|--------|---------------|-------|
| PACF cuts off, ACF decays | Autoregressive behaviour | **AR** |
| ACF cuts off, PACF decays | Moving-average behaviour | **MA** |
| Both decay gradually | Mixed AR + MA components | **ARMA** |
| Trend detected (non-stationary) | Differencing needed | **ARIMA** |
| Seasonal spikes in ACF | Repeating seasonal pattern | **SARIMA** |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computations |
| `pandas` | Data handling (Series, DataFrame) |
| `matplotlib` | ACF / PACF chart rendering |
| `statsmodels` | ACF and PACF computation |
| `openpyxl` | Excel file reading support |

---

## 📄 License

This project is licensed under the MIT License.
