"""
=============================================================================
  TIME SERIES ANALYZER  –  Tkinter Desktop Application
=============================================================================
  Purpose : Generate random time-series data, display ACF / PACF plots,
            and recommend the best forecasting model with plain-English
            reasons.

  Models  : AR, MA, ARMA, ARIMA, SARIMA

  Layout  :
    ┌─────────────────────────────────────────────────────────┐
    │  TOP  –  Data Generator Controls                        │
    ├──────────────────────────┬──────────────────────────────┤
    │  MIDDLE-LEFT  –  ACF    │  MIDDLE-RIGHT  –  PACF        │
    ├──────────────────────────┴──────────────────────────────┤
    │  BOTTOM  –  Model Cards (AR│MA│ARMA│ARIMA│SARIMA)       │
    │           + Reasons + Radio Buttons                     │
    │           + Summary Text Box                            │
    └─────────────────────────────────────────────────────────┘

  Usage   : python ts_analyzer.py
  Convert : Use auto-py-to-exe to package as .exe
=============================================================================
"""

# ─── Standard library ───────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# ─── Third-party ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# matplotlib must use the TkAgg backend so figures can embed in Tkinter
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# statsmodels provides the ACF / PACF computation and stationarity test
from statsmodels.tsa.stattools import acf, pacf, adfuller

# ─── App-wide constants ────────────────────────────────────────────────────
# Colours
BG_COLOR        = "#1e1e2e"        # dark background
PANEL_BG        = "#2a2a3c"        # card / panel background
ACCENT          = "#7c3aed"        # purple accent
ACCENT_LIGHT    = "#a78bfa"        # lighter purple for hover
TEXT_COLOR      = "#e2e2f0"        # light text
HIGHLIGHT_BG    = "#3b1f6e"        # highlighted card background
DISABLED_FG     = "#666680"        # greyed-out text
SUCCESS_GREEN   = "#22c55e"        # green for selected model
CARD_BORDER     = "#3d3d56"        # subtle border colour

# List of models we evaluate
MODEL_NAMES = ["AR", "MA", "ARMA", "ARIMA", "SARIMA"]

# ─── Reason / trait flags we track ─────────────────────────────────────────
# Each "reason" is a trait detected in the data.  The indicator next to it
# highlights when the trait is detected.
REASON_KEYS = [
    "stationarity",    # ADF test says non-stationary (p > 0.05)
    "trend",           # data has an upward / downward trend
    "seasonality",     # repeating seasonal pattern exists
    "pacf_cutoff",     # PACF cuts off sharply  → AR signature
    "acf_cutoff",      # ACF cuts off sharply   → MA signature
    "both_decay",      # both ACF & PACF decay gradually → ARMA signature
]

REASON_LABELS = {
    "stationarity": "ADF test: series is non-stationary (p > 0.05)",
    "trend":        "Data has a visible trend (non-stationary)",
    "seasonality":  "Repeating seasonal pattern detected",
    "pacf_cutoff":  "PACF cuts off sharply → AR signature",
    "acf_cutoff":   "ACF cuts off sharply  → MA signature",
    "both_decay":   "Both ACF & PACF decay gradually → mixed AR+MA",
}

# ─── Model equations for display ───────────────────────────────────────────
MODEL_EQUATIONS = {
    "AR":     "y(t) = c + φ₁·y(t-1) + … + φₚ·y(t-p) + ε(t)",
    "MA":     "y(t) = μ + ε(t) + θ₁·ε(t-1) + … + θ_q·ε(t-q)",
    "ARMA":   "y(t) = c + Σφᵢ·y(t-i) + Σθⱼ·ε(t-j) + ε(t)",
    "ARIMA":  "Δᵈy(t) = c + Σφᵢ·Δᵈy(t-i) + Σθⱼ·ε(t-j) + ε(t)",
    "SARIMA": "ARIMA(p,d,q)×(P,D,Q)[s]  with seasonal differencing",
}


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER :  Random Time-Series Generator
# ═══════════════════════════════════════════════════════════════════════════
def generate_random_ts(n_points: int = 200,
                       add_trend: bool = False,
                       add_season: bool = False,
                       noise_level: float = 1.0,
                       season_period: int = 12) -> pd.Series:
    """
    Build a synthetic time series by stacking components:
      1. Base AR(1) process  (autoregressive skeleton)
      2. Optional linear trend
      3. Optional sinusoidal seasonality
      4. Gaussian noise scaled by `noise_level`

    Returns a pandas Series indexed 0 … n_points-1.
    """
    np.random.seed(None)  # fresh randomness each call

    # --- Component 1: AR(1) skeleton ---
    # y[t] = 0.7 * y[t-1] + noise
    ar_coeff = 0.7
    base = np.zeros(n_points)
    for t in range(1, n_points):
        base[t] = ar_coeff * base[t - 1] + np.random.normal(0, 0.5)

    # --- Component 2: Optional trend ---
    trend = np.zeros(n_points)
    if add_trend:
        # gentle linear slope + slight quadratic curvature
        slope = np.random.uniform(0.02, 0.08)
        trend = slope * np.arange(n_points) + 0.0001 * np.arange(n_points) ** 1.3

    # --- Component 3: Optional seasonality ---
    season = np.zeros(n_points)
    if add_season:
        amplitude = np.random.uniform(2.0, 5.0)
        season = amplitude * np.sin(2 * np.pi * np.arange(n_points) / season_period)

    # --- Component 4: Noise ---
    noise = noise_level * np.random.normal(0, 1, n_points)

    # --- Combine ---
    series = base + trend + season + noise
    return pd.Series(series, name="Generated Time Series")


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER :  Analyse ACF / PACF and recommend a model
# ═══════════════════════════════════════════════════════════════════════════
def _detect_cutoff(vals: np.ndarray, threshold: float, max_lags: int):
    """
    Detect whether a correlogram (ACF or PACF) "cuts off" cleanly.

    A true cutoff means:
      • The first few lags (up to some order p or q) are significant
      • The remaining lags are mostly within the confidence band

    Returns (is_cutoff: bool, order: int)
    """
    sig = np.abs(vals[1:]) > threshold  # boolean mask, True = significant
    n = len(sig)
    if n == 0 or not sig.any():
        return False, 0

    # Find the first lag that is NOT significant → candidate cutoff point
    # We require at least 1 significant early lag.
    first_nonsig = np.argmin(sig)  # first False in the mask
    if first_nonsig == 0:
        # Even lag-1 is not significant → no cutoff pattern
        return False, 0

    order = first_nonsig  # number of significant lags before the cutoff

    # After the cutoff, at least 60 % of remaining lags should be
    # inside the band for it to count as a "clean" cutoff
    remaining = sig[order:]
    if len(remaining) == 0:
        return True, order
    fraction_inside = 1 - remaining.mean()  # fraction within the band
    is_cutoff = fraction_inside >= 0.60
    return is_cutoff, order


def _detect_gradual_decay(vals: np.ndarray, threshold: float):
    """
    Detect whether a correlogram decays gradually rather than cutting off.

    Gradual decay means:
      • Many lags are significant
      • The absolute values decrease over time (not sudden drop)

    Returns True if decay pattern is detected.
    """
    abs_vals = np.abs(vals[1:])  # skip lag-0
    n = len(abs_vals)
    if n < 6:
        return False

    sig = abs_vals > threshold
    fraction_significant = sig.mean()

    # "Gradual" = more than 40 % of lags are significant …
    if fraction_significant < 0.40:
        return False

    # … AND the values are trending downward (negative slope)
    x = np.arange(n)
    slope = np.polyfit(x, abs_vals, 1)[0]
    return slope < 0  # decreasing absolute correlation = decay


def _detect_seasonality_from_acf(acf_vals: np.ndarray, threshold: float,
                                  season_period: int):
    """
    Look for recurring spikes in the ACF at multiples of `season_period`.

    Returns (detected: bool, detected_period: int or None)
    """
    # Check lags at 1×, 2×, and 3× the candidate season period
    multiples = [season_period * k for k in range(1, 4)]
    hits = 0
    for lag in multiples:
        if lag < len(acf_vals) and abs(acf_vals[lag]) > threshold:
            hits += 1

    # Also scan for ANY repeating peak if the user-supplied period didn't hit
    auto_period = None
    if hits == 0:
        # Scan lags 2..len/2 for the first significant spike after an
        # initial decay, which could indicate a seasonal cycle.
        for lag in range(2, len(acf_vals)):
            if abs(acf_vals[lag]) > threshold * 1.5:
                # Verify a second spike at 2× this lag
                double = lag * 2
                if double < len(acf_vals) and abs(acf_vals[double]) > threshold:
                    auto_period = lag
                    hits = 2
                    break

    detected = hits >= 1
    period = auto_period if auto_period else (season_period if detected else None)
    return detected, period


def analyse_acf_pacf(series: pd.Series,
                     nlags: int = 40,
                     has_trend: bool = False,
                     has_season: bool = False,
                     season_period: int = 12):
    """
    Full diagnostic pipeline:
      1. ADF stationarity test
      2. Compute ACF and PACF
      3. Detect traits from plot shapes (cutoff, decay, seasonality)
      4. Score each model
      5. Build a rich human-readable summary with equations

    Returns
    -------
    acf_vals   : np.ndarray – ACF values (length nlags+1)
    pacf_vals  : np.ndarray – PACF values (length nlags+1)
    best_model : str        – name of the recommended model
    reasons    : dict       – {reason_key: bool} indicating which traits detected
    summary    : str        – multi-line English explanation
    scores     : dict       – {model_name: float} confidence-like ranking (0-1)
    """

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  Step 1 :  ADF Stationarity Test                                   │
    # │  This is the REAL way to detect trend / non-stationarity.          │
    # │  If p-value > 0.05 → we cannot reject the null hypothesis that     │
    # │  a unit root exists → series is NON-stationary.                    │
    # └─────────────────────────────────────────────────────────────────────┘
    adf_result = adfuller(series, autolag="AIC")
    adf_pvalue = round(adf_result[1], 4)
    is_nonstationary = adf_pvalue > 0.05  # True = non-stationary

    # Trend detection: use ADF result as primary signal,
    # but also honour the user toggle (for synthetic data)
    trend_detected = is_nonstationary or has_trend

    # Suggested differencing order
    d_order = 1 if is_nonstationary else 0

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  Step 2 :  Compute ACF and PACF                                    │
    # └─────────────────────────────────────────────────────────────────────┘
    max_lags = min(nlags, len(series) // 2 - 1)
    if max_lags < 5:
        max_lags = 5

    acf_vals  = acf(series, nlags=max_lags, fft=True)
    pacf_vals = pacf(series, nlags=max_lags, method="ywm")

    # 95 % confidence threshold
    threshold = 1.96 / np.sqrt(len(series))

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  Step 3 :  Detect traits from ACF / PACF shapes                    │
    # └─────────────────────────────────────────────────────────────────────┘

    # -- 3a. PACF cutoff → AR signature --
    pacf_cutoff, p_order = _detect_cutoff(pacf_vals, threshold, max_lags)

    # -- 3b. ACF cutoff → MA signature --
    acf_cutoff, q_order = _detect_cutoff(acf_vals, threshold, max_lags)

    # -- 3c. Gradual decay in both → ARMA signature --
    acf_decays  = _detect_gradual_decay(acf_vals, threshold)
    pacf_decays = _detect_gradual_decay(pacf_vals, threshold)
    both_decay  = acf_decays and pacf_decays

    # -- 3d. Seasonality detection (auto from ACF peaks + user hint) --
    seasonality_detected, detected_period = _detect_seasonality_from_acf(
        acf_vals, threshold, season_period)
    # Also honour the user toggle as a hint
    if has_season:
        seasonality_detected = True
        detected_period = detected_period or season_period

    # ─── Build the reasons dict ──────────────────────────────────────────
    reasons = {
        "stationarity": is_nonstationary,
        "trend":        trend_detected,
        "seasonality":  seasonality_detected,
        "pacf_cutoff":  pacf_cutoff,
        "acf_cutoff":   acf_cutoff,
        "both_decay":   both_decay,
    }

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  Step 4 :  Score each model                                        │
    # │  Weights are heuristic but follow the textbook decision tree:      │
    # │    Stationary?  →  ACF/PACF shape  →  AR / MA / ARMA               │
    # │    Non-stationary?  →  ARIMA   (+ seasonal? → SARIMA)              │
    # └─────────────────────────────────────────────────────────────────────┘
    scores = {m: 0.0 for m in MODEL_NAMES}

    # AR – PACF cuts off, ACF decays, data is stationary
    if pacf_cutoff and not acf_cutoff:
        scores["AR"] += 0.55
    if pacf_cutoff and acf_decays:
        scores["AR"] += 0.15
    if not trend_detected:
        scores["AR"] += 0.15
    if not seasonality_detected:
        scores["AR"] += 0.05

    # MA – ACF cuts off, PACF decays, data is stationary
    if acf_cutoff and not pacf_cutoff:
        scores["MA"] += 0.55
    if acf_cutoff and pacf_decays:
        scores["MA"] += 0.15
    if not trend_detected:
        scores["MA"] += 0.15
    if not seasonality_detected:
        scores["MA"] += 0.05

    # ARMA – both ACF and PACF decay gradually, data is stationary
    if both_decay:
        scores["ARMA"] += 0.55
    if acf_decays and pacf_cutoff:
        scores["ARMA"] += 0.10
    if pacf_decays and acf_cutoff:
        scores["ARMA"] += 0.10
    if not trend_detected:
        scores["ARMA"] += 0.10
    if not seasonality_detected:
        scores["ARMA"] += 0.05

    # ARIMA – series is non-stationary (ADF test), needs differencing
    if is_nonstationary:
        scores["ARIMA"] += 0.50
    if trend_detected:
        scores["ARIMA"] += 0.15
    if is_nonstationary and pacf_cutoff:
        scores["ARIMA"] += 0.10
    if is_nonstationary and both_decay:
        scores["ARIMA"] += 0.10
    if not seasonality_detected:
        scores["ARIMA"] += 0.05

    # SARIMA – seasonality present (with or without trend)
    if seasonality_detected:
        scores["SARIMA"] += 0.50
    if seasonality_detected and trend_detected:
        scores["SARIMA"] += 0.20
    if seasonality_detected and is_nonstationary:
        scores["SARIMA"] += 0.10
    if seasonality_detected and both_decay:
        scores["SARIMA"] += 0.05

    # Normalise to [0, 1]
    max_score = max(scores.values()) if max(scores.values()) > 0 else 1
    scores = {m: round(s / max_score, 2) for m, s in scores.items()}

    # ─── Pick the best model ────────────────────────────────────────────
    best_model = max(scores, key=scores.get)

    # ─── Build suggested order string (e.g. "ARIMA(2,1,0)") ─────────────
    p = p_order if pacf_cutoff else (1 if pacf_decays else 0)
    q = q_order if acf_cutoff  else (1 if acf_decays  else 0)
    d = d_order
    s = detected_period or season_period

    order_str = {
        "AR":     f"AR({p})",
        "MA":     f"MA({q})",
        "ARMA":   f"ARMA({p},{q})",
        "ARIMA":  f"ARIMA({p},{d},{q})",
        "SARIMA": f"SARIMA({p},{d},{q})({1},{1 if is_nonstationary else 0},{1})[{s}]",
    }

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  Step 5 :  Build 3 separate text panels for side-by-side display   │
    # └─────────────────────────────────────────────────────────────────────┘

    # ── Panel 1 : ADF Test + Pattern Analysis ────────────────────────────
    p1 = []
    stationarity_word = "Non-stationary" if is_nonstationary else "Stationary"
    p1.append("═══ Stationarity Test ═══")
    p1.append(f"")
    p1.append(f"ADF p-value : {adf_pvalue}")
    p1.append(f"Result      : {stationarity_word}")
    if is_nonstationary:
        p1.append(f"  → Differencing needed (d={d_order})")
    p1.append("")
    p1.append("═══ Pattern Analysis ═══")
    p1.append("")
    if acf_cutoff:
        p1.append(f"ACF    : Cuts off at lag {q_order}")
        p1.append(f"         → q = {q_order}")
    elif acf_decays:
        p1.append(f"ACF    : Gradual decay")
    else:
        p1.append(f"ACF    : No strong pattern")
    p1.append("")
    if pacf_cutoff:
        p1.append(f"PACF   : Cuts off at lag {p_order}")
        p1.append(f"         → p = {p_order}")
    elif pacf_decays:
        p1.append(f"PACF   : Gradual decay")
    else:
        p1.append(f"PACF   : No strong pattern")
    p1.append("")
    if trend_detected:
        p1.append("Trend  : ● Detected")
    else:
        p1.append("Trend  : ○ Not detected")
    if seasonality_detected:
        p1.append(f"Season : ● Period ≈ {detected_period or season_period}")
    else:
        p1.append("Season : ○ Not detected")

    # ── Panel 2 : Recommendation + Equation ──────────────────────────────
    p2 = []
    p2.append(f"✦  {order_str[best_model]}")
    p2.append("")
    p2.append("Equation:")
    p2.append(f"  {MODEL_EQUATIONS[best_model]}")
    p2.append("")
    p2.append("Why this model?")
    # Add short plain-English reason
    if best_model == "AR":
        p2.append("  PACF cuts off → past values")
        p2.append("  directly influence the future.")
        p2.append("  Data is stationary, no need")
        p2.append("  for differencing.")
    elif best_model == "MA":
        p2.append("  ACF cuts off → shocks persist")
        p2.append(f"  for {q} period(s) then fade.")
        p2.append("  Data is stationary.")
    elif best_model == "ARMA":
        p2.append("  Both ACF and PACF decay")
        p2.append("  gradually → mixed auto-")
        p2.append("  regressive + moving-average.")
    elif best_model == "ARIMA":
        p2.append("  ADF test shows non-stationary")
        p2.append(f"  data → differencing (d={d}) is")
        p2.append("  needed before fitting AR/MA.")
    elif best_model == "SARIMA":
        p2.append("  Seasonal pattern detected in")
        p2.append(f"  ACF at period ≈ {detected_period or season_period}.")
        p2.append("  A seasonal component is needed")
        p2.append("  on top of ARIMA.")

    # ── Panel 3 : Confidence Scores ──────────────────────────────────────
    p3 = []
    for m in MODEL_NAMES:
        pct = int(scores[m] * 100)
        bar = "█" * int(scores[m] * 15)
        empty = "░" * (15 - int(scores[m] * 15))
        p3.append(f"  {order_str[m]}")
        p3.append(f"  {bar}{empty}  {pct:3d}%")
        p3.append("")

    summary = {
        "diagnostics": "\n".join(p1),
        "recommendation": "\n".join(p2),
        "scores": "\n".join(p3),
    }

    return acf_vals, pacf_vals, best_model, reasons, summary, scores


# ═══════════════════════════════════════════════════════════════════════════
#  DIALOG :  Column Picker (shown after loading a CSV / Excel file)
# ═══════════════════════════════════════════════════════════════════════════
class ColumnPickerDialog(tk.Toplevel):
    """
    Modal dialog that shows two dropdown menus:
      1. Date / time column  (optional – user can pick "None")
      2. Value column        (required – the numeric series to analyse)

    After the user clicks OK, the selected column names are stored in
    self.date_col  and  self.value_col.  If the user cancels, both are None.
    """

    def __init__(self, parent, columns: list[str]):
        super().__init__(parent)
        self.title("Select Columns")
        self.configure(bg=BG_COLOR)
        self.resizable(False, False)

        # Centre the dialog over the parent window
        self.transient(parent)
        self.grab_set()  # make it modal

        # Return values
        self.date_col: str | None = None
        self.value_col: str | None = None

        # ── Header ──────────────────────────────────────────────────────
        tk.Label(self, text="Select Time Series Column",
                 bg=BG_COLOR, fg=ACCENT_LIGHT,
                 font=("Segoe UI", 13, "bold")).pack(padx=24, pady=(16, 12))

        # ── Date column dropdown (optional) ─────────────────────────────
        row_date = tk.Frame(self, bg=BG_COLOR)
        row_date.pack(fill="x", padx=24, pady=4)
        tk.Label(row_date, text="Date column (optional):",
                 bg=BG_COLOR, fg=TEXT_COLOR,
                 font=("Segoe UI", 10)).pack(side="left", padx=(0, 8))

        self.var_date = tk.StringVar(value="None")
        date_options = ["None"] + columns
        self.dd_date = ttk.Combobox(row_date, textvariable=self.var_date,
                                    values=date_options, state="readonly", width=25)
        self.dd_date.pack(side="left")

        # ── Value column dropdown (required) ────────────────────────────
        row_val = tk.Frame(self, bg=BG_COLOR)
        row_val.pack(fill="x", padx=24, pady=4)
        tk.Label(row_val, text="Value column (required):",
                 bg=BG_COLOR, fg=TEXT_COLOR,
                 font=("Segoe UI", 10)).pack(side="left", padx=(0, 8))

        # Pre-select the first numeric-looking column if possible
        self.var_value = tk.StringVar(value=columns[0] if columns else "")
        self.dd_value = ttk.Combobox(row_val, textvariable=self.var_value,
                                     values=columns, state="readonly", width=25)
        self.dd_value.pack(side="left")

        # ── OK / Cancel buttons ─────────────────────────────────────────
        btn_row = tk.Frame(self, bg=BG_COLOR)
        btn_row.pack(pady=(16, 16))
        ttk.Button(btn_row, text="OK", style="Accent.TButton",
                   command=self._on_ok).pack(side="left", padx=8)
        ttk.Button(btn_row, text="Cancel",
                   command=self._on_cancel).pack(side="left", padx=8)

        # Wait for the dialog to close before returning control
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.wait_window(self)

    def _on_ok(self):
        """Validate and store the user's selections."""
        val = self.var_value.get()
        if not val:
            messagebox.showwarning("Missing column",
                                   "Please select a value column.",
                                   parent=self)
            return
        self.value_col = val
        date = self.var_date.get()
        self.date_col = None if date == "None" else date
        self.destroy()

    def _on_cancel(self):
        """User cancelled – leave both columns as None."""
        self.date_col = None
        self.value_col = None
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION CLASS
# ═══════════════════════════════════════════════════════════════════════════
class TimeSeriesAnalyzerApp:
    """
    Root Tkinter application.

    The window is divided into three vertical sections:
      1. Top   – data generation controls
      2. Middle – ACF and PACF matplotlib canvases
      3. Bottom – model recommendation cards + summary
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Time Series Analyzer — ACF / PACF Model Recommender")
        self.root.configure(bg=BG_COLOR)
        self.root.minsize(1100, 780)

        # Track the generated data so other methods can access it
        self.current_series: pd.Series | None = None

        # ── Styling (ttk) ───────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use("clam")

        # General frame / label styles
        style.configure("Dark.TFrame",      background=BG_COLOR)
        style.configure("Panel.TFrame",     background=PANEL_BG, relief="flat")
        style.configure("Dark.TLabel",      background=BG_COLOR,  foreground=TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("Panel.TLabel",     background=PANEL_BG,  foreground=TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("Header.TLabel",    background=BG_COLOR,  foreground=ACCENT_LIGHT, font=("Segoe UI", 14, "bold"))
        style.configure("SubHeader.TLabel", background=PANEL_BG,  foreground=ACCENT_LIGHT, font=("Segoe UI", 11, "bold"))

        # Button styles
        style.configure("Accent.TButton",
                        background=ACCENT, foreground="white",
                        font=("Segoe UI", 10, "bold"), padding=(12, 6))
        style.map("Accent.TButton",
                  background=[("active", ACCENT_LIGHT)])

        style.configure("Disabled.TButton",
                        background="#3d3d56", foreground=DISABLED_FG,
                        font=("Segoe UI", 10), padding=(12, 6))

        # Scale (slider) style
        style.configure("TScale", background=BG_COLOR, troughcolor=PANEL_BG)

        # Checkbutton style
        style.configure("Dark.TCheckbutton",
                        background=BG_COLOR, foreground=TEXT_COLOR,
                        font=("Segoe UI", 10))

        # ── Build the three sections ────────────────────────────────────
        self._build_top_section()
        self._build_middle_section()
        self._build_bottom_section()

    # ===================================================================
    #  SECTION 1 :  TOP – Data Generation Controls
    # ===================================================================
    def _build_top_section(self):
        """
        Contains:
          - Title label
          - Number of data-points spinner
          - Trend checkbox
          - Seasonality checkbox
          - Season period spinner
          - Noise level slider
          - "Generate Random Data" button
          - "Import CSV" button (disabled)
        """
        frame = ttk.Frame(self.root, style="Dark.TFrame")
        frame.pack(fill="x", padx=16, pady=(16, 8))

        # ── Title row ───────────────────────────────────────────────────
        ttk.Label(frame, text="📈 Time Series Analyzer",
                  style="Header.TLabel").grid(row=0, column=0, columnspan=8,
                                              sticky="w", pady=(0, 10))

        # ── Controls row ────────────────────────────────────────────────
        ctrl = ttk.Frame(frame, style="Dark.TFrame")
        ctrl.grid(row=1, column=0, columnspan=8, sticky="ew")

        # --- Data points ---
        ttk.Label(ctrl, text="Data Points:", style="Dark.TLabel").grid(
            row=0, column=0, padx=(0, 4), sticky="e")
        self.var_n_points = tk.IntVar(value=200)
        spin_n = ttk.Spinbox(ctrl, from_=50, to=1000, increment=50,
                             textvariable=self.var_n_points, width=7)
        spin_n.grid(row=0, column=1, padx=(0, 16))

        # --- Trend toggle ---
        self.var_trend = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Add Trend", variable=self.var_trend,
                        style="Dark.TCheckbutton").grid(
            row=0, column=2, padx=(0, 16))

        # --- Seasonality toggle ---
        self.var_season = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Add Seasonality", variable=self.var_season,
                        style="Dark.TCheckbutton").grid(
            row=0, column=3, padx=(0, 8))

        # --- Season period ---
        ttk.Label(ctrl, text="Period:", style="Dark.TLabel").grid(
            row=0, column=4, padx=(0, 4), sticky="e")
        self.var_season_period = tk.IntVar(value=12)
        spin_sp = ttk.Spinbox(ctrl, from_=4, to=52, increment=1,
                              textvariable=self.var_season_period, width=5)
        spin_sp.grid(row=0, column=5, padx=(0, 16))

        # --- Noise slider ---
        ttk.Label(ctrl, text="Noise:", style="Dark.TLabel").grid(
            row=0, column=6, padx=(0, 4), sticky="e")
        self.var_noise = tk.DoubleVar(value=1.0)
        noise_slider = ttk.Scale(ctrl, from_=0.0, to=5.0,
                                 variable=self.var_noise, orient="horizontal",
                                 length=120, style="TScale")
        noise_slider.grid(row=0, column=7, padx=(0, 16))

        # ── Buttons row ─────────────────────────────────────────────────
        btn_frame = ttk.Frame(frame, style="Dark.TFrame")
        btn_frame.grid(row=2, column=0, columnspan=8, sticky="w", pady=(10, 0))

        # "Generate" button – triggers data creation + analysis
        ttk.Button(btn_frame, text="⚡  Generate Random Data",
                   style="Accent.TButton",
                   command=self._on_generate).grid(row=0, column=0, padx=(0, 12))

        # "Import CSV / Excel" button – opens a file picker dialog
        ttk.Button(btn_frame, text="📂  Import CSV / Excel",
                   style="Accent.TButton",
                   command=self._on_import).grid(row=0, column=1)

    # ===================================================================
    #  SECTION 2 :  MIDDLE – ACF / PACF Graphs
    # ===================================================================
    def _build_middle_section(self):
        """
        Two matplotlib figures side by side:
          Left  → ACF bar chart
          Right → PACF bar chart

        Figures are embedded using FigureCanvasTkAgg.
        """
        frame = ttk.Frame(self.root, style="Panel.TFrame")
        frame.pack(fill="both", expand=True, padx=16, pady=8)

        # --- ACF figure (left) ---
        self.fig_acf = Figure(figsize=(5, 2.8), dpi=100,
                              facecolor=PANEL_BG)
        self.ax_acf = self.fig_acf.add_subplot(111)
        self._style_axis(self.ax_acf, "ACF (Autocorrelation)")
        self.canvas_acf = FigureCanvasTkAgg(self.fig_acf, master=frame)
        self.canvas_acf.get_tk_widget().pack(side="left", fill="both",
                                             expand=True, padx=(8, 4), pady=8)

        # --- PACF figure (right) ---
        self.fig_pacf = Figure(figsize=(5, 2.8), dpi=100,
                               facecolor=PANEL_BG)
        self.ax_pacf = self.fig_pacf.add_subplot(111)
        self._style_axis(self.ax_pacf, "PACF (Partial Autocorrelation)")
        self.canvas_pacf = FigureCanvasTkAgg(self.fig_pacf, master=frame)
        self.canvas_pacf.get_tk_widget().pack(side="right", fill="both",
                                              expand=True, padx=(4, 8), pady=8)

    @staticmethod
    def _style_axis(ax, title: str):
        """Apply dark-theme styling to a matplotlib Axes."""
        ax.set_facecolor("#22223a")
        ax.set_title(title, color=ACCENT_LIGHT, fontsize=11, fontweight="bold", pad=10)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax.spines["bottom"].set_color(CARD_BORDER)
        ax.spines["left"].set_color(CARD_BORDER)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ===================================================================
    #  SECTION 3 :  BOTTOM – Model Cards + Summary
    # ===================================================================
    def _build_bottom_section(self):
        """
        A horizontal row of "model cards" (one per model), each with:
          - Model name header
          - Recommendation badge (auto-highlighted for the best model)
          - Short confidence score label
        Below the cards: a row of detected-reason indicators, and
        a scrolled text box with the full recommendation summary.
        """
        outer = ttk.Frame(self.root, style="Dark.TFrame")
        outer.pack(fill="x", padx=16, pady=(8, 4))

        ttk.Label(outer, text="Model Recommendations",
                  style="Header.TLabel").pack(anchor="w", pady=(0, 6))

        # ── Model card row ──────────────────────────────────────────────
        cards_frame = ttk.Frame(outer, style="Dark.TFrame")
        cards_frame.pack(fill="x")

        # Keep references to card widgets so we can update colours later
        self.model_cards: dict[str, dict] = {}

        for idx, model in enumerate(MODEL_NAMES):
            # Each card is a tk.Frame (not ttk) so we can set bg colour directly
            card = tk.Frame(cards_frame, bg=PANEL_BG, bd=1,
                            relief="solid", highlightbackground=CARD_BORDER,
                            highlightthickness=1)
            card.grid(row=0, column=idx, padx=6, pady=4, sticky="nsew")
            cards_frame.columnconfigure(idx, weight=1)  # equal column widths

            # Model name
            lbl_name = tk.Label(card, text=model, bg=PANEL_BG, fg=TEXT_COLOR,
                                font=("Segoe UI", 13, "bold"))
            lbl_name.pack(pady=(10, 4))

            # Recommendation badge — hidden by default, shows "✓ RECOMMENDED"
            # only on the best model after analysis
            lbl_badge = tk.Label(card, text="", bg=PANEL_BG, fg=PANEL_BG,
                                 font=("Segoe UI", 9, "bold"))
            lbl_badge.pack(pady=2)

            # Confidence score label (updated when analysis runs)
            lbl_score = tk.Label(card, text="—", bg=PANEL_BG, fg=DISABLED_FG,
                                 font=("Segoe UI", 9))
            lbl_score.pack(pady=(2, 10))

            self.model_cards[model] = {
                "frame": card,
                "label": lbl_name,
                "badge": lbl_badge,
                "score_label": lbl_score,
            }

        # ── Reasons / trait indicators ──────────────────────────────────
        reasons_frame = ttk.Frame(outer, style="Dark.TFrame")
        reasons_frame.pack(fill="x", pady=(10, 4))

        ttk.Label(reasons_frame, text="Detected Traits",
                  style="SubHeader.TLabel",
                  background=BG_COLOR).grid(row=0, column=0, columnspan=len(REASON_KEYS),
                                            sticky="w", pady=(0, 4))

        # One indicator per reason key.  We use IntVar radio-style indicators
        # (read-only) to show which traits are active.
        self.reason_vars: dict[str, tk.IntVar] = {}
        self.reason_labels_widgets: dict[str, tk.Label] = {}

        for idx, key in enumerate(REASON_KEYS):
            var = tk.IntVar(value=0)
            self.reason_vars[key] = var

            # A small coloured circle label acts as the "radio indicator"
            indicator = tk.Label(reasons_frame, text="○", bg=BG_COLOR,
                                 fg=DISABLED_FG, font=("Segoe UI", 12))
            indicator.grid(row=1, column=idx * 2, padx=(8, 2), sticky="e")

            lbl = tk.Label(reasons_frame, text=REASON_LABELS[key],
                           bg=BG_COLOR, fg=DISABLED_FG,
                           font=("Segoe UI", 9), wraplength=180, justify="left")
            lbl.grid(row=1, column=idx * 2 + 1, padx=(0, 12), sticky="w")

            self.reason_labels_widgets[key] = (indicator, lbl)

        # ── Three side-by-side summary panels ────────────────────────────
        # Instead of one big scrolled text, we use 3 compact panels:
        #   Left   = ADF test + pattern analysis
        #   Middle = recommended model + equation + why
        #   Right  = confidence scores for all models
        summary_frame = tk.Frame(outer, bg=BG_COLOR)
        summary_frame.pack(fill="x", pady=(8, 12))

        panel_cfg = dict(
            bg="#1a1a2e", fg=TEXT_COLOR, insertbackground=TEXT_COLOR,
            font=("Consolas", 8), relief="flat", bd=0,
            state="disabled", wrap="word", height=20,
        )

        # Panel 1 – Diagnostics
        f1 = tk.Frame(summary_frame, bg=PANEL_BG, bd=1,
                      highlightbackground=CARD_BORDER, highlightthickness=1)
        f1.pack(side="left", fill="both", expand=True, padx=(0, 4))
        tk.Label(f1, text="Diagnostics", bg=PANEL_BG, fg=ACCENT_LIGHT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=8, pady=(6, 2))
        self.txt_diagnostics = tk.Text(f1, **panel_cfg)
        self.txt_diagnostics.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # Panel 2 – Recommendation
        f2 = tk.Frame(summary_frame, bg=PANEL_BG, bd=1,
                      highlightbackground=CARD_BORDER, highlightthickness=1)
        f2.pack(side="left", fill="both", expand=True, padx=4)
        tk.Label(f2, text="Recommendation", bg=PANEL_BG, fg=ACCENT_LIGHT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=8, pady=(6, 2))
        self.txt_recommendation = tk.Text(f2, **panel_cfg)
        self.txt_recommendation.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # Panel 3 – Confidence Scores
        f3 = tk.Frame(summary_frame, bg=PANEL_BG, bd=1,
                      highlightbackground=CARD_BORDER, highlightthickness=1)
        f3.pack(side="left", fill="both", expand=True, padx=(4, 0))
        tk.Label(f3, text="Confidence Scores", bg=PANEL_BG, fg=ACCENT_LIGHT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=8, pady=(6, 2))
        self.txt_scores = tk.Text(f3, **panel_cfg)
        self.txt_scores.pack(fill="both", expand=True, padx=6, pady=(0, 6))

    # ===================================================================
    #  EVENT HANDLER :  Generate button clicked
    # ===================================================================
    def _on_generate(self):
        """
        1. Read control values
        2. Generate random time-series data
        3. Run the shared analysis + UI-update pipeline
        """
        # ── 1. Read user inputs ─────────────────────────────────────────
        n_points      = self.var_n_points.get()
        add_trend     = self.var_trend.get()
        add_season    = self.var_season.get()
        noise_level   = self.var_noise.get()
        season_period = self.var_season_period.get()

        # ── 2. Generate the data ────────────────────────────────────────
        self.current_series = generate_random_ts(
            n_points=n_points,
            add_trend=add_trend,
            add_season=add_season,
            noise_level=noise_level,
            season_period=season_period,
        )

        # ── 3. Analyse and update the UI ────────────────────────────────
        # For generated data we know the trend/season flags from the toggles
        self._run_analysis(
            has_trend=add_trend,
            has_season=add_season,
            season_period=season_period,
        )

    # ===================================================================
    #  EVENT HANDLER :  Import CSV / Excel button clicked
    # ===================================================================
    def _on_import(self):
        """
        1. Open a file-picker dialog (CSV / Excel)
        2. Load the file into a pandas DataFrame
        3. Show a column-picker popup
        4. Extract the chosen column as a pd.Series
        5. Run the shared analysis + UI-update pipeline
        """
        # ── 1. File picker ──────────────────────────────────────────────
        filepath = filedialog.askopenfilename(
            title="Select a CSV or Excel file",
            filetypes=[
                ("CSV files",   "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("All files",   "*.*"),
            ],
        )
        if not filepath:
            return  # user cancelled

        # ── 2. Load the file ────────────────────────────────────────────
        try:
            if filepath.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath)
        except Exception as e:
            messagebox.showerror("File Error",
                                 f"Could not read file:\n{e}")
            return

        if df.empty:
            messagebox.showwarning("Empty File",
                                   "The selected file contains no data.")
            return

        # ── 3. Column picker popup ──────────────────────────────────────
        dialog = ColumnPickerDialog(self.root, list(df.columns))

        # If user cancelled the dialog, abort
        if dialog.value_col is None:
            return

        # ── 4. Extract the series ───────────────────────────────────────
        try:
            series = pd.to_numeric(df[dialog.value_col], errors="coerce").dropna()
        except Exception as e:
            messagebox.showerror("Column Error",
                                 f"Could not convert column to numbers:\n{e}")
            return

        if len(series) < 10:
            messagebox.showwarning("Too few data points",
                                   "Need at least 10 numeric values to analyse.")
            return

        # If a date column was chosen, try to set it as the index
        if dialog.date_col and dialog.date_col in df.columns:
            try:
                series.index = pd.to_datetime(
                    df.loc[series.index, dialog.date_col])
            except Exception:
                pass  # non-fatal – just keep integer index

        series.name = dialog.value_col
        self.current_series = series.reset_index(drop=True)

        # ── 5. Analyse ──────────────────────────────────────────────────
        # For imported data we don't know trend/season flags upfront,
        # so we pass False and let the ACF/PACF heuristics decide.
        self._run_analysis(
            has_trend=False,
            has_season=False,
            season_period=self.var_season_period.get(),
        )

    # ===================================================================
    #  SHARED :  Run ACF/PACF analysis and update the entire UI
    # ===================================================================
    def _run_analysis(self, *, has_trend: bool, has_season: bool,
                      season_period: int):
        """
        Shared pipeline used by both _on_generate and _on_import:
          1. Compute ACF / PACF + model recommendation
          2. Redraw graphs
          3. Update model cards, reason indicators, and summary text
        """
        # ── 1. Analyse ──────────────────────────────────────────────────
        try:
            acf_vals, pacf_vals, best_model, reasons, summary, scores = \
                analyse_acf_pacf(
                    self.current_series,
                    nlags=40,
                    has_trend=has_trend,
                    has_season=has_season,
                    season_period=season_period,
                )
        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))
            return

        # ── 2a. Update ACF graph ────────────────────────────────────────
        self.ax_acf.clear()
        self._style_axis(self.ax_acf, "ACF (Autocorrelation)")
        lags_acf = np.arange(len(acf_vals))
        self.ax_acf.bar(lags_acf, acf_vals, width=0.4, color=ACCENT, alpha=0.85)
        # Draw 95 % confidence band
        ci = 1.96 / np.sqrt(len(self.current_series))
        self.ax_acf.axhline(y=ci,  color=SUCCESS_GREEN, linestyle="--", linewidth=0.8, alpha=0.6)
        self.ax_acf.axhline(y=-ci, color=SUCCESS_GREEN, linestyle="--", linewidth=0.8, alpha=0.6)
        self.ax_acf.set_xlabel("Lag", color=TEXT_COLOR, fontsize=8)
        self.ax_acf.set_ylabel("ACF", color=TEXT_COLOR, fontsize=8)
        self.fig_acf.tight_layout()
        self.canvas_acf.draw()

        # ── 2b. Update PACF graph ───────────────────────────────────────
        self.ax_pacf.clear()
        self._style_axis(self.ax_pacf, "PACF (Partial Autocorrelation)")
        lags_pacf = np.arange(len(pacf_vals))
        self.ax_pacf.bar(lags_pacf, pacf_vals, width=0.4, color=ACCENT_LIGHT, alpha=0.85)
        self.ax_pacf.axhline(y=ci,  color=SUCCESS_GREEN, linestyle="--", linewidth=0.8, alpha=0.6)
        self.ax_pacf.axhline(y=-ci, color=SUCCESS_GREEN, linestyle="--", linewidth=0.8, alpha=0.6)
        self.ax_pacf.set_xlabel("Lag", color=TEXT_COLOR, fontsize=8)
        self.ax_pacf.set_ylabel("PACF", color=TEXT_COLOR, fontsize=8)
        self.fig_pacf.tight_layout()
        self.canvas_pacf.draw()

        # ── 3a. Update model cards ──────────────────────────────────────
        for model, widgets in self.model_cards.items():
            if model == best_model:
                # Highlight the recommended card with green border + badge
                widgets["frame"].configure(bg=HIGHLIGHT_BG,
                                           highlightbackground=SUCCESS_GREEN,
                                           highlightthickness=2)
                widgets["label"].configure(bg=HIGHLIGHT_BG, fg=SUCCESS_GREEN,
                                           font=("Segoe UI", 13, "bold"))
                widgets["badge"].configure(bg=HIGHLIGHT_BG, fg=SUCCESS_GREEN,
                                           text="✓ RECOMMENDED")
                widgets["score_label"].configure(
                    bg=HIGHLIGHT_BG, fg=SUCCESS_GREEN,
                    text=f"Score: {scores[model]:.2f}  ★")
            else:
                # Reset non-recommended cards to default look
                widgets["frame"].configure(bg=PANEL_BG,
                                           highlightbackground=CARD_BORDER,
                                           highlightthickness=1)
                widgets["label"].configure(bg=PANEL_BG, fg=TEXT_COLOR,
                                           font=("Segoe UI", 13, "bold"))
                widgets["badge"].configure(bg=PANEL_BG, fg=PANEL_BG,
                                           text="")
                widgets["score_label"].configure(
                    bg=PANEL_BG, fg=DISABLED_FG,
                    text=f"Score: {scores[model]:.2f}")

        # ── 3b. Update reason indicators ────────────────────────────────
        for key in REASON_KEYS:
            indicator, lbl = self.reason_labels_widgets[key]
            if reasons.get(key, False):
                # Trait detected → highlight green
                indicator.configure(text="●", fg=SUCCESS_GREEN)
                lbl.configure(fg=SUCCESS_GREEN)
                self.reason_vars[key].set(1)
            else:
                # Trait not detected → grey
                indicator.configure(text="○", fg=DISABLED_FG)
                lbl.configure(fg=DISABLED_FG)
                self.reason_vars[key].set(0)

        # ── 3c. Update the three summary panels ─────────────────────────
        for widget, key in [
            (self.txt_diagnostics,    "diagnostics"),
            (self.txt_recommendation, "recommendation"),
            (self.txt_scores,         "scores"),
        ]:
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", summary[key])
            widget.configure(state="disabled")


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app = TimeSeriesAnalyzerApp(root)
    root.mainloop()
