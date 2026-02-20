# =========================================================
# BTC Forecast – Darts 0.40 – FINAL Script
# =========================================================
"""
Purpose
-------
End-to-end pipeline for multi-horizon Bitcoin price forecasting using Darts (v0.40).

Main steps
----------
1) Load minute-level BTC spot OHLCV data and resample to 1-hour OHLCV.
2) Run exploratory plots on the full dataset range (EDA).
3) Cut the modeling window (with a small buffer for rolling features).
4) Fetch external covariates (funding rate, fear & greed index) for the modeling window.
5) Engineer features (log-price, returns, volatility, RSI, calendar features, lags).
6) Build Darts TimeSeries objects and apply a 70/20/10 time split (train/val/test).
7) Fit scalers on train only (leakage-free), train models (TFT + optional LSTM/GRU + baselines).
8) Rolling evaluation on validation and test (stride = horizon), report RMSE/MAE on price scale.
9) Explainability for TFT (encoder/decoder importance + attention heatmap).

Outputs
-------
- EDA figures, feature dataset export, split metadata, hyperparameters, evaluation metrics,
  horizon error curves (tau curves), and TFT explainability artifacts.

Notes
-----
- Target variable is log(price) to stabilize variance; reported errors are computed on price scale
  via exp(log_close).
- External series are forward-filled to 1-hour frequency to align with the resampled spot series.
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import requests

from pandas.tseries.frequencies import to_offset

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel, BlockRNNModel, NaiveMean, NaiveDrift
from darts.metrics import rmse, mae
from darts.explainability import TFTExplainer

warnings.filterwarnings("ignore")

print("=== START SCRIPT ===", flush=True)
print("PYTHON:", sys.executable, flush=True)
print("CWD:", os.getcwd(), flush=True)
try:
    print("SCRIPT:", os.path.abspath(__file__), flush=True)
except Exception:
    print("SCRIPT: (interactive/no __file__)", flush=True)
time.sleep(0.3)

# =========================================================
# SETTINGS
# =========================================================

# Input/Output paths (adjust to your local environment)
CSV_PATH = r"C:\Users\Nikla\OneDrive\Desktop\Hochschule\HTW\Semester\Wintersemester 2025 2026\Seminar\btcusd_1-min_data.csv"
OUT_DIR  = r"C:\Users\Nikla\OneDrive\Desktop\Hochschule\HTW\Semester\Wintersemester 2025 2026\Seminar\out_clean"
os.makedirs(OUT_DIR, exist_ok=True)

# Reproducibility (controls random initialization and sampling)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# CPU-only execution (restrict threading to reduce overhead on limited machines)
torch.set_num_threads(max(1, os.cpu_count() // 2))

# Frequency definition:
# - raw data: 1-minute
# - modeling frequency: 1-hour (reduces noise and computation)
FREQ        = "1h"
FREQ_OFF    = to_offset(FREQ)

# Model input/output lengths:
# INPUT_CHUNK = L lookback window (history length provided to the model)
# H           = forecast horizon (multi-step prediction length)
INPUT_CHUNK = 48   # L
H           = 24   # forecast horizon

# Chronological split shares (time-based; no shuffling)
TRAIN_SHARE = 0.70
VAL_SHARE   = 0.20
TEST_SHARE  = 0.10

# Modeling window used for the empirical evaluation (with a buffer for rolling features)
START_DATE  = "2020-01-01"
END_DATE    = "2025-11-22"
BUFFER_DAYS = 7

# Training configuration
BATCH_SIZE  = 64
LR          = 1e-3
EARLYSTOP_PATIENCE = 10
MIN_EPOCHS = 12
MAX_EPOCHS = 15

# Target definition:
# - forecast in log space (stabilizes variance)
# - transform back to price for metric reporting
TARGET = "log_close"

# Covariates:
# - Past covariates: observed up to current/past time (available historically)
# - Future covariates: known in advance (calendar features)
PAST_COVARIATES = [
    "Volume",
    "volatility",
    "rsi",
    "funding_rate",
    "fear_greed",
    "funding_change",
    "fear_change",
    "funding_lag6",
    "fear_lag24",
    "log_return",
]
FUTURE_COVARIATES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

# Toggle deep learning baselines (BlockRNN LSTM/GRU) in addition to TFT
RUN_RNNS = True

# =========================================================
# HELPERS
# =========================================================
def print_section(title: str):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def _to_ms(ts) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return int(t.timestamp() * 1000)

def align_to_1h_ffill(df: pd.DataFrame, value_col: str, freq: str = "1h") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Timestamp", value_col])
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.set_index("Timestamp").sort_index().resample(freq).ffill().reset_index()
    return df[["Timestamp", value_col]]

def fetch_binance_funding_rate_usdtm(symbol: str, start_ts, end_ts, limit: int = 1000, pause: float = 0.15) -> pd.DataFrame:
    base_url = "https://fapi.binance.com"
    url = f"{base_url}/fapi/v1/fundingRate"
    start_ms = _to_ms(start_ts)
    end_ms   = _to_ms(end_ts)

    out = []
    cur = start_ms
    while cur < end_ms:
        params = {"symbol": symbol, "startTime": cur, "endTime": end_ms, "limit": limit}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            break
        out.extend(data)
        last_ms = int(data[-1]["fundingTime"])
        cur = last_ms + 1
        time.sleep(pause)

    df = pd.DataFrame(out)
    if df.empty:
        return pd.DataFrame(columns=["Timestamp", "funding_rate"])

    df["Timestamp"] = pd.to_datetime(pd.to_numeric(df["fundingTime"], errors="coerce"), unit="ms", utc=True).dt.tz_convert(None)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df[["Timestamp", "funding_rate"]].dropna().sort_values("Timestamp").reset_index(drop=True)

def fetch_fear_greed_full() -> pd.DataFrame:
    url = "https://api.alternative.me/fng/"
    r = requests.get(url, params={"limit": 0, "format": "json"}, timeout=30)
    r.raise_for_status()
    js = r.json()
    rows = js.get("data", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["Timestamp", "fear_greed"])

    df["fear_greed"] = pd.to_numeric(df["value"], errors="coerce")
    df["Timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="s", utc=True).dt.tz_convert(None)
    df = df[["Timestamp", "fear_greed"]].dropna().sort_values("Timestamp")
    df = df.set_index("Timestamp").resample("1h").ffill().reset_index()
    return df

def save_bar_compare(df_cmp: pd.DataFrame, out_path: str, title: str):
    df_plot = df_cmp.copy().sort_values("RMSE", ascending=True)
    y = np.arange(len(df_plot))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(y - 0.2, df_plot["RMSE"].values, height=0.35, label="RMSE")
    ax.barh(y + 0.2, df_plot["MAE"].values,  height=0.35, label="MAE")

    ax.set_yticks(y)
    ax.set_yticklabels(df_plot["model"].values)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Error (price scale)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
    plt.close(fig)

def horizon_error_curves_from_forecasts(fc_list_log, true_log: TimeSeries, H: int):
    """
    fc_list_log: list of TimeSeries forecasts (log_close), each ~length H
    true_log   : true TimeSeries (log_close)
    Output: (tau_rmse, tau_mae) arrays length H on PRICE scale via exp()
    """
    se = {tau: [] for tau in range(1, H + 1)}
    ae = {tau: [] for tau in range(1, H + 1)}

    true_df = true_log.to_dataframe()
    true_df.columns = ["true_log"]

    for fc in fc_list_log:
        fc_df = fc.to_dataframe()
        fc_df.columns = ["pred_log"]
        joined = true_df.join(fc_df, how="inner").dropna()
        if len(joined) < 1:
            continue

        joined["true_price"] = np.exp(joined["true_log"].values)
        joined["pred_price"] = np.exp(joined["pred_log"].values)
        err = (joined["true_price"].values - joined["pred_price"].values)

        L = min(H, len(err))
        for tau in range(1, L + 1):
            e = err[tau - 1]
            se[tau].append(e * e)
            ae[tau].append(abs(e))

    tau_rmse = np.full(H, np.nan, dtype=float)
    tau_mae  = np.full(H, np.nan, dtype=float)
    for tau in range(1, H + 1):
        if len(se[tau]) > 0:
            tau_rmse[tau - 1] = float(np.sqrt(np.mean(se[tau])))
            tau_mae[tau - 1]  = float(np.mean(ae[tau]))
    return tau_rmse, tau_mae

def plot_tau_curves(tau_rmse, tau_mae, title, out_path, H: int):
    taus = np.arange(1, H + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(taus, tau_rmse, label="RMSE(τ)", linewidth=2)
    ax.plot(taus, tau_mae,  label="MAE(τ)", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(r"Horizon step $\tau$")
    ax.set_ylabel("Error (price scale)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
    plt.close(fig)

def rolling_eval_global(model, name: str, start_time, true_segment_log: TimeSeries,
                        y_full_s: TimeSeries, sc_y: Scaler,
                        p_full_s: TimeSeries, f_ext_s: TimeSeries,
                        H: int, needs_future: bool):
    """
    For TorchForecastingModels (TFT/LSTM/GRU): use retrain=False to avoid val_loss errors.
    """
    hf_kwargs = dict(
        series=y_full_s,
        past_covariates=p_full_s,
        start=start_time,
        forecast_horizon=H,
        stride=H,
        retrain=False,
        last_points_only=False,
        verbose=True,
    )
    if needs_future:
        hf_kwargs["future_covariates"] = f_ext_s

    fc_list_s = model.historical_forecasts(**hf_kwargs)
    fc_list_log = [sc_y.inverse_transform(fc) for fc in fc_list_s]

    true_df = true_segment_log.to_dataframe()
    true_df.columns = ["true_log"]

    pred_df = pd.concat([fc.to_dataframe() for fc in fc_list_log]).sort_index()
    pred_df.columns = ["pred_log"]
    pred_df = pred_df[~pred_df.index.duplicated(keep="first")]

    joined = true_df.join(pred_df, how="inner").dropna()
    joined["true_price"] = np.exp(joined["true_log"].values)
    joined["pred_price"] = np.exp(joined["pred_log"].values)

    r_rmse = float(np.sqrt(np.mean((joined["true_price"] - joined["pred_price"]) ** 2)))
    r_mae  = float(np.mean(np.abs(joined["true_price"] - joined["pred_price"])))

    tau_rmse, tau_mae = horizon_error_curves_from_forecasts(fc_list_log, true_segment_log, H)
    return joined, r_rmse, r_mae, tau_rmse, tau_mae

def rolling_eval_local_baseline(model, name: str, start_time, true_segment_log: TimeSeries,
                                y_full_s: TimeSeries, sc_y: Scaler, H: int):
    """
    For LocalForecastingModels (NaiveMean/NaiveDrift): retrain must be True in Darts.
    (They are fast anyway.)
    """
    hf_kwargs = dict(
        series=y_full_s,
        start=start_time,
        forecast_horizon=H,
        stride=H,
        retrain=True,
        last_points_only=False,
        verbose=False,
    )
    fc_list_s = model.historical_forecasts(**hf_kwargs)
    fc_list_log = [sc_y.inverse_transform(fc) for fc in fc_list_s]

    true_df = true_segment_log.to_dataframe()
    true_df.columns = ["true_log"]

    pred_df = pd.concat([fc.to_dataframe() for fc in fc_list_log]).sort_index()
    pred_df.columns = ["pred_log"]
    pred_df = pred_df[~pred_df.index.duplicated(keep="first")]

    joined = true_df.join(pred_df, how="inner").dropna()
    joined["true_price"] = np.exp(joined["true_log"].values)
    joined["pred_price"] = np.exp(joined["pred_log"].values)

    r_rmse = float(np.sqrt(np.mean((joined["true_price"] - joined["pred_price"]) ** 2)))
    r_mae  = float(np.mean(np.abs(joined["true_price"] - joined["pred_price"])))

    tau_rmse, tau_mae = horizon_error_curves_from_forecasts(fc_list_log, true_segment_log, H)
    return joined, r_rmse, r_mae, tau_rmse, tau_mae


# =========================================================
# 1) LOAD RAW + RESAMPLE SPOT OHLCV (FULL RANGE FOR EDA)
# =========================================================
# ---------------------------------------------------------
# Step 1: Load raw minute data and resample to 1h OHLCV
# - Ensures a consistent hourly time grid
# - Uses OHLC aggregation and sums volume
# - Forward/backward fill to handle occasional missing hours
# ---------------------------------------------------------
print_section("LOAD RAW MINUTE DATA")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(CSV_PATH)

df_raw = pd.read_csv(CSV_PATH)
df_raw["Timestamp"] = pd.to_datetime(df_raw["Timestamp"], unit="s")
df_raw = df_raw.sort_values("Timestamp").reset_index(drop=True)
df_raw = df_raw.replace([np.inf, -np.inf], np.nan).ffill().bfill()

raw_min = df_raw["Timestamp"].min()
raw_max = df_raw["Timestamp"].max()
print("Raw full range:", df_raw.shape, "|", raw_min, "->", raw_max, flush=True)

print_section("RESAMPLE TO 1H (OHLCV) – FULL RANGE (EDA)")
df_spot_1h = (
    df_raw.set_index("Timestamp")
          .resample(FREQ)
          .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
          .dropna()
          .reset_index()
)
df_spot_1h = df_spot_1h.set_index("Timestamp").asfreq(FREQ).ffill().bfill().reset_index()
print("Spot 1h full:", df_spot_1h.shape, "|", df_spot_1h["Timestamp"].min(), "->", df_spot_1h["Timestamp"].max(), flush=True)


# =========================================================
# 2) EDA PLOTS – FULL RANGE (Close MUST be full)
# =========================================================
# ---------------------------------------------------------
# Step 2: EDA on full dataset range
# - Visual sanity checks for Close/Volume/log-price/log-returns
# - Helps detect structural breaks, missing segments, outliers
# ---------------------------------------------------------
print_section("EDA PLOTS – FULL RANGE")

# full range Close
plt.figure(figsize=(12, 4))
plt.plot(df_spot_1h["Timestamp"], df_spot_1h["Close"])
plt.title("BTC Close (1h resampled) – FULL dataset range")
plt.xlabel("Time")
plt.ylabel("Price (USD)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_close_fullrange.png"), dpi=250)
plt.show()
plt.close()

# full range Volume
plt.figure(figsize=(12, 4))
plt.plot(df_spot_1h["Timestamp"], df_spot_1h["Volume"])
plt.title("BTC Volume (1h resampled) – FULL dataset range")
plt.xlabel("Time")
plt.ylabel("Volume")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_volume_fullrange.png"), dpi=250)
plt.show()
plt.close()

# log price + log return (full range)
df_spot_1h["Close"] = df_spot_1h["Close"].clip(lower=1e-8)
df_spot_1h["log_close"]  = np.log(df_spot_1h["Close"])
df_spot_1h["log_return"] = df_spot_1h["log_close"].diff()

plt.figure(figsize=(12, 4))
plt.plot(df_spot_1h["Timestamp"], df_spot_1h["log_close"])
plt.title("BTC log(price) (1h) – FULL dataset range")
plt.xlabel("Time")
plt.ylabel("log(USD)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_log_price_fullrange.png"), dpi=250)
plt.show()
plt.close()

plt.figure(figsize=(12, 4))
plt.plot(df_spot_1h["Timestamp"], df_spot_1h["log_return"])
plt.title("BTC log-return (1h) – FULL dataset range")
plt.xlabel("Time")
plt.ylabel("log-return")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_log_return_fullrange.png"), dpi=250)
plt.show()
plt.close()


# =========================================================
# 3) BUILD MODELING DATASET (CUT WINDOW) + EXTERNALS + FEATURES
# =========================================================
# ---------------------------------------------------------
# Step 3c: Feature engineering
# - Target: log_close and log_return
# - Risk features: rolling volatility, RSI
# - Calendar features: hour/day-of-week (sin/cos encoding)
# - External dynamics: differences and lags
# - Drop rows with NaNs created by rolling/lags to ensure clean training matrix
# ---------------------------------------------------------
print_section("MODEL WINDOW CUT + EXTERNAL FETCH (FAST)")

start = pd.Timestamp(START_DATE)
end   = pd.Timestamp(END_DATE)
end = min(end, df_spot_1h["Timestamp"].max())

min_ts = start - pd.Timedelta(days=BUFFER_DAYS)   # buffer for rolling features
max_ts = end

df_cut = df_spot_1h[(df_spot_1h["Timestamp"] >= min_ts) & (df_spot_1h["Timestamp"] <= max_ts)].copy()
df_cut = df_cut.reset_index(drop=True)
print("Cut+buffer spot 1h:", df_cut.shape, "|", df_cut["Timestamp"].min(), "->", df_cut["Timestamp"].max(), flush=True)

# Fetch externals ONLY for modeling window (no buffer needed, but safe to use cut range)
fetch_start = df_cut["Timestamp"].min()
fetch_end   = df_cut["Timestamp"].max()

print_section("FETCH FUNDING + FEAR&G (ONLY MODEL WINDOW)")

try:
    df_funding = fetch_binance_funding_rate_usdtm("BTCUSDT", fetch_start, fetch_end)
    df_funding_1h = align_to_1h_ffill(df_funding, "funding_rate", freq="1h")
except Exception as e:
    print("Funding fetch failed:", e, flush=True)
    df_funding_1h = pd.DataFrame({"Timestamp": df_cut["Timestamp"], "funding_rate": np.nan})

try:
    df_fng_1h = fetch_fear_greed_full()
    df_fng_1h = df_fng_1h[(df_fng_1h["Timestamp"] >= fetch_start) & (df_fng_1h["Timestamp"] <= fetch_end)].copy()
    df_fng_1h = align_to_1h_ffill(df_fng_1h, "fear_greed", freq="1h")
except Exception as e:
    print("FNG fetch failed:", e, flush=True)
    df_fng_1h = pd.DataFrame({"Timestamp": df_cut["Timestamp"], "fear_greed": np.nan})

df_cut = df_cut.merge(df_funding_1h, on="Timestamp", how="left").merge(df_fng_1h, on="Timestamp", how="left")
df_cut["funding_rate"] = df_cut["funding_rate"].ffill()
df_cut["fear_greed"]   = df_cut["fear_greed"].ffill()

print("funding NaN%:", float(df_cut["funding_rate"].isna().mean()), flush=True)
print("fng     NaN%:", float(df_cut["fear_greed"].isna().mean()), flush=True)

print_section("FEATURE ENGINEERING (MODEL WINDOW)")

df_cut["Close"] = df_cut["Close"].clip(lower=1e-8)
df_cut["log_close"]  = np.log(df_cut["Close"])
df_cut["log_return"] = df_cut["log_close"].diff()

df_cut["returns"]    = df_cut["Close"].pct_change()
df_cut["volatility"] = df_cut["returns"].rolling(24).std()
df_cut["rsi"]        = calculate_rsi(df_cut["Close"], 14)

df_cut["hour"] = df_cut["Timestamp"].dt.hour
df_cut["dow"]  = df_cut["Timestamp"].dt.weekday
df_cut["hour_sin"] = np.sin(2*np.pi*df_cut["hour"]/24)
df_cut["hour_cos"] = np.cos(2*np.pi*df_cut["hour"]/24)
df_cut["dow_sin"]  = np.sin(2*np.pi*df_cut["dow"]/7)
df_cut["dow_cos"]  = np.cos(2*np.pi*df_cut["dow"]/7)

df_cut["funding_change"] = df_cut["funding_rate"].diff()
df_cut["fear_change"]    = df_cut["fear_greed"].diff()
df_cut["funding_lag6"]   = df_cut["funding_rate"].shift(6)
df_cut["fear_lag24"]     = df_cut["fear_greed"].shift(24)

df_cut = df_cut.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

print("df_cut final:", df_cut.shape, "|", df_cut["Timestamp"].min(), "->", df_cut["Timestamp"].max(), flush=True)
df_cut.to_csv(os.path.join(OUT_DIR, "dataset_features_modelwindow.csv"), index=False)

# EDA plots for externals (only available in cut window)
plt.figure(figsize=(12, 4))
plt.plot(df_cut["Timestamp"], df_cut["funding_rate"])
plt.title("BTC Funding Rate (ffill to 1h) – model window")
plt.xlabel("Time")
plt.ylabel("Funding rate")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_funding_rate_modelwindow.png"), dpi=250)
plt.show()
plt.close()

plt.figure(figsize=(12, 4))
plt.plot(df_cut["Timestamp"], df_cut["fear_greed"])
plt.title("Fear & Greed Index (ffill to 1h) – model window")
plt.xlabel("Time")
plt.ylabel("Index value")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_fear_greed_modelwindow.png"), dpi=250)
plt.show()
plt.close()

plt.figure(figsize=(12, 4))
plt.plot(df_cut["Timestamp"], df_cut["volatility"])
plt.title("BTC Volatility (rolling 24h std) – model window")
plt.xlabel("Time")
plt.ylabel("Volatility")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_volatility_modelwindow.png"), dpi=250)
plt.show()
plt.close()

plt.figure(figsize=(12, 4))
plt.plot(df_cut["Timestamp"], df_cut["rsi"])
plt.title("BTC RSI (14) – model window")
plt.xlabel("Time")
plt.ylabel("RSI")
plt.axhline(70, linestyle="--", linewidth=1)
plt.axhline(30, linestyle="--", linewidth=1)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_rsi_modelwindow.png"), dpi=250)
plt.show()
plt.close()

plt.figure(figsize=(12, 4))
plt.plot(df_cut["Timestamp"], df_cut["log_return"])
plt.title("BTC log-return (1h) – model window")
plt.xlabel("Time")
plt.ylabel("log-return")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_log_return_modelwindow.png"), dpi=250)
plt.show()
plt.close()


# =========================================================
# 4) FUTURE COVARIATES EXTENSION (+H) for Multi-Horizon
# =========================================================
print_section("FUTURE COVARIATES EXTENSION (+H)")

last_t = df_cut["Timestamp"].max()
future_index = pd.date_range(last_t + FREQ_OFF, periods=H, freq=FREQ)
df_future = pd.DataFrame({"Timestamp": future_index})
df_future["hour"] = df_future["Timestamp"].dt.hour
df_future["dow"]  = df_future["Timestamp"].dt.weekday
df_future["hour_sin"] = np.sin(2*np.pi*df_future["hour"]/24)
df_future["hour_cos"] = np.cos(2*np.pi*df_future["hour"]/24)
df_future["dow_sin"]  = np.sin(2*np.pi*df_future["dow"]/7)
df_future["dow_cos"]  = np.cos(2*np.pi*df_future["dow"]/7)

df_futcov = pd.concat(
    [df_cut[["Timestamp"] + FUTURE_COVARIATES],
     df_future[["Timestamp"] + FUTURE_COVARIATES]],
    ignore_index=True
).drop_duplicates("Timestamp").sort_values("Timestamp").reset_index(drop=True)


# =========================================================
# 5) BUILD DARTS SERIES + 70/20/10 SPLIT + SCALING (NO LEAKAGE)
# =========================================================
# ---------------------------------------------------------
# Step 5: Convert to Darts TimeSeries and apply chronological split
# - 70/20/10 split
# - Fit scalers on TRAIN only to avoid leakage
# - Transform all segments with train-fitted scalers
# ---------------------------------------------------------
print_section("DARTS SERIES + 70/20/10 SPLIT + SCALING (NO LEAKAGE)")

cols_needed = ["Timestamp", "Close", TARGET] + PAST_COVARIATES + FUTURE_COVARIATES
cols_needed = list(dict.fromkeys(cols_needed))
df_model = df_cut[cols_needed].copy()

y = TimeSeries.from_dataframe(df_model, "Timestamp", [TARGET])               # log_close
p = TimeSeries.from_dataframe(df_model, "Timestamp", PAST_COVARIATES)        # past covariates
f_ext = TimeSeries.from_dataframe(df_futcov, "Timestamp", FUTURE_COVARIATES) # future covariates (+H)

N = len(y)
train_end = int(N * TRAIN_SHARE)
val_end   = int(N * (TRAIN_SHARE + VAL_SHARE))

train_y = y[:train_end]
val_y   = y[train_end:val_end]
test_y  = y[val_end:]

train_p = p[:train_end]
val_p   = p[train_end:val_end]
test_p  = p[val_end:]

print(f"N={N} | train={len(train_y)} | val={len(val_y)} | test={len(test_y)}", flush=True)
print("train:", train_y.start_time(), "->", train_y.end_time(), flush=True)
print("val  :", val_y.start_time(), "->", val_y.end_time(), flush=True)
print("test :", test_y.start_time(), "->", test_y.end_time(), flush=True)

# Fit scalers ONLY on TRAIN (leakage-free)
train_f = f_ext.slice(f_ext.start_time(), train_y.end_time())

sc_y = Scaler().fit(train_y)
sc_p = Scaler().fit(train_p)
sc_f = Scaler().fit(train_f)

y_train_s = sc_y.transform(train_y)
y_val_s   = sc_y.transform(val_y)
y_test_s  = sc_y.transform(test_y)
y_full_s  = sc_y.transform(y)

p_train_s = sc_p.transform(train_p)
p_val_s   = sc_p.transform(val_p)
p_test_s  = sc_p.transform(test_p)
p_full_s  = sc_p.transform(p)

f_ext_s   = sc_f.transform(f_ext)  # includes +H extension

if len(y_val_s) < H:
    raise ValueError(f"Validation-Set zu kurz: {len(y_val_s)} < H={H}")
if len(y_test_s) < H:
    raise ValueError(f"Test-Set zu kurz: {len(y_test_s)} < H={H}")

# Save split info (for thesis text)
split_info = {
    "N": int(N),
    "train_n": int(len(train_y)),
    "val_n": int(len(val_y)),
    "test_n": int(len(test_y)),
    "train_start": str(train_y.start_time()),
    "train_end": str(train_y.end_time()),
    "val_start": str(val_y.start_time()),
    "val_end": str(val_y.end_time()),
    "test_start": str(test_y.start_time()),
    "test_end": str(test_y.end_time()),
}
with open(os.path.join(OUT_DIR, "split_info.json"), "w", encoding="utf-8") as f:
    json.dump(split_info, f, indent=2)


# =========================================================
# 6) EARLY STOPPING SETUP
# =========================================================
print_section("EARLY STOPPING SETUP")
try:
    from pytorch_lightning.callbacks import EarlyStopping
except Exception:
    from lightning.pytorch.callbacks import EarlyStopping

pl_trainer_kwargs = {
    "callbacks": [EarlyStopping(monitor="val_loss", mode="min", patience=EARLYSTOP_PATIENCE)],
    "enable_checkpointing": False,
    "logger": False,
    "min_epochs": MIN_EPOCHS,
}


# =========================================================
# 7) MODELS (same objective)
# =========================================================
# ---------------------------------------------------------
# Step 7–8: Model specification and training
# - TFT: multi-horizon, uses past + future covariates
# - LSTM/GRU: BlockRNN baselines using past covariates only
# - Naive baselines: mean and drift
# - Early stopping on validation loss (val_loss)
# ---------------------------------------------------------
print_section("CREATE MODELS")

loss_fn = torch.nn.MSELoss()

# TFT (deterministic; fairness vs RNNs)
try:
    tft = TFTModel(
        input_chunk_length=INPUT_CHUNK,
        output_chunk_length=H,
        hidden_size=16,
        lstm_layers=1,
        dropout=0.1,
        batch_size=BATCH_SIZE,
        n_epochs=MAX_EPOCHS,
        add_relative_index=True,
        random_state=SEED,
        optimizer_kwargs={"lr": LR},
        likelihood=None,
        loss_fn=loss_fn,
        pl_trainer_kwargs=pl_trainer_kwargs,
    )
except TypeError:
    # fallback if TFTModel in your build doesn't accept loss_fn
    tft = TFTModel(
        input_chunk_length=INPUT_CHUNK,
        output_chunk_length=H,
        hidden_size=16,
        lstm_layers=1,
        dropout=0.1,
        batch_size=BATCH_SIZE,
        n_epochs=MAX_EPOCHS,
        add_relative_index=True,
        random_state=SEED,
        optimizer_kwargs={"lr": LR},
        likelihood=None,
        pl_trainer_kwargs=pl_trainer_kwargs,
    )

if RUN_RNNS:
    lstm = BlockRNNModel(
        model="LSTM",
        input_chunk_length=INPUT_CHUNK,
        output_chunk_length=H,
        hidden_dim=8,
        n_rnn_layers=1,
        dropout=0.1,
        batch_size=BATCH_SIZE,
        n_epochs=MAX_EPOCHS,
        random_state=SEED,
        loss_fn=loss_fn,
        pl_trainer_kwargs=pl_trainer_kwargs,
    )

    gru = BlockRNNModel(
        model="GRU",
        input_chunk_length=INPUT_CHUNK,
        output_chunk_length=H,
        hidden_dim=8,
        n_rnn_layers=1,
        dropout=0.1,
        batch_size=BATCH_SIZE,
        n_epochs=MAX_EPOCHS,
        random_state=SEED,
        loss_fn=loss_fn,
        pl_trainer_kwargs=pl_trainer_kwargs,
    )

naive_mean  = NaiveMean()
naive_drift = NaiveDrift()

# Hyperparameter export
hyperparams = {
    "global": {
        "freq": FREQ,
        "L_input_chunk_length": INPUT_CHUNK,
        "H_forecast_horizon": H,
        "train_share": TRAIN_SHARE,
        "val_share": VAL_SHARE,
        "test_share": TEST_SHARE,
        "target": TARGET,
        "past_covariates": PAST_COVARIATES,
        "future_covariates": FUTURE_COVARIATES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "earlystop_patience": EARLYSTOP_PATIENCE,
        "min_epochs": MIN_EPOCHS,
        "max_epochs": MAX_EPOCHS,
        "seed": SEED,
        "modeling_window": {"start": START_DATE, "end": END_DATE},
        "metrics_reported": "MAE/RMSE on price scale via exp(log_close)",
    },
    "TFT": {
        "hidden_size": 16,
        "lstm_layers": 1,
        "dropout": 0.1,
        "add_relative_index": True,
        "objective": "MSE (deterministic)",
    },
    "LSTM": {
        "hidden_dim": 8,
        "n_rnn_layers": 1,
        "dropout": 0.1,
        "objective": "MSE",
    },
    "GRU": {
        "hidden_dim": 8,
        "n_rnn_layers": 1,
        "dropout": 0.1,
        "objective": "MSE",
    },
    "Baselines": {
        "NaiveMean": "predict mean of train series",
        "NaiveDrift": "drift baseline",
    }
}
with open(os.path.join(OUT_DIR, "hyperparameters.json"), "w", encoding="utf-8") as f:
    json.dump(hyperparams, f, indent=2)
print("Saved hyperparameters ->", os.path.join(OUT_DIR, "hyperparameters.json"), flush=True)


# =========================================================
# 8) FIT MODELS (TRAIN) with VAL for early stopping
# =========================================================
print_section("FIT MODELS (TRAIN + VAL for early stopping)")

val_f_ext = f_ext_s.slice(y_val_s.start_time(), y_val_s.end_time() + (H * FREQ_OFF))

tft.fit(
    series=y_train_s,
    past_covariates=p_train_s,
    future_covariates=f_ext_s,
    val_series=y_val_s,
    val_past_covariates=p_full_s.slice_intersect(y_val_s),
    val_future_covariates=val_f_ext,
    verbose=True,
)

if RUN_RNNS:
    lstm.fit(
        series=y_train_s,
        past_covariates=p_train_s,
        val_series=y_val_s,
        val_past_covariates=p_full_s.slice_intersect(y_val_s),
        verbose=True,
    )
    gru.fit(
        series=y_train_s,
        past_covariates=p_train_s,
        val_series=y_val_s,
        val_past_covariates=p_full_s.slice_intersect(y_val_s),
        verbose=True,
    )

naive_mean.fit(y_train_s)
naive_drift.fit(y_train_s)


# =========================================================
# 9) ROLLING VALIDATION EVALUATION (stride=H) + τ-curves
# =========================================================
# ---------------------------------------------------------
# Step 9–10: Rolling evaluation (stride = H)
# - Produces rolling forecasts over validation/test
# - Computes RMSE/MAE on price scale (exp of log predictions)
# - Computes horizon-wise tau curves (error as a function of τ)
# ---------------------------------------------------------
print_section("VALIDATION EVALUATION (rolling forecasts, stride=H, PRICE scale)")

true_val_log = sc_y.inverse_transform(y_val_s)

val_rows = []

# TFT
joined, v_rmse, v_mae, tau_rmse, tau_mae = rolling_eval_global(
    tft, "TFT",
    start_time=y_val_s.start_time(),
    true_segment_log=true_val_log,
    y_full_s=y_full_s, sc_y=sc_y,
    p_full_s=p_full_s, f_ext_s=f_ext_s,
    H=H, needs_future=True
)
val_rows.append({"model": "TFT", "RMSE": v_rmse, "MAE": v_mae, "n": int(len(joined))})
plot_tau_curves(tau_rmse, tau_mae, "VAL τ-curves – TFT (price scale)", os.path.join(OUT_DIR, "val_tau_curves_TFT.png"), H)

# LSTM/GRU
if RUN_RNNS:
    joined, v_rmse, v_mae, tau_rmse, tau_mae = rolling_eval_global(
        lstm, "LSTM",
        start_time=y_val_s.start_time(),
        true_segment_log=true_val_log,
        y_full_s=y_full_s, sc_y=sc_y,
        p_full_s=p_full_s, f_ext_s=f_ext_s,
        H=H, needs_future=False
    )
    val_rows.append({"model": "LSTM", "RMSE": v_rmse, "MAE": v_mae, "n": int(len(joined))})
    plot_tau_curves(tau_rmse, tau_mae, "VAL τ-curves – LSTM (price scale)", os.path.join(OUT_DIR, "val_tau_curves_LSTM.png"), H)

    joined, v_rmse, v_mae, tau_rmse, tau_mae = rolling_eval_global(
        gru, "GRU",
        start_time=y_val_s.start_time(),
        true_segment_log=true_val_log,
        y_full_s=y_full_s, sc_y=sc_y,
        p_full_s=p_full_s, f_ext_s=f_ext_s,
        H=H, needs_future=False
    )
    val_rows.append({"model": "GRU", "RMSE": v_rmse, "MAE": v_mae, "n": int(len(joined))})
    plot_tau_curves(tau_rmse, tau_mae, "VAL τ-curves – GRU (price scale)", os.path.join(OUT_DIR, "val_tau_curves_GRU.png"), H)

# Naive baselines (Local models -> retrain=True required)
joined, v_rmse, v_mae, tau_rmse, tau_mae = rolling_eval_local_baseline(
    naive_mean, "NaiveMean",
    start_time=y_val_s.start_time(),
    true_segment_log=true_val_log,
    y_full_s=y_full_s, sc_y=sc_y, H=H
)
val_rows.append({"model": "NaiveMean", "RMSE": v_rmse, "MAE": v_mae, "n": int(len(joined))})
plot_tau_curves(tau_rmse, tau_mae, "VAL τ-curves – NaiveMean (price scale)", os.path.join(OUT_DIR, "val_tau_curves_NaiveMean.png"), H)

joined, v_rmse, v_mae, tau_rmse, tau_mae = rolling_eval_local_baseline(
    naive_drift, "NaiveDrift",
    start_time=y_val_s.start_time(),
    true_segment_log=true_val_log,
    y_full_s=y_full_s, sc_y=sc_y, H=H
)
val_rows.append({"model": "NaiveDrift", "RMSE": v_rmse, "MAE": v_mae, "n": int(len(joined))})
plot_tau_curves(tau_rmse, tau_mae, "VAL τ-curves – NaiveDrift (price scale)", os.path.join(OUT_DIR, "val_tau_curves_NaiveDrift.png"), H)

df_val_cmp = pd.DataFrame(val_rows).sort_values("RMSE").reset_index(drop=True)
print("\n=== VAL COMPARISON (rolling, price scale; lower is better) ===", flush=True)
print(df_val_cmp.to_string(index=False), flush=True)
df_val_cmp.to_csv(os.path.join(OUT_DIR, "val_comparison_rolling_price.csv"), index=False)
save_bar_compare(df_val_cmp, os.path.join(OUT_DIR, "val_model_comparison_rmse_mae.png"),
                 "VAL model comparison (rolling; price scale; lower is better)")


# OPTIONAL: readable VAL plot (last 14 days) on PRICE scale for TFT/LSTM/GRU
print_section("VAL PLOT (last 14 days) – PRICE scale")

last_days = 14
true_val_df = true_val_log.to_dataframe()
true_val_df.columns = ["true_log"]
true_val_df["true_price"] = np.exp(true_val_df["true_log"].values)

t0 = true_val_df.index.max() - pd.Timedelta(days=last_days)
val_last = true_val_df.loc[true_val_df.index >= t0].copy()

plt.figure(figsize=(12, 6))
plt.plot(val_last.index, val_last["true_price"], label="True (val price)", linewidth=2)

# predict rolling points (one-shot) from trained models just for plotting
def predict_series_points(model, needs_future: bool):
    preds = model.historical_forecasts(
        series=y_full_s,
        past_covariates=p_full_s,
        future_covariates=f_ext_s if needs_future else None,
        start=y_val_s.start_time(),
        forecast_horizon=H,
        stride=H,
        retrain=False,
        last_points_only=False,
        verbose=False
    )
    preds_log = pd.concat([sc_y.inverse_transform(fc).to_dataframe() for fc in preds]).sort_index()
    preds_log.columns = ["pred_log"]
    preds_log = preds_log[~preds_log.index.duplicated(keep="first")]
    preds_log["pred_price"] = np.exp(preds_log["pred_log"].values)
    return preds_log[["pred_price"]]

tft_pred = predict_series_points(tft, needs_future=True)
plt.plot(tft_pred.loc[tft_pred.index >= t0].index, tft_pred.loc[tft_pred.index >= t0]["pred_price"], label="TFT", linewidth=2)

if RUN_RNNS:
    lstm_pred = predict_series_points(lstm, needs_future=False)
    gru_pred  = predict_series_points(gru, needs_future=False)
    plt.plot(lstm_pred.loc[lstm_pred.index >= t0].index, lstm_pred.loc[lstm_pred.index >= t0]["pred_price"], label="LSTM", linewidth=2)
    plt.plot(gru_pred.loc[gru_pred.index >= t0].index,  gru_pred.loc[gru_pred.index >= t0]["pred_price"],  label="GRU", linewidth=2)

plt.title(f"VAL (last {last_days} days): price forecasts (from log_close)")
plt.xlabel("Time")
plt.ylabel("Price (USD)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"val_price_forecast_last{last_days}d.png"), dpi=250)
plt.show()
plt.close()


# =========================================================
# 10) ROLLING TEST EVALUATION (stride=H) + τ-curves
# =========================================================
print_section("TEST EVALUATION (rolling forecasts, stride=H, PRICE scale)")

true_test_log = sc_y.inverse_transform(y_test_s)

test_rows = []

joined, t_rmse, t_mae, tau_rmse, tau_mae = rolling_eval_global(
    tft, "TFT",
    start_time=y_test_s.start_time(),
    true_segment_log=true_test_log,
    y_full_s=y_full_s, sc_y=sc_y,
    p_full_s=p_full_s, f_ext_s=f_ext_s,
    H=H, needs_future=True
)
test_rows.append({"model": "TFT", "TEST_RMSE": t_rmse, "TEST_MAE": t_mae, "n": int(len(joined))})
plot_tau_curves(tau_rmse, tau_mae, "TEST τ-curves – TFT (price scale)", os.path.join(OUT_DIR, "test_tau_curves_TFT.png"), H)

if RUN_RNNS:
    joined, t_rmse, t_mae, tau_rmse, tau_mae = rolling_eval_global(
        lstm, "LSTM",
        start_time=y_test_s.start_time(),
        true_segment_log=true_test_log,
        y_full_s=y_full_s, sc_y=sc_y,
        p_full_s=p_full_s, f_ext_s=f_ext_s,
        H=H, needs_future=False
    )
    test_rows.append({"model": "LSTM", "TEST_RMSE": t_rmse, "TEST_MAE": t_mae, "n": int(len(joined))})
    plot_tau_curves(tau_rmse, tau_mae, "TEST τ-curves – LSTM (price scale)", os.path.join(OUT_DIR, "test_tau_curves_LSTM.png"), H)

    joined, t_rmse, t_mae, tau_rmse, tau_mae = rolling_eval_global(
        gru, "GRU",
        start_time=y_test_s.start_time(),
        true_segment_log=true_test_log,
        y_full_s=y_full_s, sc_y=sc_y,
        p_full_s=p_full_s, f_ext_s=f_ext_s,
        H=H, needs_future=False
    )
    test_rows.append({"model": "GRU", "TEST_RMSE": t_rmse, "TEST_MAE": t_mae, "n": int(len(joined))})
    plot_tau_curves(tau_rmse, tau_mae, "TEST τ-curves – GRU (price scale)", os.path.join(OUT_DIR, "test_tau_curves_GRU.png"), H)

df_test_cmp = pd.DataFrame(test_rows).sort_values("TEST_RMSE").reset_index(drop=True)
print("\n=== TEST COMPARISON (rolling, price scale; lower is better) ===", flush=True)
print(df_test_cmp.to_string(index=False), flush=True)
df_test_cmp.to_csv(os.path.join(OUT_DIR, "test_comparison_rolling_price.csv"), index=False)


# =========================================================
# 11) EXPLAINABILITY (TFT)
# =========================================================
# ---------------------------------------------------------
# Step 11: Explainability (TFT)
# - Extracts encoder/decoder feature importance and attention weights
# - Saves raw tables and generates top-k plots + attention heatmap
# ---------------------------------------------------------
print_section("EXPLAINABILITY (TFT)")

def _to_df(x):
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        return x
    if hasattr(x, "to_dataframe"):
        return x.to_dataframe()
    raise TypeError(type(x))

EXPLAIN_DAYS = 14
fg_end = y_train_s.end_time() - (H * FREQ_OFF)
fg_start = fg_end - pd.Timedelta(days=EXPLAIN_DAYS)
min_start_fg = y_train_s.time_index[INPUT_CHUNK]
if fg_start < min_start_fg:
    fg_start = min_start_fg

fg_y = y_train_s.slice(fg_start, fg_end)
fg_p = p_train_s.slice_intersect(fg_y)
fg_f = f_ext_s.slice(fg_y.start_time(), fg_end + (H * FREQ_OFF))

explainer = TFTExplainer(tft)
expl = explainer.explain(
    foreground_series=fg_y,
    foreground_past_covariates=fg_p,
    foreground_future_covariates=fg_f,
)

enc_w = expl.get_encoder_importance()
dec_w = expl.get_decoder_importance()
attn  = expl.get_attention()

if isinstance(enc_w, list): enc_w = enc_w[0]
if isinstance(dec_w, list): dec_w = dec_w[0]
if isinstance(attn,  list): attn  = attn[0]

enc_df = _to_df(enc_w)
dec_df = _to_df(dec_w)
att_df = _to_df(attn)

enc_df.to_csv(os.path.join(OUT_DIR, "tft_encoder_importance_raw.csv"))
dec_df.to_csv(os.path.join(OUT_DIR, "tft_decoder_importance_raw.csv"))
att_df.to_csv(os.path.join(OUT_DIR, "tft_attention_raw.csv"))

# Encoder top-15
enc_mean = enc_df.mean(axis=0).sort_values(ascending=False).head(15)
plt.figure(figsize=(10, 5))
plt.barh(enc_mean.index[::-1], enc_mean.values[::-1])
plt.title("TFT Encoder importance (mean over foreground) – Top 15")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tft_encoder_importance_top15.png"), dpi=300)
plt.show()
plt.close()

# Decoder top-15
dec_mean = dec_df.mean(axis=0).sort_values(ascending=False).head(15)
plt.figure(figsize=(10, 5))
plt.barh(dec_mean.index[::-1], dec_mean.values[::-1])
plt.title("TFT Decoder importance (mean over foreground) – Top 15")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tft_decoder_importance_top15.png"), dpi=300)
plt.show()
plt.close()

# Attention heatmap
try:
    A = att_df.values
    plt.figure(figsize=(10, 5))
    plt.imshow(A, aspect="auto")
    plt.title("TFT Attention (foreground) – heatmap")
    plt.xlabel("Decoder step / horizon index")
    plt.ylabel("Encoder time index")
    plt.colorbar(label="Attention weight")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tft_attention_heatmap.png"), dpi=300)
    plt.show()
    plt.close()
except Exception as e:
    print("Attention plot skipped:", e, flush=True)

print_section("DONE")
print("OUT_DIR:", OUT_DIR, flush=True)
