"""Feature Engineering Module for RL-APM.

Calculates technical indicators (SMA, RSI, MACD, Bollinger Bands) and daily returns
for raw equity datasets and stores validated processed datasets for RL agents.
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Configure UTF-8 encoding on Windows to support Unicode glyphs (✓, ✗)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Attempt to import ta; fall back to vectorized pandas/numpy if unavailable
try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

# Resolve project paths using pathlib (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_TICKERS: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
REQUIRED_RAW_COLUMNS: List[str] = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def compute_technical_indicators(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Calculate technical indicators and daily returns on raw market data.

    Features generated:
    - Daily_Return: Percentage change of Adjusted Close price
    - SMA_20: 20-day Simple Moving Average
    - SMA_50: 50-day Simple Moving Average
    - RSI_14: 14-day Relative Strength Index
    - MACD: Moving Average Convergence Divergence (12-day EMA - 26-day EMA)
    - MACD_Signal: 9-day EMA of MACD line
    - MACD_Hist: MACD - MACD_Signal
    - BB_High: Upper Bollinger Band (20-day SMA + 2 standard deviations)
    - BB_Mid: Middle Bollinger Band (20-day SMA)
    - BB_Low: Lower Bollinger Band (20-day SMA - 2 standard deviations)
    - BB_Width: Bollinger Band Width percentage ((BB_High - BB_Low) / BB_Mid * 100)

    Args:
        df: Input DataFrame with required raw columns and a DatetimeIndex.

    Returns:
        Tuple of (processed_df, indicator_metrics_dict).
    """
    df = df.copy()

    # Ensure Date index is clean, datetime-typed, and sorted chronologically
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.sort_index()

    # 1. Daily Percentage Return (on Adjusted Close for corporate action adjustments)
    df["Daily_Return"] = df["Adj Close"].pct_change()

    if HAS_TA:
        # Compute via official ta library
        df["SMA_20"] = ta.trend.SMAIndicator(close=df["Close"], window=20, fillna=False).sma_indicator()
        df["SMA_50"] = ta.trend.SMAIndicator(close=df["Close"], window=50, fillna=False).sma_indicator()
        df["RSI_14"] = ta.momentum.RSIIndicator(close=df["Close"], window=14, fillna=False).rsi()

        macd = ta.trend.MACD(
            close=df["Close"],
            window_fast=12,
            window_slow=26,
            window_sign=9,
            fillna=False,
        )
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Hist"] = macd.macd_diff()

        bollinger = ta.volatility.BollingerBands(
            close=df["Close"],
            window=20,
            window_dev=2,
            fillna=False,
        )
        df["BB_High"] = bollinger.bollinger_hband()
        df["BB_Mid"] = bollinger.bollinger_mavg()
        df["BB_Low"] = bollinger.bollinger_lband()
        df["BB_Width"] = bollinger.bollinger_wband()
    else:
        # Standard native pandas/numpy calculation fallback
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()

        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

        ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema_fast - ema_slow
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        df["BB_Mid"] = df["Close"].rolling(window=20).mean()
        std = df["Close"].rolling(window=20).std(ddof=0)
        df["BB_High"] = df["BB_Mid"] + (2.0 * std)
        df["BB_Low"] = df["BB_Mid"] - (2.0 * std)
        df["BB_Width"] = ((df["BB_High"] - df["BB_Low"]) / df["BB_Mid"]) * 100.0

    initial_len = len(df)
    # In financial time-series RL, rolling lookback windows (e.g. 50-day SMA, 26-day MACD)
    # produce warmup NaNs. Dropping warmup rows ensures no artificial lookahead imputation
    # or zero-padding biases the RL agent's observation space.
    df_clean = df.dropna()
    dropped_warmup = initial_len - len(df_clean)

    metrics = {
        "raw_rows": initial_len,
        "processed_rows": len(df_clean),
        "dropped_warmup": dropped_warmup,
        "remaining_nans": int(df_clean.isna().sum().sum()),
        "first_date": df_clean.index.min().strftime("%Y-%m-%d") if not df_clean.empty else None,
        "last_date": df_clean.index.max().strftime("%Y-%m-%d") if not df_clean.empty else None,
    }

    return df_clean, metrics


def process_single_ticker(
    ticker: str,
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> Tuple[bool, Optional[pd.DataFrame], Dict[str, any], str]:
    """Validate raw CSV, calculate technical indicators, and save processed file.

    Args:
        ticker: Asset ticker symbol.
        raw_dir: Directory containing raw CSV files.
        processed_dir: Target directory for processed CSV files.

    Returns:
        Tuple of (success_bool, processed_df_or_none, quality_metrics, message).
    """
    raw_path = raw_dir / f"{ticker}.csv"
    metrics = {
        "raw_rows": 0,
        "processed_rows": 0,
        "dropped_warmup": 0,
        "remaining_nans": 0,
        "first_date": None,
        "last_date": None,
    }

    if not raw_path.exists():
        return False, None, metrics, f"Raw data file missing at {raw_path}"

    try:
        df_raw = pd.read_csv(raw_path, parse_dates=["Date"], index_col="Date")
        if df_raw.empty:
            return False, None, metrics, f"Raw data file {raw_path.name} is empty"

        missing_cols = [c for c in REQUIRED_RAW_COLUMNS if c not in df_raw.columns]
        if missing_cols:
            return False, None, metrics, f"Raw data missing required columns: {missing_cols}"

        df_processed, feat_metrics = compute_technical_indicators(df_raw)

        if df_processed.empty:
            return False, None, feat_metrics, f"Indicator calculation produced empty dataset for {ticker}"

        # Save processed dataset
        processed_dir.mkdir(parents=True, exist_ok=True)
        target_csv = processed_dir / f"{ticker}_processed.csv"
        df_processed.to_csv(target_csv, index=True)

        return True, df_processed, feat_metrics, f"Saved {len(df_processed)} rows to {target_csv.name}"

    except Exception as exc:
        return False, None, metrics, f"Error processing {ticker}: {exc}"


def process_all_tickers(
    tickers: List[str] = DEFAULT_TICKERS,
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> Tuple[bool, Dict[str, bool], Dict[str, dict]]:
    """Process all available raw datasets and write processed outputs.

    Args:
        tickers: List of ticker symbols.
        raw_dir: Directory containing raw CSVs.
        processed_dir: Directory to save processed CSVs.

    Returns:
        Tuple of (overall_success, ticker_status_dict, quality_metrics_dict).
    """
    ticker_status: Dict[str, bool] = {}
    quality_metrics: Dict[str, dict] = {}
    failure_reasons: Dict[str, str] = {}

    print("\n--- Starting Feature Engineering Pipeline ---")
    for ticker in tickers:
        success, df, metrics, msg = process_single_ticker(ticker, raw_dir, processed_dir)
        ticker_status[ticker] = success
        quality_metrics[ticker] = metrics

        if success:
            print(
                f"[{ticker}] SUCCESS: {metrics['processed_rows']} rows (dropped {metrics['dropped_warmup']} warmup) | "
                f"Range: {metrics['first_date']} to {metrics['last_date']} | "
                f"Remaining NaNs: {metrics['remaining_nans']}"
            )
        else:
            failure_reasons[ticker] = msg
            print(f"[{ticker}] FAILED: {msg}")

    all_success = all(ticker_status.values()) and len(ticker_status) == len(tickers)

    if all_success:
        print("\n========================================")
        print("FEATURE ENGINEERING COMPLETED")
        print("========================================")
        for ticker in tickers:
            print(f"{ticker:<6} ✓")
        print("========================================\n")
    else:
        print("\n========================================")
        print("FEATURE ENGINEERING FAILED")
        print("========================================")
        for ticker in tickers:
            mark = "✓" if ticker_status.get(ticker, False) else "✗"
            print(f"{ticker:<6} {mark}")
        print("========================================")
        print("Failure details:")
        for ticker, reason in failure_reasons.items():
            print(f" - {ticker}: {reason}")
        print("========================================\n")

    return all_success, ticker_status, quality_metrics


if __name__ == "__main__":
    import sys

    success, statuses, metrics = process_all_tickers()
    if not success:
        sys.exit(1)
