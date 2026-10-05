"""Master Market Data Pipeline Orchestrator for RL-APM.

Executes sequential end-to-end data acquisition and feature engineering with
rigorous step-by-step validation, diagnostics, and quality reporting.

Workflow:
  STEP 1: Environment & Dependency Validation
  STEP 2: Yahoo Finance Connectivity Test
  STEP 3: Download Raw Market Data
  STEP 4: Validate Downloaded Raw Files
  STEP 5: Feature Engineering & Technical Indicator Calculation
  STEP 6: Validate Processed Feature Files
  STEP 7: Final Comprehensive Pipeline Summary & Exit Code
"""

from pathlib import Path
import sys
from typing import Dict, List, Tuple

# Configure UTF-8 encoding on Windows to support Unicode glyphs (✓, ✗)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path for relative package imports
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.data.download_market_data import (
    DEFAULT_TICKERS,
    RAW_DATA_DIR,
    REQUIRED_COLUMNS,
    download_all_market_data,
    get_robust_yfinance_session,
    test_yahoo_connectivity,
)
from backend.app.data.feature_engineering import (
    PROCESSED_DATA_DIR,
    process_all_tickers,
)


def validate_environment() -> Tuple[bool, Dict[str, str]]:
    """Validate that all required runtime packages and versions are available.

    Returns:
        Tuple of (is_valid, environment_info_dict).
    """
    env_info: Dict[str, str] = {
        "Python": sys.version.split()[0],
    }

    try:
        import pandas as pd
        env_info["pandas"] = pd.__version__
    except ImportError:
        env_info["pandas"] = "MISSING"

    try:
        import numpy as np
        env_info["numpy"] = np.__version__
    except ImportError:
        env_info["numpy"] = "MISSING"

    try:
        import yfinance as yf
        env_info["yfinance"] = yf.__version__
    except ImportError:
        env_info["yfinance"] = "MISSING"

    try:
        import curl_cffi
        env_info["curl_cffi"] = curl_cffi.__version__
    except ImportError:
        env_info["curl_cffi"] = "Not Installed"

    try:
        import ta
        env_info["ta"] = getattr(ta, "__version__", "Installed")
    except ImportError:
        env_info["ta"] = "Using Pandas Fallback"

    # Verify critical dependencies
    critical_missing = [pkg for pkg in ["pandas", "numpy", "yfinance"] if env_info.get(pkg) == "MISSING"]
    is_valid = len(critical_missing) == 0

    return is_valid, env_info


def validate_raw_files(
    tickers: List[str] = DEFAULT_TICKERS,
    raw_dir: Path = RAW_DATA_DIR,
) -> Tuple[bool, Dict[str, bool], Dict[str, str]]:
    """Verify that all raw CSV files exist and satisfy schema requirements.

    Returns:
        Tuple of (all_valid, status_dict, details_dict).
    """
    import pandas as pd

    status: Dict[str, bool] = {}
    details: Dict[str, str] = {}

    for ticker in tickers:
        file_path = raw_dir / f"{ticker}.csv"
        if not file_path.exists():
            status[ticker] = False
            details[ticker] = f"File missing: {file_path.name}"
            continue

        try:
            df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
            if df.empty:
                status[ticker] = False
                details[ticker] = "File is empty"
                continue

            missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing_cols:
                status[ticker] = False
                details[ticker] = f"Missing columns: {missing_cols}"
                continue

            status[ticker] = True
            details[ticker] = f"{len(df)} valid rows ({df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')})"

        except Exception as exc:
            status[ticker] = False
            details[ticker] = f"Read error: {exc}"

    all_valid = all(status.values()) and len(status) == len(tickers)
    return all_valid, status, details


def validate_processed_files(
    tickers: List[str] = DEFAULT_TICKERS,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> Tuple[bool, Dict[str, bool], Dict[str, str]]:
    """Verify that all processed CSV files exist with zero remaining NaNs.

    Returns:
        Tuple of (all_valid, status_dict, details_dict).
    """
    import pandas as pd

    required_features = [
        "Adj Close", "Close", "Daily_Return", "SMA_20", "SMA_50",
        "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
        "BB_High", "BB_Mid", "BB_Low", "BB_Width"
    ]

    status: Dict[str, bool] = {}
    details: Dict[str, str] = {}

    for ticker in tickers:
        file_path = processed_dir / f"{ticker}_processed.csv"
        if not file_path.exists():
            status[ticker] = False
            details[ticker] = f"File missing: {file_path.name}"
            continue

        try:
            df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
            if df.empty:
                status[ticker] = False
                details[ticker] = "File is empty"
                continue

            missing_cols = [c for c in required_features if c not in df.columns]
            if missing_cols:
                status[ticker] = False
                details[ticker] = f"Missing technical indicators: {missing_cols}"
                continue

            nan_count = int(df.isna().sum().sum())
            if nan_count > 0:
                status[ticker] = False
                details[ticker] = f"Contains {nan_count} unhandled NaN values"
                continue

            status[ticker] = True
            details[ticker] = f"{len(df)} rows, 0 NaNs ({df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')})"

        except Exception as exc:
            status[ticker] = False
            details[ticker] = f"Read error: {exc}"

    all_valid = all(status.values()) and len(status) == len(tickers)
    return all_valid, status, details


def run_pipeline() -> bool:
    """Execute the end-to-end data pipeline with validation at every stage.

    Returns:
        bool: True if all stages succeeded, False otherwise.
    """
    print("=" * 60)
    print("       RL-APM: REAL MARKET DATA & FEATURE PIPELINE")
    print("=" * 60)

    # -------------------------------------------------------------
    # STEP 1: Environment & Dependency Validation
    # -------------------------------------------------------------
    print("\n[STEP 1/6] Validating Runtime Environment...")
    env_valid, env_info = validate_environment()
    for k, v in env_info.items():
        print(f"  - {k:<12}: {v}")

    if not env_valid:
        print("\nERROR: Critical dependencies missing. Cannot proceed.")
        return False
    print("  => Environment validation PASSED.")

    # -------------------------------------------------------------
    # STEP 2: Yahoo Finance Connectivity Test
    # -------------------------------------------------------------
    print("\n[STEP 2/6] Testing Yahoo Finance Live Connectivity...")
    session = get_robust_yfinance_session()
    conn_ok, conn_msg = test_yahoo_connectivity(session)
    print(f"  - Connectivity status: {'SUCCESS' if conn_ok else 'FAILED'}")
    print(f"  - Details: {conn_msg}")

    if not conn_ok:
        print("\nERROR: Cannot connect to Yahoo Finance. Aborting pipeline.")
        return False
    print("  => Connectivity test PASSED.")

    # -------------------------------------------------------------
    # STEP 3: Download Raw Market Data
    # -------------------------------------------------------------
    print("\n[STEP 3/6] Downloading Raw Market Data (2015-01-01 to 2024-12-31)...")
    raw_dl_ok, raw_statuses, raw_metrics = download_all_market_data(
        tickers=DEFAULT_TICKERS,
        start="2015-01-01",
        end="2025-01-01",
        output_dir=RAW_DATA_DIR,
    )

    if not raw_dl_ok:
        print("\nERROR: One or more raw market datasets failed to download.")
        return False

    # -------------------------------------------------------------
    # STEP 4: Validate Downloaded Raw Files
    # -------------------------------------------------------------
    print("\n[STEP 4/6] Validating Downloaded Raw CSV Files on Disk...")
    raw_files_ok, raw_file_statuses, raw_file_details = validate_raw_files(DEFAULT_TICKERS, RAW_DATA_DIR)
    for ticker, ok in raw_file_statuses.items():
        mark = "✓" if ok else "✗"
        print(f"  - {ticker:<6} {mark} : {raw_file_details[ticker]}")

    if not raw_files_ok:
        print("\nERROR: Raw market data file validation failed.")
        return False
    print("  => Raw data file validation PASSED.")

    # -------------------------------------------------------------
    # STEP 5: Feature Engineering
    # -------------------------------------------------------------
    print("\n[STEP 5/6] Executing Feature Engineering & Indicator Calculations...")
    feat_ok, feat_statuses, feat_metrics = process_all_tickers(
        tickers=DEFAULT_TICKERS,
        raw_dir=RAW_DATA_DIR,
        processed_dir=PROCESSED_DATA_DIR,
    )

    if not feat_ok:
        print("\nERROR: Feature engineering failed for one or more tickers.")
        return False

    # -------------------------------------------------------------
    # STEP 6: Validate Processed Feature Files
    # -------------------------------------------------------------
    print("\n[STEP 6/6] Validating Processed Feature Datasets on Disk...")
    proc_files_ok, proc_statuses, proc_details = validate_processed_files(DEFAULT_TICKERS, PROCESSED_DATA_DIR)
    for ticker, ok in proc_statuses.items():
        mark = "✓" if ok else "✗"
        print(f"  - {ticker:<6} {mark} : {proc_details[ticker]}")

    if not proc_files_ok:
        print("\nERROR: Processed feature dataset validation failed.")
        return False
    print("  => Processed feature validation PASSED.")

    # -------------------------------------------------------------
    # STEP 7: Print Final Summary
    # -------------------------------------------------------------
    overall_success = raw_dl_ok and raw_files_ok and feat_ok and proc_files_ok

    print("\n" + "=" * 40)
    print("RL-APM DATA PIPELINE")
    print("=" * 40)
    print("\nEnvironment:")
    print(f"Python:    {env_info.get('Python')}")
    print(f"yfinance:  {env_info.get('yfinance')}")
    print(f"curl_cffi: {env_info.get('curl_cffi')}")
    print(f"pandas:    {env_info.get('pandas')}")
    print(f"numpy:     {env_info.get('numpy')}")

    print("\nMarket data:")
    for ticker in DEFAULT_TICKERS:
        mark = "✓" if raw_file_statuses.get(ticker, False) else "✗"
        print(f"{ticker:<6} {mark}")

    print("\nProcessed data:")
    for ticker in DEFAULT_TICKERS:
        mark = "✓" if proc_statuses.get(ticker, False) else "✗"
        print(f"{ticker:<6} {mark}")

    print("\nStatus:")
    if overall_success:
        print("SUCCESS")
    else:
        print("FAILED")
    print("=" * 40 + "\n")

    return overall_success


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
