"""Market Data Ingestion Module for RL-APM.

Downloads authentic historical daily market data from Yahoo Finance for specified assets
and stores validated datasets as raw CSV files in data/raw/.
"""

from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf

# Configure UTF-8 encoding on Windows to support Unicode glyphs (✓, ✗)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Resolve project paths using pathlib (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Target configuration
DEFAULT_TICKERS: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
START_DATE: str = "2015-01-01"
# In yfinance, end date is exclusive. Using 2025-01-01 includes all dates through 2024-12-31.
END_DATE: str = "2025-01-01"
REQUIRED_COLUMNS: List[str] = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


class SNIAdapter(HTTPAdapter):
    """Custom HTTPAdapter that sets server_hostname for TLS ClientHello SNI.

    Yahoo's CDN edge servers in certain regions/ISPs experience middlebox TCP resets
    when SNI explicitly matches '*.finance.yahoo.com'. Because Yahoo's SSL certificate
    is a multi-domain wildcard valid for '*.yahoo.com', establishing TLS with SNI
    'media.yahoo.com' succeeds across middleboxes while HTTP 'Host: query1.finance.yahoo.com'
    is preserved and routed normally.
    """

    def __init__(self, sni_host: str = "media.yahoo.com", *args, **kwargs):
        self.sni_host = sni_host
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs["server_hostname"] = self.sni_host
        return super().init_poolmanager(*args, **kwargs)


def get_robust_yfinance_session() -> requests.Session:
    """Construct a resilient requests Session with retries and SNI configuration.

    Returns:
        Configured requests.Session instance for yfinance.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=4,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = SNIAdapter("media.yahoo.com", max_retries=retry_strategy)
    session.mount("https://query1.finance.yahoo.com", adapter)
    session.mount("https://query2.finance.yahoo.com", adapter)
    session.mount("https://fc.yahoo.com", adapter)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    })
    return session


def test_yahoo_connectivity(session: Optional[requests.Session] = None) -> Tuple[bool, str]:
    """Test connectivity to Yahoo Finance with a 5-day sample request.

    Args:
        session: Configured requests Session.

    Returns:
        Tuple of (success_bool, diagnostic_message).
    """
    if session is None:
        session = get_robust_yfinance_session()

    try:
        test_df = yf.download(
            "AAPL",
            period="5d",
            auto_adjust=False,
            session=session,
            progress=False,
        )
        if test_df is not None and not test_df.empty and len(test_df) >= 3:
            return True, f"Successfully retrieved {len(test_df)} sample test bars for AAPL."
        return False, "Query completed but returned insufficient or empty test data."
    except Exception as exc:
        return False, f"Connectivity test exception: {exc}"


def download_single_ticker(
    ticker: str,
    start: str = START_DATE,
    end: str = END_DATE,
    output_dir: Path = RAW_DATA_DIR,
    session: Optional[requests.Session] = None,
    max_attempts: int = 3,
) -> Tuple[bool, Optional[pd.DataFrame], Dict[str, any], str]:
    """Download, validate, and save daily historical market data for a single ticker.

    Args:
        ticker: Equity ticker symbol (e.g. 'AAPL').
        start: Start date string (YYYY-MM-DD).
        end: End date string (YYYY-MM-DD, exclusive).
        output_dir: Target directory for raw CSV output.
        session: Active requests session.
        max_attempts: Number of retry attempts.

    Returns:
        Tuple of (success_bool, dataframe_or_none, quality_dict, message).
    """
    if session is None:
        session = get_robust_yfinance_session()

    quality = {
        "rows": 0,
        "first_date": None,
        "last_date": None,
        "missing_values": 0,
        "duplicate_dates": 0,
    }

    df: Optional[pd.DataFrame] = None
    last_error: str = ""

    for attempt in range(1, max_attempts + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                session=session,
                progress=False,
            )

            if df is not None and not df.empty:
                break

            last_error = "Returned empty dataframe"
            time.sleep(1.0 * attempt)
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1.0 * attempt)

    if df is None or df.empty:
        return False, None, quality, f"Download failed for {ticker}: {last_error}"

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1. Remove completely empty rows
    df = df.dropna(how="all")

    # 2. Check required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return False, None, quality, f"{ticker} is missing required columns: {missing_cols}"

    df = df[REQUIRED_COLUMNS].copy()

    # 3. Format Date index, check duplicates, sort chronologically
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    duplicate_dates_count = int(df.index.duplicated().sum())
    if duplicate_dates_count > 0:
        df = df[~df.index.duplicated(keep="first")]

    df = df.sort_index()

    # Filter explicitly to the exact date window
    df = df.loc["2015-01-01":"2024-12-31"]

    if df.empty:
        return False, None, quality, f"{ticker} has no rows in 2015-01-01 through 2024-12-31"

    # 4. Check missing values
    missing_values_count = int(df.isna().sum().sum())
    if missing_values_count > 0:
        df = df.ffill().bfill()

    # Record data quality metrics
    quality["rows"] = len(df)
    quality["first_date"] = df.index.min().strftime("%Y-%m-%d")
    quality["last_date"] = df.index.max().strftime("%Y-%m-%d")
    quality["missing_values"] = missing_values_count
    quality["duplicate_dates"] = duplicate_dates_count

    # 5. Save only valid data
    output_dir.mkdir(parents=True, exist_ok=True)
    target_csv = output_dir / f"{ticker}.csv"
    df.to_csv(target_csv, index=True)

    return True, df, quality, f"Successfully saved {len(df)} rows to {target_csv.name}"


def download_all_market_data(
    tickers: List[str] = DEFAULT_TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    output_dir: Path = RAW_DATA_DIR,
) -> Tuple[bool, Dict[str, bool], Dict[str, dict]]:
    """Download and validate raw market data for all specified tickers.

    Args:
        tickers: List of ticker symbols.
        start: Start date string.
        end: End date string.
        output_dir: Destination path for raw CSV files.

    Returns:
        Tuple of (overall_success, ticker_status_dict, quality_metrics_dict).
    """
    session = get_robust_yfinance_session()
    ticker_status: Dict[str, bool] = {}
    quality_metrics: Dict[str, dict] = {}
    failure_reasons: Dict[str, str] = {}

    print("\n--- Starting Raw Market Data Download ---")
    for ticker in tickers:
        success, df, quality, message = download_single_ticker(
            ticker=ticker,
            start=start,
            end=end,
            output_dir=output_dir,
            session=session,
        )
        ticker_status[ticker] = success
        quality_metrics[ticker] = quality

        if success:
            print(
                f"[{ticker}] SUCCESS: {quality['rows']} rows | "
                f"Range: {quality['first_date']} to {quality['last_date']} | "
                f"Missing: {quality['missing_values']} | Duplicates: {quality['duplicate_dates']}"
            )
        else:
            failure_reasons[ticker] = message
            print(f"[{ticker}] FAILED: {message}")

    all_success = all(ticker_status.values()) and len(ticker_status) == len(tickers)

    if all_success:
        print("\n========================================")
        print("MARKET DATA DOWNLOAD COMPLETED")
        print("========================================")
        for ticker in tickers:
            print(f"{ticker:<6} ✓")
        print("========================================\n")
    else:
        print("\n========================================")
        print("MARKET DATA DOWNLOAD FAILED")
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

    success, statuses, metrics = download_all_market_data()
    if not success:
        sys.exit(1)
