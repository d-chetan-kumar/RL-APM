"""Market Data Service for RL-APM.

Loads, caches, and queries processed equity market data and technical indicators.
All data is sourced strictly from data/processed/.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from backend.app.core.config import DATA_PROCESSED_DIR, TICKERS
from backend.app.api.schemas import (
    AssetSummary,
    IndicatorRow,
    MarketDataRow,
    MarketDataResponse,
    MarketSummaryResponse,
)


class MarketService:
    """Service providing efficient, cached access to processed financial market data."""

    def __init__(self, data_dir: Path = DATA_PROCESSED_DIR, tickers: List[str] = TICKERS):
        self.data_dir = Path(data_dir)
        self.tickers = list(tickers)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._common_dates: List[pd.Timestamp] = []
        self._is_loaded: bool = False

    def load_data(self, force_reload: bool = False) -> None:
        """Load and cache all processed CSV files into memory."""
        if self._is_loaded and not force_reload:
            return

        date_sets = []
        for ticker in self.tickers:
            file_path = self.data_dir / f"{ticker}_processed.csv"
            if not file_path.exists():
                raise FileNotFoundError(f"Processed market data missing for ticker '{ticker}' at: {file_path}")

            df = pd.read_csv(file_path, parse_dates=["Date"])
            df.sort_values("Date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            self._cache[ticker] = df
            date_sets.append(set(df["Date"]))

        if date_sets:
            self._common_dates = sorted(list(set.intersection(*date_sets)))
        self._is_loaded = True

    def validate_ticker(self, ticker: str) -> str:
        """Validate and canonicalize ticker symbol."""
        t_clean = ticker.strip().upper()
        if t_clean not in self.tickers:
            raise ValueError(f"Ticker '{ticker}' is not supported. Supported assets: {self.tickers}")
        return t_clean

    def get_latest_common_date(self) -> str:
        """Return the latest available date present across all tracked assets."""
        self.load_data()
        if not self._common_dates:
            raise ValueError("No common dates available across tracked assets.")
        return self._common_dates[-1].strftime("%Y-%m-%d")

    def get_common_dates(self) -> List[str]:
        """Return list of formatted string dates common to all assets."""
        self.load_data()
        return [d.strftime("%Y-%m-%d") for d in self._common_dates]

    def get_market_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> MarketDataResponse:
        """Query historical market data for a given ticker with date filtering and pagination."""
        t_clean = self.validate_ticker(ticker)
        self.load_data()

        df = self._cache[t_clean].copy()

        # Date filtering
        if start_date:
            s_dt = pd.to_datetime(start_date)
            df = df[df["Date"] >= s_dt]
        if end_date:
            e_dt = pd.to_datetime(end_date)
            df = df[df["Date"] <= e_dt]

        total_available = len(df)

        # Pagination
        offset = max(0, offset)
        limit = min(max(1, limit), 2000)
        paged_df = df.iloc[offset : offset + limit]

        rows: List[MarketDataRow] = []
        for _, row in paged_df.iterrows():
            rows.append(
                MarketDataRow(
                    date=row["Date"].strftime("%Y-%m-%d"),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    adj_close=float(row["Adj Close"]),
                    volume=float(row["Volume"]),
                    daily_return=float(row["Daily_Return"]),
                    sma_20=float(row["SMA_20"]),
                    sma_50=float(row["SMA_50"]),
                    rsi_14=float(row["RSI_14"]),
                    macd=float(row["MACD"]),
                    macd_signal=float(row["MACD_Signal"]),
                    macd_hist=float(row["MACD_Hist"]),
                    bb_high=float(row["BB_High"]),
                    bb_mid=float(row["BB_Mid"]),
                    bb_low=float(row["BB_Low"]),
                    bb_width=float(row["BB_Width"]),
                )
            )

        return MarketDataResponse(
            ticker=t_clean,
            count=len(rows),
            total_available=total_available,
            data=rows,
        )

    def get_market_summary(self) -> MarketSummaryResponse:
        """Return the latest indicator and price summary across all 5 assets on the common latest date."""
        self.load_data()
        latest_date_str = self.get_latest_common_date()
        latest_dt = pd.to_datetime(latest_date_str)

        summaries: List[AssetSummary] = []
        for ticker in self.tickers:
            df = self._cache[ticker]
            match = df[df["Date"] == latest_dt]
            if match.empty:
                raise ValueError(f"No record found for asset {ticker} on latest common date {latest_date_str}")
            row = match.iloc[0]

            close = float(row["Close"])
            sma20 = float(row["SMA_20"])
            sma50 = float(row["SMA_50"])
            sma20_ratio = (close / (sma20 + 1e-8)) - 1.0
            sma50_ratio = (close / (sma50 + 1e-8)) - 1.0

            bb_high = float(row["BB_High"])
            bb_low = float(row["BB_Low"])
            bb_span = bb_high - bb_low
            bb_pos = (close - bb_low) / (bb_span + 1e-8)

            summaries.append(
                AssetSummary(
                    ticker=ticker,
                    date=latest_date_str,
                    close=close,
                    daily_return=float(row["Daily_Return"]),
                    sma_20=sma20,
                    sma_50=sma50,
                    sma20_ratio=float(sma20_ratio),
                    sma50_ratio=float(sma50_ratio),
                    rsi=float(row["RSI_14"]),
                    macd=float(row["MACD"]),
                    macd_signal=float(row["MACD_Signal"]),
                    macd_histogram=float(row["MACD_Hist"]),
                    bollinger_position=float(bb_pos),
                    bollinger_width=float(row["BB_Width"]),
                )
            )

        return MarketSummaryResponse(date=latest_date_str, assets=summaries)

    def get_indicators(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> List[IndicatorRow]:
        """Fetch historical technical indicators for a specified asset."""
        t_clean = self.validate_ticker(ticker)
        self.load_data()

        df = self._cache[t_clean].copy()
        if start_date:
            df = df[df["Date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["Date"] <= pd.to_datetime(end_date)]

        offset = max(0, offset)
        limit = min(max(1, limit), 2000)
        paged_df = df.iloc[offset : offset + limit]

        rows: List[IndicatorRow] = []
        for _, row in paged_df.iterrows():
            rows.append(
                IndicatorRow(
                    date=row["Date"].strftime("%Y-%m-%d"),
                    close=float(row["Close"]),
                    daily_return=float(row["Daily_Return"]),
                    sma_20=float(row["SMA_20"]),
                    sma_50=float(row["SMA_50"]),
                    rsi_14=float(row["RSI_14"]),
                    macd=float(row["MACD"]),
                    macd_signal=float(row["MACD_Signal"]),
                    macd_hist=float(row["MACD_Hist"]),
                    bb_high=float(row["BB_High"]),
                    bb_mid=float(row["BB_Mid"]),
                    bb_low=float(row["BB_Low"]),
                    bb_width=float(row["BB_Width"]),
                )
            )
        return rows


# Global singleton instance
market_service = MarketService()
