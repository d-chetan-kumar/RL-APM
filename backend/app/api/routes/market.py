"""Market Data and Technical Indicator Endpoints."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from backend.app.api.schemas import (
    IndicatorResponse,
    MarketDataResponse,
    MarketSummaryResponse,
)
from backend.app.services.market_service import market_service

router = APIRouter(tags=["Market Data"])


@router.get("/market-data", response_model=MarketDataResponse)
def get_market_data(
    ticker: str = Query("AAPL", description="Equity ticker symbol (AAPL, MSFT, GOOGL, AMZN, NVDA)"),
    start_date: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)"),
    limit: int = Query(500, ge=1, le=2000, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Pagination record offset"),
) -> MarketDataResponse:
    """Retrieve historical processed market data and indicators for a specified asset."""
    try:
        return market_service.get_market_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/market-summary", response_model=MarketSummaryResponse)
def get_market_summary() -> MarketSummaryResponse:
    """Retrieve the latest financial indicator and price snapshot for all 5 assets."""
    try:
        return market_service.get_market_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate market summary: {str(e)}")


@router.get("/indicators/{ticker}", response_model=IndicatorResponse)
def get_indicators(
    ticker: str,
    start_date: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)"),
    limit: int = Query(500, ge=1, le=2000, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Pagination record offset"),
) -> IndicatorResponse:
    """Retrieve historical technical indicators (SMA, RSI, MACD, Bollinger Bands) for a specific asset."""
    try:
        rows = market_service.get_indicators(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )
        return IndicatorResponse(
            ticker=ticker.strip().upper(),
            count=len(rows),
            indicators=rows,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
