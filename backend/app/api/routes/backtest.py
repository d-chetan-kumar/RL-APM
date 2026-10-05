"""Backtest Results and Equity Curve Endpoints."""

from fastapi import APIRouter, HTTPException

from backend.app.api.schemas import BacktestMetricsResponse, EquityCurveResponse
from backend.app.services.backtest_service import backtest_service

router = APIRouter(tags=["Backtest & Evaluation"])


@router.get("/backtest", response_model=BacktestMetricsResponse)
def get_backtest_metrics() -> BacktestMetricsResponse:
    """Retrieve verified held-out 2024 out-of-sample backtest metrics for DDPG vs. Equal-Weight."""
    try:
        return backtest_service.get_backtest_metrics()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Backtest metrics file not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading backtest metrics: {str(e)}")


@router.get("/backtest/equity-curve", response_model=EquityCurveResponse)
def get_backtest_equity_curve() -> EquityCurveResponse:
    """Retrieve structured daily portfolio values throughout the 2024 test horizon."""
    try:
        return backtest_service.get_equity_curve()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Equity curve data not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading equity curve: {str(e)}")
