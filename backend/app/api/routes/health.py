"""Health Check Endpoint for RL-APM API."""

from fastapi import APIRouter
from backend.app.api.schemas import HealthResponse
from backend.app.services.backtest_service import backtest_service
from backend.app.services.market_service import market_service
from backend.app.services.model_service import model_service

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Check service health and availability of models, market data, and metric files."""
    model_ok = model_service.is_available()

    # Verify processed data files exist
    data_ok = True
    try:
        market_service.load_data()
    except Exception:
        data_ok = False

    metrics_ok = backtest_service.are_metrics_available()
    overall_status = "ok" if (model_ok and data_ok and metrics_ok) else "degraded"

    return HealthResponse(
        status=overall_status,
        service="RL-APM API",
        model_available=model_ok,
        data_available=data_ok,
        metrics_available=metrics_ok,
    )
