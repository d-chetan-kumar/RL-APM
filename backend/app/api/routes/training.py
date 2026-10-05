"""Training and Validation Metrics Endpoints."""

from fastapi import APIRouter, HTTPException

from backend.app.api.schemas import (
    TrainingMetricsResponse,
    ValidationMetricsResponse,
)
from backend.app.services.backtest_service import backtest_service

router = APIRouter(tags=["Training Logs"])


@router.get("/training-metrics", response_model=TrainingMetricsResponse)
def get_training_metrics() -> TrainingMetricsResponse:
    """Retrieve logged metrics for all 15 training episodes (rewards, actor/critic loss, returns)."""
    try:
        return backtest_service.get_training_metrics()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Training metrics file not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading training metrics: {str(e)}")


@router.get("/validation-metrics", response_model=ValidationMetricsResponse)
def get_validation_metrics() -> ValidationMetricsResponse:
    """Retrieve deterministic validation checkpoints evaluated during training."""
    try:
        return backtest_service.get_validation_metrics()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Validation metrics file not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading validation metrics: {str(e)}")
