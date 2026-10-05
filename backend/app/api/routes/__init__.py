"""API Routes package initialization."""

from fastapi import APIRouter
from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.market import router as market_router
from backend.app.api.routes.portfolio import router as portfolio_router
from backend.app.api.routes.backtest import router as backtest_router
from backend.app.api.routes.training import router as training_router

api_router = APIRouter(prefix="/api")
api_router.include_router(health_router)
api_router.include_router(market_router)
api_router.include_router(portfolio_router)
api_router.include_router(backtest_router)
api_router.include_router(training_router)

__all__ = ["api_router"]
