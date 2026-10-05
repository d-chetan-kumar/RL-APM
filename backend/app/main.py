"""Main FastAPI Application for RL-APM.

Reinforcement Learning for Portfolio Optimization in Equity Markets.
Exposes validated DDPG model inference, technical indicators, and backtest results.
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import sys
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    CORS_ORIGINS,
)
from backend.app.api.routes import api_router
from backend.app.services.backtest_service import backtest_service
from backend.app.services.market_service import market_service
from backend.app.services.model_service import model_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rl_apm_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle orchestrator.

    Pre-warms data caches and loads the PyTorch Actor checkpoint once.
    """
    logger.info("Initializing RL-APM backend service...")

    # 1. Warm market data cache
    try:
        market_service.load_data()
        logger.info("Processed market data loaded and cached successfully.")
    except Exception as e:
        logger.warning(f"Market data cache initialization warning: {e}")

    # 2. Warm DDPG Actor model into memory
    try:
        model_service.load_model()
        logger.info("Trained DDPG Actor checkpoint loaded and set to eval mode.")
    except Exception as e:
        logger.error(f"Critical error loading model checkpoint: {e}")
        # We allow app startup so /api/health can accurately report degraded status if needed,
        # but log a critical error.

    # 3. Warm backtest and training metrics
    try:
        backtest_service.get_backtest_metrics()
        backtest_service.get_equity_curve()
        backtest_service.get_training_metrics()
        logger.info("Validated research metrics and equity curves cached successfully.")
    except Exception as e:
        logger.warning(f"Metrics initialization warning: {e}")

    logger.info("RL-APM backend initialization complete. Ready for requests.")
    yield
    logger.info("Shutting down RL-APM backend service.")


# Initialize FastAPI application
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for local frontend development (e.g., React + Vite on port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle request validation errors with clean structured details."""
    logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
    serialized_errors = []
    for err in exc.errors():
        err_copy = dict(err)
        if "ctx" in err_copy:
            err_copy["ctx"] = {k: str(v) for k, v in err_copy["ctx"].items()}
        serialized_errors.append(err_copy)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "The request payload failed validation schema checks.",
            "details": serialized_errors,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Standardized HTTP exception handler."""
    logger.warning(f"HTTP {exc.status_code} on {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "status_code": exc.status_code,
            "detail": exc.detail,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all internal server error handler to prevent leaking raw tracebacks."""
    logger.error(f"Unhandled exception on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred while processing the request.",
            "detail": str(exc),
        },
    )


# Root Endpoint
@app.get("/", tags=["Root"])
def root() -> dict:
    """Root endpoint providing quick navigation links."""
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/health",
    }


# Register main API router
app.include_router(api_router)


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("backend.app.main:app", host=host, port=port)
