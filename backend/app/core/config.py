"""Application Configuration for RL-APM FastAPI Backend.

Provides project-relative paths, environment defaults, financial hyperparameters,
and model metadata without hardcoding machine-specific absolute paths.
"""

from pathlib import Path
from typing import List
import os

# Project root dynamically determined relative to this file
# backend/app/core/config.py -> 3 levels up to project root: RL-APM
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Data and Artifact directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_METRICS_DIR = RESULTS_DIR / "metrics"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"

# Model Checkpoint Paths
ACTOR_CHECKPOINT_PATH = MODELS_DIR / "ddpg_actor_best.pth"
CRITIC_CHECKPOINT_PATH = MODELS_DIR / "ddpg_critic_best.pth"
ACTOR_FINAL_CHECKPOINT_PATH = MODELS_DIR / "ddpg_actor_final.pth"
CRITIC_FINAL_CHECKPOINT_PATH = MODELS_DIR / "ddpg_critic_final.pth"

# Core Financial Universe (Canonical Order)
TICKERS: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
NUM_ASSETS: int = len(TICKERS)

# Portfolio Constants
INITIAL_CAPITAL: float = 100_000.0
TRANSACTION_COST: float = 0.001

# Chronological Dataset Splits
TRAIN_START: str = "2015-03-16"
TRAIN_END: str = "2021-12-31"
VAL_START: str = "2022-01-01"
VAL_END: str = "2023-12-31"
TEST_START: str = "2024-01-01"
TEST_END: str = "2024-12-31"

# DDPG Architecture & Hyperparameters
STATE_DIM: int = 50
ACTION_DIM: int = 5
FEATURES_PER_ASSET: int = 9
GAMMA: float = 0.99
TAU: float = 0.005
ACTOR_LR: float = 1e-4
CRITIC_LR: float = 1e-3
REPLAY_BUFFER_SIZE: int = 100_000
BATCH_SIZE: int = 64
ACTION_NORMALIZATION: str = "softmax"

# Feature Order Specification (per asset)
FEATURE_NAMES_PER_ASSET: List[str] = [
    "Daily_Return",
    "SMA_20_Ratio",
    "SMA_50_Ratio",
    "RSI_14_Norm",
    "MACD_Norm",
    "MACD_Signal_Norm",
    "MACD_Hist_Norm",
    "BB_Position",
    "BB_Width_Norm",
]

# Server and CORS Configuration
API_TITLE: str = "RL-APM API"
API_VERSION: str = "1.0.0"
API_DESCRIPTION: str = (
    "FastAPI backend for Reinforcement Learning for Portfolio Optimization in Equity Markets. "
    "Exposes historical market indicators, real DDPG model inference, and validated backtest results."
)

# Configurable CORS origins for frontend access
_cors_env = os.getenv("CORS_ORIGINS")
_frontend_url = os.getenv("FRONTEND_URL")
_origins: List[str] = []
if _cors_env:
    _origins.extend([orig.strip() for orig in _cors_env.split(",") if orig.strip()])
if _frontend_url:
    _origins.extend([orig.strip() for orig in _frontend_url.split(",") if orig.strip()])
if not _origins:
    _origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
# Deduplicate while preserving order
CORS_ORIGINS: List[str] = list(dict.fromkeys(_origins))
