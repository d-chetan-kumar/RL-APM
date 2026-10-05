"""Pydantic Schemas for RL-APM API Request and Response Models."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Health Schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status of the service")
    service: str = Field(..., description="Service identifier")
    model_available: bool = Field(..., description="Whether required trained model checkpoint exists")
    data_available: bool = Field(..., description="Whether processed financial data files exist")
    metrics_available: bool = Field(..., description="Whether research metrics files exist")


# ---------------------------------------------------------------------------
# Market Data Schemas
# ---------------------------------------------------------------------------
class MarketDataRow(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float
    daily_return: float
    sma_20: float
    sma_50: float
    rsi_14: float
    macd: float
    macd_signal: float
    macd_hist: float
    bb_high: float
    bb_mid: float
    bb_low: float
    bb_width: float


class MarketDataResponse(BaseModel):
    ticker: str
    count: int
    total_available: int
    data: List[MarketDataRow]


class AssetSummary(BaseModel):
    ticker: str
    date: str
    close: float
    daily_return: float
    sma20_ratio: float
    sma50_ratio: float
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bollinger_position: float
    bollinger_width: float


class MarketSummaryResponse(BaseModel):
    date: str
    assets: List[AssetSummary]


# ---------------------------------------------------------------------------
# Technical Indicator Schemas
# ---------------------------------------------------------------------------
class IndicatorRow(BaseModel):
    date: str
    close: float
    daily_return: float
    sma_20: float
    sma_50: float
    rsi_14: float
    macd: float
    macd_signal: float
    macd_hist: float
    bb_high: float
    bb_mid: float
    bb_low: float
    bb_width: float


class IndicatorResponse(BaseModel):
    ticker: str
    count: int
    indicators: List[IndicatorRow]


# ---------------------------------------------------------------------------
# Portfolio Prediction Schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    date: Optional[str] = Field(
        None,
        description="Historical date string (YYYY-MM-DD) to construct state observation for. "
                    "Defaults to latest available processed market date."
    )
    current_weights: Optional[Dict[str, float]] = Field(
        None,
        description="Current portfolio allocation weights. Must contain AAPL, MSFT, GOOGL, AMZN, NVDA, "
                    "with non-negative values summing to approximately 1.0. Defaults to equal weights (0.20 each)."
    )

    @field_validator("current_weights")
    @classmethod
    def validate_weights(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if v is None:
            return v
        required_tickers = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"}
        provided_tickers = set(v.keys())
        if provided_tickers != required_tickers:
            missing = required_tickers - provided_tickers
            extra = provided_tickers - required_tickers
            msg_parts = []
            if missing:
                msg_parts.append(f"Missing tickers: {sorted(list(missing))}")
            if extra:
                msg_parts.append(f"Unexpected tickers: {sorted(list(extra))}")
            raise ValueError("; ".join(msg_parts))

        for ticker, weight in v.items():
            if not isinstance(weight, (int, float)):
                raise ValueError(f"Weight for '{ticker}' must be numeric, got {type(weight)}")
            if weight < -1e-6:
                raise ValueError(f"Weight for '{ticker}' must be non-negative, got {weight}")

        total = sum(v.values())
        if abs(total - 1.0) > 0.05:
            raise ValueError(f"Portfolio weights must sum to approximately 1.0 (got {total:.4f})")

        return v


class PredictionResponse(BaseModel):
    date: str
    allocations: Dict[str, float]
    total_weight: float
    model: str
    state_dimension: int
    inference_type: str = "latest available historical-data inference"
    state_vector: Optional[List[float]] = Field(None, description="Actual 50-dimensional state vector")


# ---------------------------------------------------------------------------
# Backtest & Evaluation Schemas
# ---------------------------------------------------------------------------
class ModelMetricRecord(BaseModel):
    model: str
    final_portfolio_value: float
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    total_transaction_costs: float


class BacktestMetricsResponse(BaseModel):
    evaluation_period: str
    initial_capital: float
    ddpg: ModelMetricRecord
    equal_weight: ModelMetricRecord


class EquityCurvePoint(BaseModel):
    date: str
    ddpg_value: float
    equal_weight_value: float


class EquityCurveResponse(BaseModel):
    initial_capital: float
    start_date: str
    end_date: str
    points_count: int
    points: List[EquityCurvePoint]


# ---------------------------------------------------------------------------
# Training & Validation Metrics Schemas
# ---------------------------------------------------------------------------
class TrainingEpisodeRecord(BaseModel):
    episode: int
    total_reward: float
    portfolio_value: float
    cumulative_return: float
    transaction_costs: float
    mean_actor_loss: float
    mean_critic_loss: float
    duration_sec: float


class TrainingMetricsResponse(BaseModel):
    total_episodes: int
    episodes: List[TrainingEpisodeRecord]


class ValidationCheckpointRecord(BaseModel):
    episode: int
    val_final_portfolio_value: float
    val_cumulative_return: float
    val_annualized_return: float
    val_annualized_volatility: float
    val_sharpe_ratio: float
    val_max_drawdown: float
    val_transaction_costs: float


class ValidationMetricsResponse(BaseModel):
    checkpoints: List[ValidationCheckpointRecord]


# ---------------------------------------------------------------------------
# Portfolio & Model Info Schemas
# ---------------------------------------------------------------------------
class PortfolioSummaryResponse(BaseModel):
    model_name: str
    evaluation_period: str
    historical_backtest: Dict[str, Any]
    latest_inference: Dict[str, Any]


class ModelInfoResponse(BaseModel):
    model: str
    framework: str
    state_dimension: int
    action_dimension: int
    assets: List[str]
    actor_architecture: str
    critic_architecture: str
    gamma: float
    tau: float
    actor_learning_rate: float
    critic_learning_rate: float
    replay_buffer_size: int
    batch_size: int
    initial_capital: float
    transaction_cost: float
    train_period: str
    validation_period: str
    test_period: str
    checkpoint_name: str
