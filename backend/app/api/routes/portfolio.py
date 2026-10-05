"""Portfolio Prediction, Summary, and Model Metadata Endpoints."""

from fastapi import APIRouter, HTTPException

from backend.app.api.schemas import (
    ModelInfoResponse,
    PortfolioSummaryResponse,
    PredictionRequest,
    PredictionResponse,
)
from backend.app.core.config import (
    ACTION_DIM,
    ACTOR_CHECKPOINT_PATH,
    ACTOR_LR,
    BATCH_SIZE,
    CRITIC_LR,
    GAMMA,
    INITIAL_CAPITAL,
    REPLAY_BUFFER_SIZE,
    STATE_DIM,
    TAU,
    TEST_END,
    TEST_START,
    TICKERS,
    TRAIN_END,
    TRAIN_START,
    TRANSACTION_COST,
    VAL_END,
    VAL_START,
)
from backend.app.services.backtest_service import backtest_service
from backend.app.services.model_service import model_service

router = APIRouter(tags=["Portfolio & Prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict_portfolio(request: PredictionRequest = PredictionRequest()) -> PredictionResponse:
    """Execute DDPG model inference for a requested historical date and current weights.

    Note on Terminology:
      This is a historical-data inference research endpoint, not a live brokerage execution service.
    """
    try:
        target_date, allocations, total_weight, state = model_service.predict(
            date_str=request.date,
            current_weights_dict=request.current_weights,
        )
        return PredictionResponse(
            date=target_date,
            allocations=allocations,
            total_weight=total_weight,
            model="DDPG",
            state_dimension=len(state),
            inference_type="latest available historical-data inference",
            state_vector=[round(float(x), 6) for x in state],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model checkpoint unavailable: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.get("/portfolio/summary", response_model=PortfolioSummaryResponse)
def get_portfolio_summary() -> PortfolioSummaryResponse:
    """Return an integrated overview clearly distinguishing historical test results from latest inference."""
    try:
        # 1. Historical Backtest Data
        backtest_data = backtest_service.get_backtest_metrics()
        hist_summary = {
            "period": backtest_data.evaluation_period,
            "initial_capital": backtest_data.initial_capital,
            "ddpg": {
                "final_portfolio_value": backtest_data.ddpg.final_portfolio_value,
                "cumulative_return": backtest_data.ddpg.cumulative_return,
                "sharpe_ratio": backtest_data.ddpg.sharpe_ratio,
                "annualized_volatility": backtest_data.ddpg.annualized_volatility,
                "max_drawdown": backtest_data.ddpg.max_drawdown,
                "total_transaction_costs": backtest_data.ddpg.total_transaction_costs,
            },
            "equal_weight_benchmark": {
                "final_portfolio_value": backtest_data.equal_weight.final_portfolio_value,
                "cumulative_return": backtest_data.equal_weight.cumulative_return,
                "sharpe_ratio": backtest_data.equal_weight.sharpe_ratio,
                "annualized_volatility": backtest_data.equal_weight.annualized_volatility,
                "max_drawdown": backtest_data.equal_weight.max_drawdown,
                "total_transaction_costs": backtest_data.equal_weight.total_transaction_costs,
            },
            "nature": "HISTORICAL OUT-OF-SAMPLE TEST (2024). Not a guarantee of future live performance.",
        }

        # 2. Latest Historical-Data Inference
        target_date, allocations, total_weight, _ = model_service.predict()
        inference_summary = {
            "date": target_date,
            "allocations": allocations,
            "total_weight": total_weight,
            "nature": "LATEST HISTORICAL-DATA INFERENCE. Deterministic policy output on most recent common data point.",
        }

        return PortfolioSummaryResponse(
            model_name="DDPG",
            evaluation_period=f"{TEST_START} to {TEST_END}",
            historical_backtest=hist_summary,
            latest_inference=inference_summary,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate portfolio summary: {str(e)}")


@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info() -> ModelInfoResponse:
    """Retrieve verified DDPG model architecture, training configuration, and research hyperparameters."""
    actor_arch = (
        "Actor MLP: Input(50) -> Linear(256) -> LayerNorm -> ReLU "
        "-> Linear(256) -> LayerNorm -> ReLU -> Linear(5) -> Raw Action Logits -> Softmax"
    )
    critic_arch = (
        "Critic Q-Network: State(50) -> Linear(256) -> LayerNorm -> ReLU -> StateFeats(256); "
        "Concat(StateFeats[256], Action[5]) -> Linear(261, 256) -> LayerNorm -> ReLU -> Linear(256, 1) -> Q(s, a)"
    )

    return ModelInfoResponse(
        model="DDPG",
        framework="PyTorch",
        state_dimension=STATE_DIM,
        action_dimension=ACTION_DIM,
        assets=TICKERS,
        actor_architecture=actor_arch,
        critic_architecture=critic_arch,
        gamma=GAMMA,
        tau=TAU,
        actor_learning_rate=ACTOR_LR,
        critic_learning_rate=CRITIC_LR,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        train_period=f"{TRAIN_START} to {TRAIN_END}",
        validation_period=f"{VAL_START} to {VAL_END}",
        test_period=f"{TEST_START} to {TEST_END}",
        checkpoint_name=ACTOR_CHECKPOINT_PATH.name,
    )
