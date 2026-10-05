"""Automated API and Research Integrity Test Suite for RL-APM FastAPI Backend.

Tests all API endpoints, model inference, state construction, no-look-ahead constraints,
and verified metrics matching.
"""

from pathlib import Path
import sys
import numpy as np
import pytest
from starlette.testclient import TestClient

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app
from backend.app.core.config import TICKERS, STATE_DIM, ACTION_DIM
from backend.app.services.model_service import model_service
from backend.app.services.market_service import market_service


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with startup and shutdown lifecycle execution."""
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Test 1: Health Endpoint
# ---------------------------------------------------------------------------
def test_health(client):
    """GET /api/health -> 200, model_available, data_available, metrics_available."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_available"] is True
    assert data["data_available"] is True
    assert data["metrics_available"] is True
    assert data["service"] == "RL-APM API"


# ---------------------------------------------------------------------------
# Test 2: Market Data Endpoint
# ---------------------------------------------------------------------------
def test_market_data_default(client):
    """GET /api/market-data?ticker=AAPL -> 200, non-empty data with required fields."""
    response = client.get("/api/market-data?ticker=AAPL")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert data["count"] > 0
    first_row = data["data"][0]
    expected_fields = [
        "date", "open", "high", "low", "close", "adj_close", "volume",
        "daily_return", "sma_20", "sma_50", "rsi_14", "macd", "macd_signal",
        "macd_hist", "bb_high", "bb_mid", "bb_low", "bb_width"
    ]
    for field in expected_fields:
        assert field in first_row, f"Missing field '{field}' in market data row"


# ---------------------------------------------------------------------------
# Test 3: Market Data Filtering & Pagination
# ---------------------------------------------------------------------------
def test_market_data_filtering(client):
    """Verify start_date, end_date, limit, and offset parameters."""
    response = client.get(
        "/api/market-data?ticker=MSFT&start_date=2024-01-01&end_date=2024-01-31&limit=5&offset=2"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "MSFT"
    assert len(data["data"]) == 5
    for row in data["data"]:
        assert "2024-01-01" <= row["date"] <= "2024-01-31"


# ---------------------------------------------------------------------------
# Test 4: Invalid Ticker
# ---------------------------------------------------------------------------
def test_market_data_invalid_ticker(client):
    """Invalid ticker should return 400."""
    response = client.get("/api/market-data?ticker=INVALID_TICKER")
    assert response.status_code == 400
    assert "not supported" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Test 5: Market Summary Endpoint
# ---------------------------------------------------------------------------
def test_market_summary(client):
    """GET /api/market-summary -> 200, summary for all five canonical assets."""
    response = client.get("/api/market-summary")
    assert response.status_code == 200
    data = response.json()
    assert "date" in data
    assets = data["assets"]
    assert len(assets) == 5
    returned_tickers = [a["ticker"] for a in assets]
    assert returned_tickers == TICKERS
    for asset in assets:
        assert asset["close"] > 0
        assert "rsi" in asset
        assert "macd" in asset
        assert "sma20_ratio" in asset
        assert "bollinger_position" in asset


# ---------------------------------------------------------------------------
# Test 6: Technical Indicators Endpoint
# ---------------------------------------------------------------------------
def test_indicators_valid_ticker(client):
    """GET /api/indicators/AAPL -> 200, non-empty list of indicator rows."""
    response = client.get("/api/indicators/AAPL?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert data["count"] == 10
    first = data["indicators"][0]
    assert "rsi_14" in first
    assert "macd" in first
    assert "bb_high" in first


def test_indicators_invalid_ticker(client):
    """GET /api/indicators/BADTICKER -> 400."""
    response = client.get("/api/indicators/BADTICKER")
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Test 7: Prediction Endpoint with Default Weights
# ---------------------------------------------------------------------------
def test_predict_default_weights(client):
    """POST /api/predict -> 200, real model inference, valid probability simplex."""
    response = client.post("/api/predict", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "DDPG"
    assert data["state_dimension"] == STATE_DIM
    allocations = data["allocations"]
    assert len(allocations) == 5
    for ticker in TICKERS:
        assert ticker in allocations
        assert allocations[ticker] >= 0.0
        assert np.isfinite(allocations[ticker])
    total_w = sum(allocations.values())
    assert pytest.approx(total_w, abs=0.01) == 1.0


# ---------------------------------------------------------------------------
# Test 8: Prediction with Custom Valid Weights
# ---------------------------------------------------------------------------
def test_predict_custom_weights(client):
    """POST /api/predict with asymmetric custom weights -> 200, valid output."""
    custom_weights = {
        "AAPL": 0.40,
        "MSFT": 0.20,
        "GOOGL": 0.20,
        "AMZN": 0.10,
        "NVDA": 0.10,
    }
    response = client.post("/api/predict", json={"current_weights": custom_weights})
    assert response.status_code == 200
    data = response.json()
    allocations = data["allocations"]
    assert len(allocations) == 5
    assert pytest.approx(sum(allocations.values()), abs=0.01) == 1.0


# ---------------------------------------------------------------------------
# Test 9: Prediction Weight Validation (Rejection of Invalid Inputs)
# ---------------------------------------------------------------------------
def test_predict_invalid_negative_weights(client):
    """Negative portfolio weight must return 422."""
    payload = {
        "current_weights": {
            "AAPL": -0.10,
            "MSFT": 0.30,
            "GOOGL": 0.30,
            "AMZN": 0.20,
            "NVDA": 0.30,
        }
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_sum_weights(client):
    """Weights that sum to 1.5 must return 422."""
    payload = {
        "current_weights": {
            "AAPL": 0.50,
            "MSFT": 0.30,
            "GOOGL": 0.30,
            "AMZN": 0.20,
            "NVDA": 0.20,
        }
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_ticker_in_weights(client):
    """Missing an asset must return 422."""
    payload = {
        "current_weights": {
            "AAPL": 0.50,
            "MSFT": 0.50,
        }
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 10: Prediction for Historical Date
# ---------------------------------------------------------------------------
def test_predict_historical_date(client):
    """POST /api/predict for historical date 2024-06-14 -> 200."""
    response = client.post("/api/predict", json={"date": "2024-06-14"})
    assert response.status_code == 200
    data = response.json()
    assert data["date"] == "2024-06-14"
    assert pytest.approx(sum(data["allocations"].values()), abs=0.01) == 1.0


def test_predict_invalid_date(client):
    """POST /api/predict for a non-existent or weekend date -> 400."""
    response = client.post("/api/predict", json={"date": "1990-01-01"})
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Test 11: No-Look-Ahead State Construction Integrity
# ---------------------------------------------------------------------------
def test_no_look_ahead_constraint():
    """Verify that constructing state for date D does NOT use rows after D.

    State for date D constructed on the original dataset must be exactly identical
    to state constructed on a truncated dataset where all rows after date D are deleted.
    """
    test_date = "2024-06-14"
    weights = np.ones(5, dtype=np.float64) / 5.0

    # Build reference state from full dataset
    state_original = model_service.build_state(test_date, weights)

    # Save backups of cache and truncate cache to simulate future data not existing
    cache_backup = {t: df.copy() for t, df in market_service._cache.items()}
    try:
        target_dt = np.datetime64(test_date)
        for t in TICKERS:
            df = market_service._cache[t]
            # Drop all future rows
            market_service._cache[t] = df[df["Date"] <= target_dt].copy()

        # Build state on truncated dataset
        state_truncated = model_service.build_state(test_date, weights)

        # Assert bit-for-bit exact equality
        np.testing.assert_allclose(
            state_original,
            state_truncated,
            rtol=1e-7,
            atol=1e-7,
            err_msg="State construction violated no-look-ahead principle by depending on future data!"
        )
    finally:
        # Restore original cache
        market_service._cache = cache_backup


# ---------------------------------------------------------------------------
# Test 12: State Dimension & Ordering Test
# ---------------------------------------------------------------------------
def test_state_dimension_and_ordering():
    """Verify state has shape (50,) with 45 market features + 5 weights."""
    weights = np.array([0.1, 0.2, 0.3, 0.15, 0.25], dtype=np.float64)
    state = model_service.build_state("2024-12-31", weights)
    assert state.shape == (50,)

    # The last 5 dimensions must match the input weights exactly
    last_5_dims = state[-5:]
    np.testing.assert_allclose(last_5_dims, weights, rtol=1e-5, atol=1e-5)

    # First 45 dimensions are finite numbers
    first_45 = state[:45]
    assert np.all(np.isfinite(first_45))


# ---------------------------------------------------------------------------
# Test 13: Backtest Metrics Verification
# ---------------------------------------------------------------------------
def test_backtest_metrics_immutable_values(client):
    """GET /api/backtest -> 200, values match existing verified test_metrics.csv.

    DDPG:
      final_portfolio_value = $144,625.75
      cumulative_return = 44.63%
      sharpe_ratio = 1.17
      max_drawdown = 22.08%

    Equal-Weight:
      final_portfolio_value = $157,948.88
      cumulative_return = 57.95%
      sharpe_ratio = 2.60
      max_drawdown = 17.31%
    """
    response = client.get("/api/backtest")
    assert response.status_code == 200
    data = response.json()

    ddpg = data["ddpg"]
    assert pytest.approx(ddpg["final_portfolio_value"], abs=1.0) == 144625.75
    assert pytest.approx(ddpg["cumulative_return"], abs=0.001) == 0.4463
    assert pytest.approx(ddpg["sharpe_ratio"], abs=0.01) == 1.17
    assert pytest.approx(ddpg["max_drawdown"], abs=0.001) == 0.2208

    bench = data["equal_weight"]
    assert pytest.approx(bench["final_portfolio_value"], abs=1.0) == 157948.88
    assert pytest.approx(bench["cumulative_return"], abs=0.001) == 0.5795
    assert pytest.approx(bench["sharpe_ratio"], abs=0.01) == 2.60
    assert pytest.approx(bench["max_drawdown"], abs=0.001) == 0.1731


# ---------------------------------------------------------------------------
# Test 14: Equity Curve Endpoint
# ---------------------------------------------------------------------------
def test_backtest_equity_curve(client):
    """GET /api/backtest/equity-curve -> 200, non-empty daily points."""
    response = client.get("/api/backtest/equity-curve")
    assert response.status_code == 200
    data = response.json()
    assert data["initial_capital"] == 100000.0
    assert data["start_date"] == "2024-01-02"
    assert data["end_date"] == "2024-12-31"
    assert data["points_count"] > 200
    first_pt = data["points"][0]
    assert first_pt["ddpg_value"] == 100000.0
    assert first_pt["equal_weight_value"] == 100000.0
    last_pt = data["points"][-1]
    assert pytest.approx(last_pt["ddpg_value"], abs=1.0) == 144625.75
    assert pytest.approx(last_pt["equal_weight_value"], abs=1.0) == 157948.88


# ---------------------------------------------------------------------------
# Test 15: Training Metrics Endpoint
# ---------------------------------------------------------------------------
def test_training_metrics(client):
    """GET /api/training-metrics -> 200, 15 validated episodes."""
    response = client.get("/api/training-metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_episodes"] == 15
    assert len(data["episodes"]) == 15
    for ep in data["episodes"]:
        assert ep["episode"] >= 1
        assert np.isfinite(ep["total_reward"])
        assert np.isfinite(ep["portfolio_value"])


# ---------------------------------------------------------------------------
# Test 16: Validation Metrics Endpoint
# ---------------------------------------------------------------------------
def test_validation_metrics(client):
    """GET /api/validation-metrics -> 200, periodic checkpoints."""
    response = client.get("/api/validation-metrics")
    assert response.status_code == 200
    data = response.json()
    checkpoints = data["checkpoints"]
    assert len(checkpoints) >= 5
    episodes = [c["episode"] for c in checkpoints]
    assert 9 in episodes  # Episode 9 was best checkpoint


# ---------------------------------------------------------------------------
# Test 17: Model Info Endpoint
# ---------------------------------------------------------------------------
def test_model_info(client):
    """GET /api/model-info -> 200, verified DDPG configuration."""
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "DDPG"
    assert data["framework"] == "PyTorch"
    assert data["state_dimension"] == 50
    assert data["action_dimension"] == 5
    assert data["gamma"] == 0.99
    assert data["tau"] == 0.005
    assert data["actor_learning_rate"] == 0.0001
    assert data["critic_learning_rate"] == 0.001
    assert data["assets"] == TICKERS


# ---------------------------------------------------------------------------
# Test 18: Portfolio Summary Endpoint
# ---------------------------------------------------------------------------
def test_portfolio_summary(client):
    """GET /api/portfolio/summary -> 200, distinguishes backtest from inference."""
    response = client.get("/api/portfolio/summary")
    assert response.status_code == 200
    data = response.json()
    assert "historical_backtest" in data
    assert "latest_inference" in data
    assert data["model_name"] == "DDPG"
    assert "HISTORICAL" in data["historical_backtest"]["nature"]
    assert "INFERENCE" in data["latest_inference"]["nature"]


# ---------------------------------------------------------------------------
# Test 19: CORS Configuration
# ---------------------------------------------------------------------------
def test_cors_headers(client):
    """Verify CORS preflight / origin headers for React Vite origin."""
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
