"""Model Inference Service for RL-APM.

Coordinates singleton loading of the trained PyTorch DDPG Actor checkpoint,
deterministic state feature construction, and action normalization for portfolio rebalancing inference.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from backend.app.core.config import (
    ACTOR_CHECKPOINT_PATH,
    STATE_DIM,
    ACTION_DIM,
    TICKERS,
    DATA_PROCESSED_DIR,
)
from backend.app.models.networks import Actor
from backend.app.services.market_service import market_service


class ModelService:
    """Singleton service for loading trained DDPG policy and executing live inference."""

    def __init__(
        self,
        checkpoint_path: Path = ACTOR_CHECKPOINT_PATH,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        tickers: List[str] = TICKERS,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tickers = list(tickers)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor: Optional[Actor] = None
        self._is_loaded: bool = False

    def load_model(self, force_reload: bool = False) -> None:
        """Load the trained Actor checkpoint once into memory."""
        if self._is_loaded and not force_reload:
            return

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Trained DDPG Actor checkpoint not found at: {self.checkpoint_path}"
            )

        actor_model = Actor(self.state_dim, self.action_dim).to(self.device)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        actor_model.load_state_dict(state_dict)
        actor_model.eval()

        self.actor = actor_model
        self._is_loaded = True

    def is_available(self) -> bool:
        """Check if model checkpoint file exists on disk."""
        return self.checkpoint_path.exists()

    def build_state(
        self,
        date_str: str,
        current_weights: np.ndarray,
    ) -> np.ndarray:
        """Construct the exact 50-dimensional observation state vector for a given date.

        Research Integrity (No Look-Ahead):
          Strictly consumes indicators and prices for `date_str` only.
          Does NOT look at future timestamps (t+1, t+2, etc.).

        State Composition:
          45 market features (5 assets x 9 indicators in exact canonical order)
          + 5 current portfolio weights
          = 50 dimensions.
        """
        if len(current_weights) != len(self.tickers):
            raise ValueError(
                f"Current weights length ({len(current_weights)}) must match asset count ({len(self.tickers)})"
            )

        market_service.load_data()
        target_dt = pd.to_datetime(date_str)

        market_features: List[float] = []

        # Canonical Asset Order: AAPL -> MSFT -> GOOGL -> AMZN -> NVDA
        for ticker in self.tickers:
            df = market_service._cache[ticker]
            # Exact date match - strictly no future rows
            row_match = df[df["Date"] == target_dt]
            if row_match.empty:
                raise ValueError(
                    f"Date '{date_str}' not found in processed data for asset '{ticker}'. "
                    f"Available range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
                )

            row = row_match.iloc[0]
            close = float(row["Close"])
            daily_return = float(row["Daily_Return"])
            sma_20 = float(row["SMA_20"])
            sma_50 = float(row["SMA_50"])
            rsi_14 = float(row["RSI_14"])
            macd = float(row["MACD"])
            macd_sig = float(row["MACD_Signal"])
            macd_hist = float(row["MACD_Hist"])
            bb_high = float(row["BB_High"])
            bb_low = float(row["BB_Low"])
            bb_width = float(row["BB_Width"])

            # Exact same 9 features and numerical scaling as PortfolioEnvironment:
            f_ret = daily_return
            f_sma20 = (close / (sma_20 + 1e-8)) - 1.0
            f_sma50 = (close / (sma_50 + 1e-8)) - 1.0
            f_rsi = rsi_14 / 100.0
            f_macd = macd / (close + 1e-8)
            f_macdsig = macd_sig / (close + 1e-8)
            f_macdhist = macd_hist / (close + 1e-8)
            bb_span = bb_high - bb_low
            f_bbpos = (close - bb_low) / (bb_span + 1e-8)
            f_bbwidth = bb_width / 100.0

            market_features.extend([
                f_ret, f_sma20, f_sma50, f_rsi,
                f_macd, f_macdsig, f_macdhist, f_bbpos, f_bbwidth
            ])

        # Append current portfolio weights (normalized)
        weights_list = list(current_weights)
        state_vec = np.array(market_features + weights_list, dtype=np.float32)

        if len(state_vec) != self.state_dim:
            raise ValueError(f"Constructed state vector length {len(state_vec)} does not match {self.state_dim}")

        return state_vec

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Convert continuous raw action vector into valid portfolio weights.

        Uses the exact same Softmax formulation as PortfolioEnvironment:
          shift_a = a - max(a)
          exp_a = exp(shift_a)
          weights = exp_a / sum(exp_a)
        """
        a = np.asarray(action, dtype=np.float64).flatten()
        if len(a) != self.action_dim:
            raise ValueError(f"Expected action of length {self.action_dim}, got {len(a)}")

        shift_a = a - np.max(a)
        exp_a = np.exp(shift_a)
        weights = exp_a / (np.sum(exp_a) + 1e-12)
        weights = weights / np.sum(weights)
        return weights

    def predict(
        self,
        date_str: Optional[str] = None,
        current_weights_dict: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, Dict[str, float], float, np.ndarray]:
        """Execute deterministic policy inference for the given date and weights.

        Returns:
            Tuple of (effective_date, allocations_dict, total_weight, raw_state_vector).
        """
        self.load_model()

        # Resolve date
        if date_str is None or not date_str.strip():
            target_date = market_service.get_latest_common_date()
        else:
            target_date = date_str.strip()

        # Resolve weights (equal-weight default: 0.20 for each asset)
        if current_weights_dict is None:
            weights_arr = np.ones(self.action_dim, dtype=np.float64) / self.action_dim
        else:
            weights_arr = np.array(
                [current_weights_dict[t] for t in self.tickers], dtype=np.float64
            )
            # Re-normalize if minor numerical variance
            weights_arr = weights_arr / np.sum(weights_arr)

        # Build exact 50-dimensional observation state
        state = self.build_state(target_date, weights_arr)

        # Deterministic Actor forward pass with gradients disabled
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            raw_action = self.actor(state_tensor).squeeze(0).cpu().numpy()

        # Softmax normalization
        target_weights = self.normalize_action(raw_action)

        allocations = {
            ticker: round(float(w), 4) for ticker, w in zip(self.tickers, target_weights)
        }
        total_weight = round(float(sum(allocations.values())), 4)

        return target_date, allocations, total_weight, state


# Global singleton instance
model_service = ModelService()
