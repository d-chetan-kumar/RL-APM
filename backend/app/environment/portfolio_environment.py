"""Portfolio Optimization Reinforcement Learning Environment for RL-APM.

Implements a continuous multi-asset portfolio rebalancing environment designed for
Deep Deterministic Policy Gradient (DDPG) agents, with transaction costs, realistic
look-ahead bias prevention, and rigorous state/action representations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Path resolution relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_TICKERS: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

# Per-asset technical feature names (9 features per asset)
FEATURE_NAMES_PER_ASSET: List[str] = [
    "Daily_Return",      # Current period daily return (close-to-close)
    "SMA_Ratio_20",     # (Close / SMA_20) - 1.0: short-term trend deviation
    "SMA_Ratio_50",     # (Close / SMA_50) - 1.0: medium-term trend deviation
    "RSI_14",           # RSI / 100.0: momentum oscillator normalized to [0, 1]
    "MACD_Scaled",       # MACD / Close: normalized MACD line
    "MACD_Signal_Scaled",# MACD_Signal / Close: normalized signal line
    "MACD_Hist_Scaled",  # MACD_Hist / Close: normalized MACD histogram
    "BB_Position",      # (Close - BB_Low) / (BB_High - BB_Low + eps): %B band position
    "BB_Width",         # BB_Width / 100.0: normalized volatility band width
]


class PortfolioEnvironment:
    """Financial portfolio rebalancing environment for RL agents.

    State Dimension:
        50 dimensions:
        - 45 market features (5 assets x 9 normalized technical indicators)
        - 5 current portfolio weights (w_AAPL, w_MSFT, w_GOOGL, w_AMZN, w_NVDA)

    Action Dimension:
        5 dimensions:
        - Continuous raw allocation weights, normalized via Softmax to non-negative weights summing to 1.
    """

    def __init__(
        self,
        tickers: List[str] = DEFAULT_TICKERS,
        data_dir: Path = DEFAULT_DATA_DIR,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,  # 0.1% per turnover unit
        risk_penalty: float = 0.0,
        transaction_cost_penalty: float = 0.0,
        action_normalization: str = "softmax",  # 'softmax' or 'l1_positive'
    ):
        """Initialize the portfolio environment.

        Args:
            tickers: List of equity ticker symbols.
            data_dir: Directory containing processed CSV files.
            start_date: Start date string (YYYY-MM-DD) for episode horizon.
            end_date: End date string (YYYY-MM-DD) for episode horizon.
            start_idx: Start index alternative to start_date.
            end_idx: End index alternative to end_date.
            initial_capital: Initial portfolio dollar value (default $100,000.00).
            transaction_cost: Proportional transaction fee rate (0.001 = 0.1%).
            risk_penalty: Coefficient for risk-penalty term in reward.
            transaction_cost_penalty: Coefficient for transaction-cost penalty term.
            action_normalization: Normalization method for raw actions ('softmax').
        """
        self.tickers = list(tickers)
        self.num_assets = len(self.tickers)
        self.action_space_size = self.num_assets
        self.action_dim = self.num_assets
        self.features_per_asset = len(FEATURE_NAMES_PER_ASSET)
        self.market_state_dim = self.num_assets * self.features_per_asset
        self.state_dim = self.market_state_dim + self.num_assets  # 45 + 5 = 50

        self.data_dir = Path(data_dir)
        self.initial_capital = float(initial_capital)
        self.transaction_cost = float(transaction_cost)
        self.risk_penalty = float(risk_penalty)
        self.transaction_cost_penalty = float(transaction_cost_penalty)
        self.action_normalization = action_normalization

        # Load, align, and construct state matrices
        self._load_and_align_market_data(start_date, end_date, start_idx, end_idx)

        # Episode runtime state variables
        self.current_step: int = 0
        self.portfolio_value: float = self.initial_capital
        self.current_weights: np.ndarray = np.ones(self.num_assets, dtype=np.float64) / self.num_assets
        self.cumulative_transaction_costs: float = 0.0
        self.history: List[Dict[str, Any]] = []

    def _load_and_align_market_data(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
        start_idx: Optional[int],
        end_idx: Optional[int],
    ) -> None:
        """Load processed CSV files, ensure alignment, and build pre-computed tensors."""
        dfs: Dict[str, pd.DataFrame] = {}

        for ticker in self.tickers:
            file_path = self.data_dir / f"{ticker}_processed.csv"
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Processed data file missing for ticker '{ticker}' at: {file_path}"
                )

            df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
            if df.empty:
                raise ValueError(f"Processed dataset for '{ticker}' is empty.")

            # Validate required indicator columns exist
            required_cols = [
                "Close", "Adj Close", "Daily_Return", "SMA_20", "SMA_50",
                "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
                "BB_High", "BB_Mid", "BB_Low", "BB_Width",
            ]
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                raise ValueError(f"Dataset for '{ticker}' missing indicator columns: {missing_cols}")

            # Check for duplicate dates
            if df.index.duplicated().any():
                raise ValueError(f"Dataset for '{ticker}' contains duplicate dates.")

            dfs[ticker] = df

        # Form common trading dates intersection
        date_sets = [set(df.index) for df in dfs.values()]
        common_dates = sorted(list(set.intersection(*date_sets)))
        if not common_dates:
            raise ValueError("Zero common trading dates found across the specified assets.")

        # Reindex all datasets to common dates
        for ticker in self.tickers:
            dfs[ticker] = dfs[ticker].loc[common_dates].sort_index()
            # Verify no missing values in common subset
            if dfs[ticker].isna().any().any():
                nan_cols = dfs[ticker].columns[dfs[ticker].isna().any()].tolist()
                raise ValueError(
                    f"Asset '{ticker}' has NaN values in common date subset for columns: {nan_cols}"
                )

        # Slice horizon by date or index if specified
        all_dates = pd.DatetimeIndex(common_dates)
        if start_date is not None:
            all_dates = all_dates[all_dates >= pd.to_datetime(start_date)]
        if end_date is not None:
            all_dates = all_dates[all_dates <= pd.to_datetime(end_date)]

        sliced_dates = all_dates.tolist()
        if start_idx is not None or end_idx is not None:
            s_idx = start_idx if start_idx is not None else 0
            e_idx = end_idx if end_idx is not None else len(sliced_dates)
            sliced_dates = sliced_dates[s_idx:e_idx]

        if len(sliced_dates) < 2:
            raise ValueError(
                f"Horizon contains {len(sliced_dates)} dates; minimum 2 required for step transitions."
            )

        self.dates = pd.DatetimeIndex(sliced_dates)
        self.num_periods = len(self.dates)

        # Precompute normalized state feature matrices: Shape [T, num_assets * 9]
        # Precompute price matrices for returns calculation: Shape [T, num_assets]
        market_features_list = []
        adj_close_list = []

        for d in self.dates:
            row_features = []
            row_prices = []
            for ticker in self.tickers:
                row = dfs[ticker].loc[d]
                close = float(row["Close"])
                adj_close = float(row["Adj Close"])
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

                # Transformed and scaled feature values for numerical stability:
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

                row_features.extend([
                    f_ret, f_sma20, f_sma50, f_rsi,
                    f_macd, f_macdsig, f_macdhist, f_bbpos, f_bbwidth
                ])
                row_prices.append(adj_close)

            market_features_list.append(row_features)
            adj_close_list.append(row_prices)

        self.market_features_matrix = np.array(market_features_list, dtype=np.float64)
        self.adj_close_matrix = np.array(adj_close_list, dtype=np.float64)

    def normalize_action(self, action: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Convert continuous raw action vector into valid portfolio weights.

        Requirements:
        - Weights must be strictly non-negative.
        - Weights must sum exactly to 1.0.

        Args:
            action: Array of shape (5,) representing continuous raw action logits.

        Returns:
            Normalized weights array of shape (5,) summing to 1.0.
        """
        a = np.asarray(action, dtype=np.float64).flatten()
        if len(a) != self.num_assets:
            raise ValueError(f"Expected action of length {self.num_assets}, got length {len(a)}")

        if self.action_normalization == "softmax":
            # Numerical stability: subtract max before exp
            shift_a = a - np.max(a)
            exp_a = np.exp(shift_a)
            weights = exp_a / (np.sum(exp_a) + 1e-12)
        elif self.action_normalization == "l1_positive":
            pos_a = np.maximum(a, 0.0)
            s = np.sum(pos_a)
            if s < 1e-8:
                # Fallback to equal weights if all non-positive
                weights = np.ones(self.num_assets, dtype=np.float64) / self.num_assets
            else:
                weights = pos_a / s
        else:
            raise ValueError(f"Unknown action normalization method: {self.action_normalization}")

        # Final sanity assertion
        weights = weights / np.sum(weights)
        return weights

    def _get_state(self) -> np.ndarray:
        """Construct the 50-dimensional observation state vector at current step t.

        Features:
        - Indices 0 to 44: 45 market technical features (5 assets x 9 indicators at step t)
        - Indices 45 to 49: 5 current portfolio weights [w_AAPL, w_MSFT, w_GOOGL, w_AMZN, w_NVDA]

        Zero future information is present in this vector.
        """
        market_feats = self.market_features_matrix[self.current_step]
        state = np.concatenate([market_feats, self.current_weights]).astype(np.float32)
        return state

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial conditions at step 0.

        Returns:
            Tuple of (initial_state, initial_info_dict).
        """
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        # Equal allocation initial portfolio: 20% each across 5 assets
        self.current_weights = np.ones(self.num_assets, dtype=np.float64) / self.num_assets
        self.cumulative_transaction_costs = 0.0
        self.history = []

        initial_state = self._get_state()
        info = self.get_portfolio_summary()
        info["step"] = 0
        return initial_state, info

    def step(
        self, action: Union[np.ndarray, List[float]]
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one trading decision step in the environment.

        Sequence:
        1. Current state S_t is observed at date D_t.
        2. Agent supplies continuous action a_t, normalized to target weights w_{t+1}^*.
        3. Transaction cost is assessed on turnover: sum(|w_{t+1}^* - w_t|).
        4. Over the NEXT period (from D_t to D_{t+1}), asset price returns r_{t+1} are realized.
        5. Portfolio value updates: V_{t+1} = (V_t - cost) * (1 + gross_return).
        6. Effective weights update to post-movement weights.
        7. Step increments: t <- t + 1. Next state S_{t+1} observed at D_{t+1}.
        8. Reward computed based on net return, risk penalty, and transaction cost penalty.

        Args:
            action: Continuous action vector of length 5.

        Returns:
            Tuple of (next_state, reward, done, info_dict).
        """
        if self.current_step >= self.num_periods - 1:
            # Reached terminal state
            state = self._get_state()
            info = self.get_portfolio_summary()
            info["terminal_reason"] = "horizon_ended"
            return state, 0.0, True, info

        # Step 1 & 2: Normalize action to valid target allocation weights
        new_weights = self.normalize_action(action)
        prev_weights = self.current_weights.copy()
        current_date = self.dates[self.current_step]
        next_date = self.dates[self.current_step + 1]

        # Step 3: Compute turnover and transaction cost incurred before return realization
        turnover = float(np.sum(np.abs(new_weights - prev_weights)))
        transaction_cost_amount = turnover * self.transaction_cost * self.portfolio_value
        self.cumulative_transaction_costs += transaction_cost_amount

        # Step 4: Asset price movement realized from date t to date t+1 (Adj Close)
        prices_t = self.adj_close_matrix[self.current_step]
        prices_t1 = self.adj_close_matrix[self.current_step + 1]
        asset_returns = (prices_t1 - prices_t) / prices_t

        # Step 5: Gross return of the rebalanced portfolio
        gross_return = float(np.sum(new_weights * asset_returns))

        # Capital after transaction fees invested into the market
        capital_after_fees = self.portfolio_value - transaction_cost_amount
        new_portfolio_value = capital_after_fees * (1.0 + gross_return)

        # Net return of the portfolio over this step
        net_return = (new_portfolio_value - self.portfolio_value) / self.portfolio_value

        # Step 6: Post-movement drifted weights at t+1 before next rebalancing
        asset_values_t1 = (capital_after_fees * new_weights) * (1.0 + asset_returns)
        if new_portfolio_value > 1e-8:
            post_drift_weights = asset_values_t1 / new_portfolio_value
        else:
            post_drift_weights = new_weights.copy()

        # Update environment internal variables
        self.portfolio_value = float(new_portfolio_value)
        self.current_weights = post_drift_weights
        self.current_step += 1

        # Step 7: Reward calculation
        # Baseline reward: Net portfolio return
        # Optional risk penalty and transaction cost penalty terms
        risk_penalty_term = self.risk_penalty * (net_return ** 2 if net_return < 0 else 0.0)
        cost_penalty_term = self.transaction_cost_penalty * (transaction_cost_amount / (self.initial_capital + 1e-8))
        reward = float(net_return - risk_penalty_term - cost_penalty_term)

        # Check termination condition
        done = bool(self.current_step >= self.num_periods - 1 or self.portfolio_value <= 0.0)

        # Step 8: Next state
        next_state = self._get_state()

        # Build comprehensive info dictionary
        info = {
            "step": self.current_step,
            "current_date": next_date.strftime("%Y-%m-%d"),
            "decision_date": current_date.strftime("%Y-%m-%d"),
            "portfolio_value": self.portfolio_value,
            "daily_return": net_return,
            "net_return": net_return,
            "gross_return": gross_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost_amount,
            "cumulative_transaction_costs": self.cumulative_transaction_costs,
            "cumulative_return": (self.portfolio_value / self.initial_capital) - 1.0,
            "target_weights": {t: float(new_weights[i]) for i, t in enumerate(self.tickers)},
            "current_weights": {t: float(self.current_weights[i]) for i, t in enumerate(self.tickers)},
        }
        self.history.append(info)

        return next_state, reward, done, info

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Return a snapshot summary of current portfolio status."""
        current_date_str = (
            self.dates[self.current_step].strftime("%Y-%m-%d")
            if self.current_step < len(self.dates)
            else self.dates[-1].strftime("%Y-%m-%d")
        )
        return {
            "current_date": current_date_str,
            "current_step": self.current_step,
            "total_steps": self.num_periods - 1,
            "portfolio_value": self.portfolio_value,
            "cumulative_return": (self.portfolio_value / self.initial_capital) - 1.0,
            "cumulative_transaction_costs": self.cumulative_transaction_costs,
            "current_weights": {t: float(self.current_weights[i]) for i, t in enumerate(self.tickers)},
        }

    def get_feature_description(self) -> List[str]:
        """Return the exact ordered list of feature names in the state vector."""
        names = []
        for ticker in self.tickers:
            for feat in FEATURE_NAMES_PER_ASSET:
                names.append(f"{ticker}_{feat}")
        for ticker in self.tickers:
            names.append(f"weight_{ticker}")
        return names
