"""Backtest and Experiment Metrics Service for RL-APM.

Loads, caches, and exposes validated out-of-sample test results, daily equity curves,
training logs, and validation checkpoints strictly from results/metrics/.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from backend.app.core.config import (
    INITIAL_CAPITAL,
    RESULTS_METRICS_DIR,
    TEST_START,
    TEST_END,
)
from backend.app.api.schemas import (
    BacktestMetricsResponse,
    EquityCurvePoint,
    EquityCurveResponse,
    ModelMetricRecord,
    TrainingEpisodeRecord,
    TrainingMetricsResponse,
    ValidationCheckpointRecord,
    ValidationMetricsResponse,
)


class BacktestService:
    """Service providing access to validated backtest trajectories and training logs."""

    def __init__(self, metrics_dir: Path = RESULTS_METRICS_DIR):
        self.metrics_dir = Path(metrics_dir)
        self.test_metrics_path = self.metrics_dir / "test_metrics.csv"
        self.equity_curve_path = self.metrics_dir / "test_equity_curve.csv"
        self.training_metrics_path = self.metrics_dir / "training_metrics.csv"
        self.validation_metrics_path = self.metrics_dir / "validation_metrics.csv"

        self._test_metrics_cache: Optional[BacktestMetricsResponse] = None
        self._equity_curve_cache: Optional[EquityCurveResponse] = None
        self._training_metrics_cache: Optional[TrainingMetricsResponse] = None
        self._validation_metrics_cache: Optional[ValidationMetricsResponse] = None

    def are_metrics_available(self) -> bool:
        """Check if all required metrics files exist on disk."""
        return (
            self.test_metrics_path.exists()
            and self.equity_curve_path.exists()
            and self.training_metrics_path.exists()
            and self.validation_metrics_path.exists()
        )

    def get_backtest_metrics(self, force_reload: bool = False) -> BacktestMetricsResponse:
        """Load and return existing held-out 2024 test metrics from results/metrics/test_metrics.csv."""
        if self._test_metrics_cache is not None and not force_reload:
            return self._test_metrics_cache

        if not self.test_metrics_path.exists():
            raise FileNotFoundError(f"Test metrics CSV not found at: {self.test_metrics_path}")

        df = pd.read_csv(self.test_metrics_path)

        # Parse DDPG Agent metrics
        ddpg_rows = df[df["Model"].str.contains("DDPG", case=False, na=False)]
        if ddpg_rows.empty:
            raise ValueError("DDPG Agent metrics missing from test_metrics.csv")
        d_row = ddpg_rows.iloc[0]

        ddpg_record = ModelMetricRecord(
            model="DDPG Agent",
            final_portfolio_value=float(d_row["final_portfolio_value"]),
            cumulative_return=float(d_row["cumulative_return"]),
            annualized_return=float(d_row["annualized_return"]),
            annualized_volatility=float(d_row["annualized_volatility"]),
            sharpe_ratio=float(d_row["sharpe_ratio"]),
            max_drawdown=float(d_row["max_drawdown"]),
            total_transaction_costs=float(d_row["total_transaction_costs"]),
        )

        # Parse Equal-Weight Benchmark metrics
        bench_rows = df[df["Model"].str.contains("Equal", case=False, na=False)]
        if bench_rows.empty:
            raise ValueError("Equal-Weight Benchmark metrics missing from test_metrics.csv")
        b_row = bench_rows.iloc[0]

        bench_record = ModelMetricRecord(
            model="Equal-Weight Benchmark",
            final_portfolio_value=float(b_row["final_portfolio_value"]),
            cumulative_return=float(b_row["cumulative_return"]),
            annualized_return=float(b_row["annualized_return"]),
            annualized_volatility=float(b_row["annualized_volatility"]),
            sharpe_ratio=float(b_row["sharpe_ratio"]),
            max_drawdown=float(b_row["max_drawdown"]),
            total_transaction_costs=float(b_row["total_transaction_costs"]),
        )

        response = BacktestMetricsResponse(
            evaluation_period=f"{TEST_START} to {TEST_END}",
            initial_capital=INITIAL_CAPITAL,
            ddpg=ddpg_record,
            equal_weight=bench_record,
        )
        self._test_metrics_cache = response
        return response

    def get_equity_curve(self, force_reload: bool = False) -> EquityCurveResponse:
        """Load and return existing daily portfolio trajectories from results/metrics/test_equity_curve.csv."""
        if self._equity_curve_cache is not None and not force_reload:
            return self._equity_curve_cache

        if not self.equity_curve_path.exists():
            raise FileNotFoundError(f"Test equity curve CSV not found at: {self.equity_curve_path}")

        df = pd.read_csv(self.equity_curve_path)
        points: List[EquityCurvePoint] = []
        for _, row in df.iterrows():
            points.append(
                EquityCurvePoint(
                    date=str(row["Date"]),
                    ddpg_value=float(row["DDPG_Portfolio_Value"]),
                    equal_weight_value=float(row["EqualWeight_Portfolio_Value"]),
                )
            )

        start_date = points[0].date if points else TEST_START
        end_date = points[-1].date if points else TEST_END

        response = EquityCurveResponse(
            initial_capital=INITIAL_CAPITAL,
            start_date=start_date,
            end_date=end_date,
            points_count=len(points),
            points=points,
        )
        self._equity_curve_cache = response
        return response

    def get_training_metrics(self, force_reload: bool = False) -> TrainingMetricsResponse:
        """Load and return training logs from results/metrics/training_metrics.csv."""
        if self._training_metrics_cache is not None and not force_reload:
            return self._training_metrics_cache

        if not self.training_metrics_path.exists():
            raise FileNotFoundError(f"Training metrics CSV not found at: {self.training_metrics_path}")

        df = pd.read_csv(self.training_metrics_path)
        episodes: List[TrainingEpisodeRecord] = []
        for _, row in df.iterrows():
            episodes.append(
                TrainingEpisodeRecord(
                    episode=int(row["episode"]),
                    total_reward=float(row["total_reward"]),
                    portfolio_value=float(row["portfolio_value"]),
                    cumulative_return=float(row["cumulative_return"]),
                    transaction_costs=float(row["transaction_costs"]),
                    mean_actor_loss=float(row["mean_actor_loss"]),
                    mean_critic_loss=float(row["mean_critic_loss"]),
                    duration_sec=float(row["duration_sec"]),
                )
            )

        response = TrainingMetricsResponse(
            total_episodes=len(episodes),
            episodes=episodes,
        )
        self._training_metrics_cache = response
        return response

    def get_validation_metrics(self, force_reload: bool = False) -> ValidationMetricsResponse:
        """Load and return periodic validation checkpoints from results/metrics/validation_metrics.csv."""
        if self._validation_metrics_cache is not None and not force_reload:
            return self._validation_metrics_cache

        if not self.validation_metrics_path.exists():
            raise FileNotFoundError(f"Validation metrics CSV not found at: {self.validation_metrics_path}")

        df = pd.read_csv(self.validation_metrics_path)
        checkpoints: List[ValidationCheckpointRecord] = []
        for _, row in df.iterrows():
            checkpoints.append(
                ValidationCheckpointRecord(
                    episode=int(row["episode"]),
                    val_final_portfolio_value=float(row["val_final_portfolio_value"]),
                    val_cumulative_return=float(row["val_cumulative_return"]),
                    val_annualized_return=float(row["val_annualized_return"]),
                    val_annualized_volatility=float(row["val_annualized_volatility"]),
                    val_sharpe_ratio=float(row["val_sharpe_ratio"]),
                    val_max_drawdown=float(row["val_max_drawdown"]),
                    val_transaction_costs=float(row["val_transaction_costs"]),
                )
            )

        response = ValidationMetricsResponse(checkpoints=checkpoints)
        self._validation_metrics_cache = response
        return response


# Global singleton instance
backtest_service = BacktestService()
