"""DDPG Out-of-Sample Test Evaluation Module for RL-APM.

Evaluates the saved best DDPG policy on the held-out TEST dataset (2024-01-01 to 2024-12-31),
comparing it against an Equal-Weight benchmark across risk-adjusted financial metrics.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Path resolution relative to project root
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.environment.portfolio_environment import PortfolioEnvironment
from backend.app.models.networks import Actor

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

TEST_START = "2024-01-01"
TEST_END = "2024-12-31"


def compute_performance_metrics(
    portfolio_values: List[float],
    daily_returns: List[float],
    transaction_costs: float,
    initial_capital: float = 100_000.0,
) -> Dict[str, float]:
    """Compute financial evaluation metrics from a backtest trajectory."""
    vals = np.array(portfolio_values, dtype=np.float64)
    rets = np.array(daily_returns, dtype=np.float64)

    final_val = float(vals[-1])
    cum_return = float((final_val / initial_capital) - 1.0)
    n_days = len(rets)

    # Annualized return (compound)
    if n_days > 0 and final_val > 0:
        ann_return = float((final_val / initial_capital) ** (252.0 / n_days) - 1.0)
    else:
        ann_return = 0.0

    # Annualized volatility
    ann_vol = float(np.std(rets) * np.sqrt(252)) if len(rets) > 1 else 0.0

    # Sharpe ratio (risk-free rate = 0.0)
    sharpe = float(ann_return / (ann_vol + 1e-8)) if ann_vol > 0 else 0.0

    # Maximum Drawdown
    peaks = np.maximum.accumulate(vals)
    drawdowns = (peaks - vals) / peaks
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    return {
        "final_portfolio_value": final_val,
        "cumulative_return": cum_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "total_transaction_costs": float(transaction_costs),
    }


def evaluate_test_period(
    actor_weights_path: Path = MODELS_DIR / "ddpg_actor_best.pth",
    test_start: str = TEST_START,
    test_end: str = TEST_END,
) -> Tuple[Dict[str, float], Dict[str, float], pd.DataFrame]:
    """Run out-of-sample evaluation on the 2024 test period.

    Args:
        actor_weights_path: Path to saved best Actor model checkpoint.
        test_start: Start date for test horizon.
        test_end: End date for test horizon.

    Returns:
        Tuple of (ddpg_metrics_dict, benchmark_metrics_dict, trajectory_df).
    """
    if not actor_weights_path.exists():
        raise FileNotFoundError(f"Actor checkpoint not found at: {actor_weights_path}")

    # Initialize Test Environment
    test_env = PortfolioEnvironment(start_date=test_start, end_date=test_end)
    benchmark_env = PortfolioEnvironment(start_date=test_start, end_date=test_end)

    # Load Actor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(test_env.state_dim, test_env.action_dim).to(device)
    actor.load_state_dict(torch.load(actor_weights_path, map_location=device))
    actor.eval()

    # 1. Run DDPG Policy
    state, info = test_env.reset()
    done = False
    ddpg_values = [test_env.portfolio_value]
    ddpg_returns = []
    dates = [test_env.dates[0].strftime("%Y-%m-%d")]
    ddpg_weights_history = []

    while not done:
        with torch.no_grad():
            s_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            raw_action = actor(s_t).squeeze(0).cpu().numpy()

        next_state, reward, done, step_info = test_env.step(raw_action)
        ddpg_values.append(step_info["portfolio_value"])
        ddpg_returns.append(step_info["net_return"])
        dates.append(step_info["current_date"])
        ddpg_weights_history.append(step_info["target_weights"])
        state = next_state

    ddpg_metrics = compute_performance_metrics(
        ddpg_values, ddpg_returns, test_env.cumulative_transaction_costs
    )

    # 2. Run Equal-Weight Benchmark
    state_b, info_b = benchmark_env.reset()
    done_b = False
    bench_values = [benchmark_env.portfolio_value]
    bench_returns = []
    equal_action = np.ones(benchmark_env.num_assets) / benchmark_env.num_assets

    while not done_b:
        next_state_b, reward_b, done_b, step_info_b = benchmark_env.step(equal_action)
        bench_values.append(step_info_b["portfolio_value"])
        bench_returns.append(step_info_b["net_return"])
        state_b = next_state_b

    bench_metrics = compute_performance_metrics(
        bench_values, bench_returns, benchmark_env.cumulative_transaction_costs
    )

    # Construct comparative trajectory DataFrame
    df_traj = pd.DataFrame({
        "Date": dates,
        "DDPG_Portfolio_Value": ddpg_values,
        "EqualWeight_Portfolio_Value": bench_values,
    })
    df_traj.set_index("Date", inplace=True)

    # Print comparative results
    print("\n" + "=" * 65)
    print("      RL-APM: OUT-OF-SAMPLE TEST EVALUATION (2024)")
    print("=" * 65)
    print(f"{'Metric':<30} | {'DDPG Agent':<15} | {'Equal-Weight':<15}")
    print("-" * 65)
    print(f"{'Final Portfolio Value':<30} | ${ddpg_metrics['final_portfolio_value']:<14,.2f} | ${bench_metrics['final_portfolio_value']:<14,.2f}")
    print(f"{'Cumulative Return':<30} | {ddpg_metrics['cumulative_return']:>14.2%} | {bench_metrics['cumulative_return']:>14.2%}")
    print(f"{'Annualized Return':<30} | {ddpg_metrics['annualized_return']:>14.2%} | {bench_metrics['annualized_return']:>14.2%}")
    print(f"{'Annualized Volatility':<30} | {ddpg_metrics['annualized_volatility']:>14.2%} | {bench_metrics['annualized_volatility']:>14.2%}")
    print(f"{'Sharpe Ratio':<30} | {ddpg_metrics['sharpe_ratio']:>14.2f} | {bench_metrics['sharpe_ratio']:>14.2f}")
    print(f"{'Maximum Drawdown':<30} | {ddpg_metrics['max_drawdown']:>14.2%} | {bench_metrics['max_drawdown']:>14.2%}")
    print(f"{'Total Transaction Costs':<30} | ${ddpg_metrics['total_transaction_costs']:<14,.2f} | ${bench_metrics['total_transaction_costs']:<14,.2f}")
    print("=" * 65 + "\n")

    # Save metrics to CSV
    metrics_records = [
        {"Model": "DDPG Agent", **ddpg_metrics},
        {"Model": "Equal-Weight Benchmark", **bench_metrics},
    ]
    df_metrics = pd.DataFrame(metrics_records)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    test_metrics_path = METRICS_DIR / "test_metrics.csv"
    df_metrics.to_csv(test_metrics_path, index=False)
    print(f"Saved test metrics to {test_metrics_path}")

    # Save trajectory to CSV for frontend and API equity curve endpoints
    test_equity_curve_path = METRICS_DIR / "test_equity_curve.csv"
    df_traj.to_csv(test_equity_curve_path)
    print(f"Saved test equity curve trajectory to {test_equity_curve_path}")

    # Plot Out-of-Sample Equity Curves
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_traj.index, df_traj["DDPG_Portfolio_Value"], label="DDPG Policy", color="#1f77b4", linewidth=2.0)
    ax.plot(df_traj.index, df_traj["EqualWeight_Portfolio_Value"], label="Equal-Weight Benchmark", color="#ff7f0e", linestyle="--", linewidth=1.8)
    ax.set_title("Out-of-Sample Test Performance: DDPG vs. Equal-Weight (2024)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (USD)")
    # Sample x-ticks so labels don't overlap
    n_ticks = 8
    step_tick = max(1, len(df_traj) // n_ticks)
    ax.set_xticks(df_traj.index[::step_tick])
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    chart_path = FIGURES_DIR / "test_equity_curve.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Saved test equity curve plot to {chart_path}\n")

    return ddpg_metrics, bench_metrics, df_traj


if __name__ == "__main__":
    evaluate_test_period()
