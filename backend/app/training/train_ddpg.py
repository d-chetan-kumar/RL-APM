"""DDPG Training Pipeline for Portfolio Optimization.

Implements episode-based training with:
  - Chronological Train/Validation splitting
  - Actor-Critic optimization with experience replay
  - Periodic deterministic validation
  - Model checkpoint saving (best & final)
  - Metrics logging to CSV
  - Real performance curve visualization with Matplotlib
"""

import argparse
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
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

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backend.app.environment.portfolio_environment import PortfolioEnvironment
from backend.app.models.ddpg_agent import DDPGAgent

# Directory paths
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

# Chronological split definitions
TRAIN_START = "2015-03-16"
TRAIN_END = "2021-12-31"
VAL_START = "2022-01-01"
VAL_END = "2023-12-31"


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_on_validation(
    agent: DDPGAgent,
    val_env: PortfolioEnvironment,
) -> Dict[str, float]:
    """Evaluate current policy on validation environment without noise or training."""
    state, info = val_env.reset()
    done = False
    daily_returns: List[float] = []

    while not done:
        # Deterministic action without exploration noise
        action = agent.select_action(state, add_noise=False)
        next_state, reward, done, step_info = val_env.step(action)
        daily_returns.append(step_info["net_return"])
        state = next_state

    val_summary = val_env.get_portfolio_summary()
    final_val = val_summary["portfolio_value"]
    cum_return = val_summary["cumulative_return"]
    tx_costs = val_summary["cumulative_transaction_costs"]

    # Compute financial risk metrics
    returns_arr = np.array(daily_returns, dtype=np.float64)
    volatility = float(np.std(returns_arr) * np.sqrt(252)) if len(returns_arr) > 1 else 0.0
    mean_ret = float(np.mean(returns_arr) * 252) if len(returns_arr) > 0 else 0.0
    sharpe = float(mean_ret / (volatility + 1e-8)) if volatility > 0 else 0.0

    # Maximum drawdown computation
    history_values = [h["portfolio_value"] for h in val_env.history]
    if history_values:
        peaks = np.maximum.accumulate(history_values)
        drawdowns = (peaks - history_values) / peaks
        max_dd = float(np.max(drawdowns))
    else:
        max_dd = 0.0

    return {
        "val_final_portfolio_value": float(final_val),
        "val_cumulative_return": float(cum_return),
        "val_annualized_return": float(mean_ret),
        "val_annualized_volatility": float(volatility),
        "val_sharpe_ratio": float(sharpe),
        "val_max_drawdown": float(max_dd),
        "val_transaction_costs": float(tx_costs),
    }


def train_ddpg(
    num_episodes: int = 50,
    eval_interval: int = 5,
    batch_size: int = 64,
    lr_actor: float = 1e-4,
    lr_critic: float = 1e-3,
    gamma: float = 0.99,
    tau: float = 0.005,
    noise_std: float = 0.1,
    seed: int = 42,
    train_start: str = TRAIN_START,
    train_end: str = TRAIN_END,
    val_start: str = VAL_START,
    val_end: str = VAL_END,
) -> Tuple[DDPGAgent, pd.DataFrame, pd.DataFrame]:
    """Execute complete DDPG training loop on the portfolio environment.

    Args:
        num_episodes: Number of training episodes.
        eval_interval: Frequency of validation evaluations (in episodes).
        batch_size: Mini-batch size for experience replay updates.
        lr_actor: Learning rate for Actor Adam optimizer.
        lr_critic: Learning rate for Critic Adam optimizer.
        gamma: Bellman discount factor.
        tau: Soft target update coefficient.
        noise_std: Standard deviation for Gaussian exploration noise.
        seed: Random seed.
        train_start: Start date for training split.
        train_end: End date for training split.
        val_start: Start date for validation split.
        val_end: End date for validation split.

    Returns:
        Tuple of (agent, train_metrics_df, val_metrics_df).
    """
    set_seed(seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("           RL-APM: DDPG PORTFOLIO TRAINING PIPELINE           ")
    print("=" * 65)
    print(f"Seed:               {seed}")
    print(f"Episodes:           {num_episodes}")
    print(f"Batch Size:         {batch_size}")
    print(f"Actor LR / Critic:  {lr_actor} / {lr_critic}")
    print(f"Gamma / Tau:        {gamma} / {tau}")
    print(f"Exploration Noise:  Gaussian(std={noise_std})")
    print(f"Train Period:       {train_start} to {train_end}")
    print(f"Val Period:         {val_start} to {val_end}")

    # Initialize isolated environments
    train_env = PortfolioEnvironment(start_date=train_start, end_date=train_end)
    val_env = PortfolioEnvironment(start_date=val_start, end_date=val_end)

    print(f"Train Steps / Ep:   {train_env.num_periods - 1}")
    print(f"Val Steps / Ep:     {val_env.num_periods - 1}")
    print("=" * 65 + "\n")

    # Initialize agent
    agent = DDPGAgent(
        state_dim=train_env.state_dim,
        action_dim=train_env.action_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        tau=tau,
        batch_size=batch_size,
        noise_std=noise_std,
    )

    training_logs: List[Dict[str, Any]] = []
    validation_logs: List[Dict[str, Any]] = []

    best_val_return = -float("inf")
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        ep_start_time = time.time()
        state, info = train_env.reset()
        done = False

        ep_reward = 0.0
        actor_losses = []
        critic_losses = []

        while not done:
            action = agent.select_action(state, add_noise=True)
            next_state, reward, done, step_info = train_env.step(action)

            # Store transition in replay buffer
            agent.store_transition(state, action, reward, next_state, done)

            # Gradient update step
            update_res = agent.update()
            if update_res is not None:
                actor_losses.append(update_res["actor_loss"])
                critic_losses.append(update_res["critic_loss"])

            state = next_state
            ep_reward += reward

        ep_duration = time.time() - ep_start_time
        summary = train_env.get_portfolio_summary()

        mean_actor_loss = float(np.mean(actor_losses)) if actor_losses else 0.0
        mean_critic_loss = float(np.mean(critic_losses)) if critic_losses else 0.0

        train_record = {
            "episode": episode,
            "total_reward": float(ep_reward),
            "portfolio_value": float(summary["portfolio_value"]),
            "cumulative_return": float(summary["cumulative_return"]),
            "transaction_costs": float(summary["cumulative_transaction_costs"]),
            "mean_actor_loss": mean_actor_loss,
            "mean_critic_loss": mean_critic_loss,
            "duration_sec": float(ep_duration),
        }
        training_logs.append(train_record)
        pd.DataFrame(training_logs).to_csv(METRICS_DIR / "training_metrics.csv", index=False)

        print(
            f"Ep [{episode:02d}/{num_episodes:02d}] "
            f"Reward: {ep_reward:>8.4f} | "
            f"Port Val: ${summary['portfolio_value']:>10,.2f} ({summary['cumulative_return']:>+6.2%}) | "
            f"Loss(A/C): {mean_actor_loss:>6.3f} / {mean_critic_loss:>6.4f} | "
            f"Time: {ep_duration:>4.1f}s",
            flush=True,
        )

        # Periodic Validation Check
        if episode % eval_interval == 0 or episode == num_episodes:
            val_metrics = evaluate_on_validation(agent, val_env)
            val_metrics["episode"] = episode
            validation_logs.append(val_metrics)
            pd.DataFrame(validation_logs).to_csv(METRICS_DIR / "validation_metrics.csv", index=False)

            print(
                f"  >>> [VAL Ep {episode:02d}] "
                f"Val Port Val: ${val_metrics['val_final_portfolio_value']:,.2f} | "
                f"Return: {val_metrics['val_cumulative_return']:>+6.2%} | "
                f"Sharpe: {val_metrics['val_sharpe_ratio']:>5.2f} | "
                f"MaxDD: {val_metrics['val_max_drawdown']:>5.2%}",
                flush=True,
            )

            # Check if this is the best validation model
            if val_metrics["val_cumulative_return"] > best_val_return:
                best_val_return = val_metrics["val_cumulative_return"]
                actor_best_path = MODELS_DIR / "ddpg_actor_best.pth"
                critic_best_path = MODELS_DIR / "ddpg_critic_best.pth"
                agent.save_checkpoint(actor_best_path, critic_best_path)
                print(f"  >>> NEW BEST MODEL saved to {actor_best_path.name}", flush=True)

            generate_training_plots(pd.DataFrame(training_logs), pd.DataFrame(validation_logs), FIGURES_DIR)

    total_time = time.time() - start_time
    print("\n" + "=" * 65, flush=True)
    print(f"Training completed in {total_time:.1f} seconds ({total_time / 60:.2f} mins).", flush=True)
    print(f"Best Validation Return: {best_val_return:+.2%}", flush=True)
    print("=" * 65 + "\n", flush=True)

    # Save final model checkpoints
    agent.save_checkpoint(
        MODELS_DIR / "ddpg_actor_final.pth",
        MODELS_DIR / "ddpg_critic_final.pth",
    )

    df_train = pd.DataFrame(training_logs)
    df_val = pd.DataFrame(validation_logs)
    df_train.to_csv(METRICS_DIR / "training_metrics.csv", index=False)
    df_val.to_csv(METRICS_DIR / "validation_metrics.csv", index=False)
    generate_training_plots(df_train, df_val, FIGURES_DIR)

    return agent, df_train, df_val


def generate_training_plots(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate and save matplotlib plots of training and validation trajectories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Training Reward Curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_train["episode"], df_train["total_reward"], marker="o", color="#1f77b4", label="Episode Reward")
    ax.set_title("DDPG Training: Episode Total Reward", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "training_rewards.png", dpi=150)
    plt.close(fig)

    # 2. Training Final Portfolio Value Curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_train["episode"], df_train["portfolio_value"], marker="s", color="#2ca02c", label="Final Portfolio Value")
    ax.axhline(100_000, color="gray", linestyle="--", label="Initial Capital ($100k)")
    ax.set_title("DDPG Training: Final Episode Portfolio Value", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "training_portfolio_value.png", dpi=150)
    plt.close(fig)

    # 3. Actor Loss Curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_train["episode"], df_train["mean_actor_loss"], marker="^", color="#d62728", label="Actor Loss (-Q)")
    ax.set_title("DDPG Training: Actor Policy Loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "actor_loss.png", dpi=150)
    plt.close(fig)

    # 4. Critic Loss Curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_train["episode"], df_train["mean_critic_loss"], marker="d", color="#9467bd", label="Critic MSE Loss")
    ax.set_title("DDPG Training: Critic Bellman Loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "critic_loss.png", dpi=150)
    plt.close(fig)

    # 5. Validation Performance Curve
    if not df_val.empty:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        color = "#ff7f0e"
        ax1.plot(df_val["episode"], df_val["val_cumulative_return"] * 100, marker="o", color=color, label="Val Return (%)")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Validation Cumulative Return (%)", color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color2 = "#17becf"
        ax2.plot(df_val["episode"], df_val["val_sharpe_ratio"], marker="x", linestyle="--", color=color2, label="Val Sharpe")
        ax2.set_ylabel("Validation Sharpe Ratio", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

        plt.title("DDPG Validation: Return & Sharpe Trajectory", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(output_dir / "validation_performance.png", dpi=150)
        plt.close(fig)

    print(f"Generated training plots in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPG Portfolio Agent")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--eval-interval", type=int, default=5, help="Validation evaluation frequency")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for experience replay")
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="Actor learning rate")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="Critic learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_ddpg(
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        batch_size=args.batch_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        seed=args.seed,
    )
