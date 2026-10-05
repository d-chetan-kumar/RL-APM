"""Comprehensive Test Suite & Rollout Verification for PortfolioEnvironment.

Validates:
 1. Environment initialization.
 2. Correct number of assets (5).
 3. Correct state dimension (50).
 4. Reset behavior restores initial values.
 5. Equal-weight initial portfolio (20% each).
 6. Action normalization maps arbitrary continuous inputs.
 7. Action weights sum to 1.0.
 8. No negative portfolio weights exist.
 9. One environment step executes correctly.
10. Portfolio value changes correctly according to market movement.
11. Transaction costs are accurately assessed on turnover.
12. Reward is finite and numeric.
13. No future data is used for the current decision (look-ahead bias check).
14. Environment reaches done=True at end of horizon.
"""

from pathlib import Path
import sys
from typing import Any, Dict
import unittest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.environment.portfolio_environment import (
    DEFAULT_TICKERS,
    PortfolioEnvironment,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class TestPortfolioEnvironment(unittest.TestCase):
    """Unit test cases verifying PortfolioEnvironment specifications."""

    @classmethod
    def setUpClass(cls):
        """Instantiate environment once for unit tests."""
        cls.env = PortfolioEnvironment(
            tickers=DEFAULT_TICKERS,
            initial_capital=100_000.0,
            transaction_cost=0.001,
        )

    def test_01_initialization(self):
        """1. Environment initializes with expected assets and horizon."""
        self.assertIsNotNone(self.env)
        self.assertGreaterEqual(self.env.num_periods, 2)
        self.assertEqual(len(self.env.dates), 2467)

    def test_02_correct_number_of_assets(self):
        """2. Correct number of assets is maintained."""
        self.assertEqual(self.env.num_assets, 5)
        self.assertEqual(self.env.tickers, ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"])

    def test_03_correct_state_dimension(self):
        """3. State dimension is exactly 50 (45 market indicators + 5 current weights)."""
        self.assertEqual(self.env.state_dim, 50)
        state, info = self.env.reset()
        self.assertEqual(state.shape, (50,))
        self.assertFalse(np.isnan(state).any(), "State contains NaNs")

    def test_04_reset_behavior(self):
        """4. Reset restores initial step, capital, and clears history."""
        # Step once
        action = [0.2, 0.2, 0.2, 0.2, 0.2]
        self.env.step(action)
        self.assertGreater(self.env.current_step, 0)

        # Reset
        state, info = self.env.reset()
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.portfolio_value, 100_000.0)
        self.assertEqual(self.env.cumulative_transaction_costs, 0.0)
        self.assertEqual(len(self.env.history), 0)
        self.assertEqual(state.shape, (50,))

    def test_05_equal_weight_initial_portfolio(self):
        """5. Initial portfolio is allocated with equal weights (20% each)."""
        self.env.reset()
        expected = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        np.testing.assert_allclose(self.env.current_weights, expected, atol=1e-5)

    def test_06_action_normalization(self):
        """6. Action normalization accepts arbitrary continuous logits."""
        arbitrary_actions = [
            np.array([-1.5, 2.0, 0.0, -0.5, 3.2]),
            np.array([10.0, 10.0, 10.0, 10.0, 10.0]),
            np.array([-10.0, -20.0, -5.0, 0.0, 5.0]),
        ]
        for act in arbitrary_actions:
            weights = self.env.normalize_action(act)
            self.assertEqual(len(weights), 5)
            self.assertTrue(np.all(weights >= 0.0))

    def test_07_action_weights_sum_to_one(self):
        """7. Normalized action weights sum strictly to 1.0."""
        random_actions = [np.random.randn(5) for _ in range(10)]
        for act in random_actions:
            w = self.env.normalize_action(act)
            self.assertAlmostEqual(float(np.sum(w)), 1.0, places=6)

    def test_08_no_negative_portfolio_weights(self):
        """8. No portfolio weight is negative after normalization."""
        large_negative_action = np.array([-100.0, -50.0, -10.0, -2.0, -1.0])
        w = self.env.normalize_action(large_negative_action)
        self.assertTrue(np.all(w >= 0.0))

    def test_09_one_environment_step_works(self):
        """9. Executing one step returns next_state, reward, done, and valid info."""
        self.env.reset()
        action = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        next_state, reward, done, info = self.env.step(action)

        self.assertEqual(next_state.shape, (50,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn("portfolio_value", info)
        self.assertIn("net_return", info)
        self.assertIn("turnover", info)
        self.assertIn("transaction_cost", info)

    def test_10_portfolio_value_changes_correctly(self):
        """10. Portfolio value responds predictably to market returns and fees."""
        self.env.reset()
        initial_val = self.env.portfolio_value
        action = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        _, _, _, info = self.env.step(action)
        new_val = info["portfolio_value"]

        # Expected value: (initial - cost) * (1 + gross_return)
        expected_val = (initial_val - info["transaction_cost"]) * (1.0 + info["gross_return"])
        self.assertAlmostEqual(new_val, expected_val, places=4)

    def test_11_transaction_costs_are_applied(self):
        """11. Transaction costs are properly assessed on non-zero turnover."""
        self.env.reset()
        # Rebalance completely into single asset (AAPL)
        action_all_aapl = np.array([10.0, -10.0, -10.0, -10.0, -10.0])
        _, _, _, info = self.env.step(action_all_aapl)

        # Turnover from [0.2, 0.2, 0.2, 0.2, 0.2] to ~[1.0, 0, 0, 0, 0] is ~1.6
        self.assertGreater(info["turnover"], 1.0)
        expected_cost = info["turnover"] * self.env.transaction_cost * 100_000.0
        self.assertAlmostEqual(info["transaction_cost"], expected_cost, places=2)
        self.assertGreater(self.env.cumulative_transaction_costs, 0.0)

    def test_12_reward_is_finite(self):
        """12. Reward is finite numeric value (not NaN or Inf)."""
        self.env.reset()
        action = np.random.randn(5)
        _, reward, _, _ = self.env.step(action)
        self.assertTrue(np.isfinite(reward))

    def test_13_no_look_ahead_bias(self):
        """13. State at step t contains only information up to date t, not t+1."""
        self.env.reset()
        t = 0
        state_0 = self.env._get_state()
        date_0 = self.env.dates[t]

        # Verify state_0 matches date_0 technical indicators exactly
        # Index 0 in state_0 should equal AAPL Daily_Return on date_0
        df_aapl = pd.read_csv(
            self.env.data_dir / "AAPL_processed.csv", parse_dates=["Date"], index_col="Date"
        )
        expected_aapl_ret_date0 = float(df_aapl.loc[date_0, "Daily_Return"])
        self.assertAlmostEqual(state_0[0], expected_aapl_ret_date0, places=5)

        # Step into t=1
        action = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        state_1, reward, done, info = self.env.step(action)
        date_1 = self.env.dates[t + 1]

        # Next state state_1 reflects date_1 return
        expected_aapl_ret_date1 = float(df_aapl.loc[date_1, "Daily_Return"])
        self.assertAlmostEqual(state_1[0], expected_aapl_ret_date1, places=5)

        # The gross return received during step was strictly based on the price transition between date_0 and date_1
        expected_gross_ret = 0.0
        for i, ticker in enumerate(self.env.tickers):
            p0 = self.env.adj_close_matrix[0, i]
            p1 = self.env.adj_close_matrix[1, i]
            asset_ret = (p1 - p0) / p0
            expected_gross_ret += action[i] * asset_ret

        self.assertAlmostEqual(info["gross_return"], expected_gross_ret, places=6)

    def test_14_environment_reaches_done(self):
        """14. Environment terminates when reaching the end of the specified horizon."""
        short_env = PortfolioEnvironment(
            tickers=DEFAULT_TICKERS,
            start_idx=0,
            end_idx=5,  # 5 periods -> 4 step transitions
        )
        short_env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = short_env.step([0.2, 0.2, 0.2, 0.2, 0.2])
            steps += 1
        self.assertTrue(done)
        self.assertEqual(steps, 4)


def run_rollout_demo(num_steps: int = 10) -> Dict[str, Any]:
    """Execute a 10-step deterministic rollout with equal-weight actions."""
    print("\n" + "=" * 60)
    print(f"   PORTFOLIO ENVIRONMENT: {num_steps}-STEP MANUAL ROLLOUT DEMO")
    print("=" * 60)

    env = PortfolioEnvironment(
        tickers=DEFAULT_TICKERS,
        initial_capital=100_000.0,
        transaction_cost=0.001,
    )
    state, info = env.reset()

    initial_portfolio_value = float(env.portfolio_value)
    initial_weights = dict(info["current_weights"])
    deterministic_action = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    total_reward = 0.0
    step_records = []

    print(f"\nInitial Date:            {info['current_date']}")
    print(f"Initial Portfolio Value: ${initial_portfolio_value:,.2f}")
    print(f"Initial Allocation:      {initial_weights}\n")
    print(f"{'Step':<5} | {'Date':<10} | {'Port Value':<12} | {'Net Return':<10} | {'Turnover':<9} | {'Cost ($)':<8} | {'Reward':<9}")
    print("-" * 75)

    for i in range(num_steps):
        next_state, reward, done, step_info = env.step(deterministic_action)
        total_reward += reward
        step_records.append(step_info)

        print(
            f"{step_info['step']:<5} | "
            f"{step_info['current_date']:<10} | "
            f"${step_info['portfolio_value']:<11,.2f} | "
            f"{step_info['net_return']:>9.4%} | "
            f"{step_info['turnover']:>8.4f} | "
            f"${step_info['transaction_cost']:>7.2f} | "
            f"{reward:>9.5f}"
        )
        if done:
            break

    summary = env.get_portfolio_summary()
    final_portfolio_value = float(summary["portfolio_value"])
    final_weights = dict(summary["current_weights"])
    total_tx_costs = float(summary["cumulative_transaction_costs"])

    print("-" * 75)
    print("\n--- Rollout Summary Results ---")
    print(f"Initial portfolio value:  ${initial_portfolio_value:,.2f}")
    print(f"Final portfolio value:    ${final_portfolio_value:,.2f}")
    print(f"Initial weights:          {initial_weights}")
    print(f"Final weights:            {final_weights}")
    print(f"Total reward:             {total_reward:.6f}")
    print(f"Number of steps:          {len(step_records)}")
    print(f"Total transaction costs:  ${total_tx_costs:,.2f}")
    print(f"Cumulative return:        {summary['cumulative_return']:+.4%}")
    print("=" * 60 + "\n")

    return {
        "initial_portfolio_value": initial_portfolio_value,
        "final_portfolio_value": final_portfolio_value,
        "initial_weights": initial_weights,
        "final_weights": final_weights,
        "total_reward": total_reward,
        "number_of_steps": len(step_records),
        "total_transaction_costs": total_tx_costs,
    }


if __name__ == "__main__":
    # 1. Run unit tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPortfolioEnvironment)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        sys.exit(1)

    # 2. Run rollout demo
    run_rollout_demo(10)
