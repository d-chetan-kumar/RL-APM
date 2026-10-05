"""DDPG Smoke Test and Component Verification.

Confirms:
  1. Actor forward pass works.
  2. Critic forward pass works.
  3. Replay buffer storage & sampling work.
  4. Backpropagation & gradient optimization work.
  5. Target networks soft-update parameters.
  6. Model checkpoint saving & loading work.
  7. 2-episode training rollout with metric generation.
"""

from pathlib import Path
import sys
import numpy as np
import torch

# Path resolution relative to project root
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.environment.portfolio_environment import PortfolioEnvironment
from backend.app.models.ddpg_agent import DDPGAgent
from backend.app.training.train_ddpg import train_ddpg


def run_smoke_test() -> bool:
    """Execute smoke test verifying all core DDPG mechanisms."""
    print("=" * 60)
    print("        RL-APM: DDPG AGENT SMOKE TEST VERIFICATION        ")
    print("=" * 60)

    # 1. Initialize environment on small slice
    env = PortfolioEnvironment(start_idx=0, end_idx=100)
    state_dim = env.state_dim
    action_dim = env.action_dim
    print(f"[CHECK 1] Environment initialized: state_dim={state_dim}, action_dim={action_dim}")

    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_capacity=1000,
        batch_size=16,
    )

    # 2. Test Actor forward pass
    state, info = env.reset()
    state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    actor_out = agent.actor(state_tensor)
    assert actor_out.shape == (1, action_dim), f"Actor output shape mismatch: {actor_out.shape}"
    print(f"[CHECK 2] Actor forward pass PASSED: shape {actor_out.shape}")

    # 3. Test Critic forward pass
    critic_out = agent.critic(state_tensor, actor_out)
    assert critic_out.shape == (1, 1), f"Critic output shape mismatch: {critic_out.shape}"
    print(f"[CHECK 3] Critic forward pass PASSED: shape {critic_out.shape}")

    # 4. Fill replay buffer with dummy transitions & test sampling
    for i in range(25):
        s = np.random.randn(state_dim).astype(np.float32)
        a = np.random.randn(action_dim).astype(np.float32)
        r = float(np.random.randn())
        s_next = np.random.randn(state_dim).astype(np.float32)
        d = bool(i == 24)
        agent.store_transition(s, a, r, s_next, d)

    assert len(agent.replay_buffer) == 25
    b_s, b_a, b_r, b_s_next, b_d = agent.replay_buffer.sample(16)
    assert b_s.shape == (16, state_dim)
    assert b_a.shape == (16, action_dim)
    print(f"[CHECK 4] Replay buffer storage and sampling PASSED: buffer size {len(agent.replay_buffer)}")

    # 5. Test backpropagation and target soft-update
    critic_param_before = [p.clone() for p in agent.critic.parameters()]
    target_param_before = [p.clone() for p in agent.critic_target.parameters()]

    update_metrics = agent.update()
    assert update_metrics is not None, "Agent update returned None"
    assert "actor_loss" in update_metrics and "critic_loss" in update_metrics
    assert np.isfinite(update_metrics["actor_loss"]) and np.isfinite(update_metrics["critic_loss"])

    critic_changed = any(
        not torch.equal(p_b, p_a)
        for p_b, p_a in zip(critic_param_before, agent.critic.parameters())
    )
    target_changed = any(
        not torch.equal(t_b, t_a)
        for t_b, t_a in zip(target_param_before, agent.critic_target.parameters())
    )
    assert critic_changed, "Critic parameters did not update during backpropagation"
    assert target_changed, "Target parameters did not soft-update"
    print(
        f"[CHECK 5] Backpropagation & Soft Target Update PASSED: "
        f"Actor Loss={update_metrics['actor_loss']:.4f}, Critic Loss={update_metrics['critic_loss']:.4f}"
    )

    # 6. Test checkpoint saving and loading
    tmp_actor_path = PROJECT_ROOT / "models" / "_tmp_smoke_actor.pth"
    tmp_critic_path = PROJECT_ROOT / "models" / "_tmp_smoke_critic.pth"
    agent.save_checkpoint(tmp_actor_path, tmp_critic_path)
    assert tmp_actor_path.exists() and tmp_critic_path.exists()

    agent_loaded = DDPGAgent(state_dim, action_dim)
    agent_loaded.load_checkpoint(tmp_actor_path, tmp_critic_path)
    # Cleanup temp test files
    tmp_actor_path.unlink()
    tmp_critic_path.unlink()
    print("[CHECK 6] Checkpoint save & load PASSED")

    # 7. Run a 2-episode training smoke test with validation
    print("[CHECK 7] Executing 2-episode training loop...")
    agent_trained, df_tr, df_v = train_ddpg(
        num_episodes=2,
        eval_interval=1,
        batch_size=16,
        train_start="2015-03-16",
        train_end="2015-06-30",  # Short span for rapid smoke verification
        val_start="2015-07-01",
        val_end="2015-08-31",
    )
    assert len(df_tr) == 2, f"Expected 2 training records, got {len(df_tr)}"
    assert len(df_v) == 2, f"Expected 2 validation records, got {len(df_v)}"
    print("[CHECK 7] 2-episode training loop & metrics logging PASSED")

    print("\n" + "=" * 60)
    print("           ALL DDPG SMOKE TESTS PASSED SUCCESSFULLY!          ")
    print("=" * 60 + "\n")
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
