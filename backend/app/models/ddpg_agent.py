"""Deep Deterministic Policy Gradient (DDPG) Agent for Portfolio Allocation.

Coordinates:
  - Online and Target Actor-Critic networks
  - Experience replay sampling
  - Bellman target computation and gradient optimization
  - Soft Polyak target updates (tau)
  - Action selection with continuous exploration noise
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from backend.app.models.networks import Actor, Critic
from backend.app.models.noise import GaussianNoise, OrnsteinUhlenbeckNoise
from backend.app.models.replay_buffer import ReplayBuffer


class DDPGAgent:
    """Standard continuous action Deep Deterministic Policy Gradient agent."""

    def __init__(
        self,
        state_dim: int = 50,
        action_dim: int = 5,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        noise_std: float = 0.1,
        noise_type: str = "gaussian",  # 'gaussian' or 'ou'
        device: Optional[Union[torch.device, str]] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize Online Networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)

        # Initialize Target Networks with identical initial weights
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=float(lr_actor))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=float(lr_critic))

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
        )

        # Exploration Noise
        if noise_type == "ou":
            self.noise = OrnsteinUhlenbeckNoise(action_dim=action_dim, sigma=noise_std)
        else:
            self.noise = GaussianNoise(action_dim=action_dim, std=noise_std)

    def select_action(
        self, state: np.ndarray, add_noise: bool = True
    ) -> np.ndarray:
        """Select action logits given the current state observation.

        Args:
            state: Array of shape (50,).
            add_noise: Whether to add exploration perturbation (False during eval).

        Returns:
            Continuous action vector of shape (5,).
        """
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            raw_action = self.actor(state_tensor).squeeze(0).cpu().numpy()
        self.actor.train()

        if add_noise:
            noise_sample = self.noise.sample()
            raw_action = raw_action + noise_sample

        return raw_action.astype(np.float32)

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store an experience tuple in the replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self) -> Optional[Dict[str, float]]:
        """Perform one training iteration updating Critic and Actor networks.

        Returns:
            Dictionary containing 'actor_loss', 'critic_loss', and 'q_mean' or None if buffer not ready.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # 1. Sample mini-batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 2. Compute Target Q-value using Target Networks
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q_next = self.critic_target(next_states, next_actions)
            # Bellman equation: y = r + gamma * (1 - done) * Q_target(s', mu_target(s'))
            target_q = rewards + (self.gamma * (1.0 - dones) * target_q_next)

        # 3. Update Critic
        current_q = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # 4. Update Actor: maximize expected Q-value (minimize -Q)
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # 5. Soft-update Target Networks: theta_target <- tau * theta + (1 - tau) * theta_target
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "q_mean": float(current_q.mean().item()),
        }

    def _soft_update(self, online_net: nn.Module, target_net: nn.Module) -> None:
        """Polyak averaging soft update of target network parameters."""
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    def save_checkpoint(self, actor_path: Union[str, Path], critic_path: Union[str, Path]) -> None:
        """Save network weights to disk."""
        Path(actor_path).parent.mkdir(parents=True, exist_ok=True)
        Path(critic_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_checkpoint(self, actor_path: Union[str, Path], critic_path: Union[str, Path]) -> None:
        """Load network weights from disk."""
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
