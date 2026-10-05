"""Exploration Noise Processes for Continuous DDPG Actions.

Supports:
  - GaussianNoise: Uncorrelated additive normal noise with optional linear/exponential decay.
  - OrnsteinUhlenbeckNoise: Temporally correlated mean-reverting process.
"""

from typing import Optional
import numpy as np


class GaussianNoise:
    """Additive Gaussian noise with standard deviation decay for exploration."""

    def __init__(
        self,
        action_dim: int = 5,
        std: float = 0.1,
        decay: float = 1.0,
        min_std: float = 0.01,
    ):
        self.action_dim = action_dim
        self.std = float(std)
        self.initial_std = float(std)
        self.decay = float(decay)
        self.min_std = float(min_std)

    def sample(self) -> np.ndarray:
        """Draw noise sample from N(0, std^2)."""
        noise = np.random.normal(loc=0.0, scale=self.std, size=self.action_dim)
        return noise.astype(np.float32)

    def step_decay(self) -> None:
        """Decay standard deviation per episode or step."""
        self.std = max(self.min_std, self.std * self.decay)

    def reset(self) -> None:
        """Reset noise process."""
        pass


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck mean-reverting exploration noise process."""

    def __init__(
        self,
        action_dim: int = 5,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1e-2,
    ):
        self.action_dim = action_dim
        self.mu = np.ones(self.action_dim, dtype=np.float32) * float(mu)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.dt = float(dt)
        self.state = np.zeros(self.action_dim, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        """Reset the internal state to mean mu."""
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        """Update and return the next internal state."""
        dx = (
            self.theta * (self.mu - self.state) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        )
        self.state = (self.state + dx).astype(np.float32)
        return self.state

    def step_decay(self) -> None:
        pass
