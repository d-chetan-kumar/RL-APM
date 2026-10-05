"""Experience Replay Buffer for DDPG.

Stores transitions (s, a, r, s', d) in a pre-allocated circular buffer for
efficient random mini-batch sampling.
"""

from typing import Tuple, Union
import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity buffer storing experience tuples for off-policy training."""

    def __init__(
        self,
        capacity: int = 100_000,
        state_dim: int = 50,
        action_dim: int = 5,
        device: Union[torch.device, str] = "cpu",
    ):
        self.capacity = int(capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        # Pre-allocate numpy arrays for memory efficiency and fast indexing
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr: int = 0
        self.size: int = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition tuple to the buffer."""
        self.states[self.ptr] = np.asarray(state, dtype=np.float32).flatten()
        self.actions[self.ptr] = np.asarray(action, dtype=np.float32).flatten()
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr] = np.asarray(next_state, dtype=np.float32).flatten()
        self.dones[self.ptr] = 1.0 if done else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int = 64
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample a mini-batch of experiences as PyTorch tensors.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones).
        """
        if self.size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions; buffer only contains {self.size}."
            )

        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.as_tensor(self.states[indices], device=self.device)
        actions = torch.as_tensor(self.actions[indices], device=self.device)
        rewards = torch.as_tensor(self.rewards[indices], device=self.device)
        next_states = torch.as_tensor(self.next_states[indices], device=self.device)
        dones = torch.as_tensor(self.dones[indices], device=self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size
