"""Neural Network Architectures for DDPG Portfolio Agent.

Implements:
  - Actor: State (50) -> Continuous Action Logits (5)
  - Critic: State (50) + Action (5) -> Q-value (1)
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def fanin_init(size: Tuple[int, ...], fanin: int = None) -> torch.Tensor:
    """Xavier / Fan-in initialization bound."""
    fanin = fanin or size[0]
    v = 1.0 / (fanin ** 0.5)
    return torch.empty(size).uniform_(-v, v)


class Actor(nn.Module):
    """Actor (Policy) Network for Continuous Portfolio Allocation.

    Architecture:
      Input (50) -> Linear(256) -> LayerNorm -> ReLU
                 -> Linear(256) -> LayerNorm -> ReLU
                 -> Linear(5) -> Raw Continuous Logits

    Note on Output Normalization:
      The Actor outputs unconstrained continuous action values (logits).
      The PortfolioEnvironment applies numerically stable Softmax normalization
      to transform these logits into valid portfolio allocation weights
      (non-negative and summing to 1). Therefore, no activation like Tanh or
      Softmax is applied at the network head, preventing redundant or conflicting
      transformations.
    """

    def __init__(
        self,
        state_dim: int = 50,
        action_dim: int = 5,
        hidden_dim: int = 256,
        init_w: float = 3e-3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self._init_weights(init_w)

    def _init_weights(self, init_w: float) -> None:
        self.fc1.weight.data.uniform_(*(-1.0 / (self.state_dim ** 0.5), 1.0 / (self.state_dim ** 0.5)))
        self.fc2.weight.data.uniform_(*(-1.0 / (256 ** 0.5), 1.0 / (256 ** 0.5)))
        # Final layer initialized with small weights so initial action logits are near zero (equal weights)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass generating continuous action logits."""
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        logits = self.fc3(x)
        return logits


class Critic(nn.Module):
    """Critic (Q-Value) Network.

    Evaluates the quality of allocating action 'a' in state 's'.

    Architecture:
      State Pathway:  Input (50) -> Linear(256) -> LayerNorm -> ReLU -> State Features (256)
      Merge Pathway:  Concat(State Features [256], Action [5]) -> Vector (261)
                      -> Linear(261, 256) -> LayerNorm -> ReLU
                      -> Linear(256, 1) -> Scalar Q-value
    """

    def __init__(
        self,
        state_dim: int = 50,
        action_dim: int = 5,
        hidden_dim: int = 256,
        init_w: float = 3e-3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # State feature extraction
        self.fc_state = nn.Linear(state_dim, hidden_dim)
        self.ln_state = nn.LayerNorm(hidden_dim)

        # Combined state-action processing
        self.fc_combined = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.ln_combined = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

        self._init_weights(init_w)

    def _init_weights(self, init_w: float) -> None:
        self.fc_state.weight.data.uniform_(*(-1.0 / (self.state_dim ** 0.5), 1.0 / (self.state_dim ** 0.5)))
        self.fc_combined.weight.data.uniform_(*(-1.0 / ((256 + self.action_dim) ** 0.5), 1.0 / ((256 + self.action_dim) ** 0.5)))
        self.fc_out.weight.data.uniform_(-init_w, init_w)
        self.fc_out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass computing Q(s, a)."""
        s = F.relu(self.ln_state(self.fc_state(state)))
        sa = torch.cat([s, action], dim=-1)
        q = F.relu(self.ln_combined(self.fc_combined(sa)))
        q_val = self.fc_out(q)
        return q_val
