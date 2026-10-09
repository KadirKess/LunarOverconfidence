from itertools import pairwise

import torch
from rl_mind.core import Action, Actor
from torch import Tensor, nn


def build_network(sizes, layer_norm=False, output_activation=None):
    """Build an MLP with optional hidden-layer normalization before ReLU"""
    layers = []
    for index, (n_in, n_out) in enumerate(pairwise(sizes)):
        layers.append(nn.Linear(n_in, n_out))
        if index < len(sizes) - 2:
            if layer_norm:
                layers.append(nn.LayerNorm(n_out))
            layers.append(nn.ReLU())
        elif output_activation is not None:
            layers.append(output_activation)
    return nn.Sequential(*layers)


class ContinuousQNetwork(nn.Module):
    """Estimate Q(s, a), returning one value per observation-action pair"""

    def __init__(
        self,
        obs_dim: int,
        hidden: tuple[int, ...],
        action_dim: int,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.model = build_network([obs_dim + action_dim, *hidden, 1], layer_norm)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        """Map [batch, obs_dim] and [batch, action_dim] to [batch]"""
        return self.model(torch.cat([obs, action], dim=1)).squeeze(-1)


class ContinuousDeterministicActor(Actor[Action]):
    """Map batched observations to continuous actions in [-1, 1]"""

    def __init__(
        self,
        obs_dim: int,
        hidden: tuple[int, ...],
        action_dim: int,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.model = build_network(
            [obs_dim, *hidden, action_dim], layer_norm, output_activation=nn.Tanh()
        )

    def forward(self, obs: Tensor) -> Action:
        return Action(value=self.model(obs))


class GaussianNoise(Actor[Action]):
    """Add exploration noise during collection while keeping evaluation deterministic"""

    def __init__(self, actor: Actor[Action], sigma: float):
        super().__init__()
        self.actor = actor
        self.sigma = sigma

    def forward(self, obs: Tensor) -> Action:
        action = self.actor(obs).value
        noisy = action + self.sigma * torch.randn_like(action)
        # The replay buffer must contain the action actually executed by the environment
        return Action(value=noisy.clamp(-1.0, 1.0))

    def act(self, obs: Tensor) -> Tensor:
        return self.actor.act(obs)
