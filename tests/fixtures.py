from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


class TwoStepEnv:
    @property
    def unwrapped(self):
        return self

    spec = SimpleNamespace(max_episode_steps=1)

    def close(self):
        pass

    def reset(self, seed=None):
        self.index = 0
        return np.zeros(8, dtype=np.float32), {}

    def step(self, action):
        self.index += 1
        return (
            np.zeros(8, dtype=np.float32),
            float(self.index),
            self.index == 2,
            False,
            {},
        )


class ConstantCritic(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, obs, action):
        return torch.full((len(obs),), float(self.value))


def make_results():
    results = {}
    for seed in (1, 2, 3):
        for variant in ("baseline", "layer-norm"):
            rows = []
            for step in (10, 20):
                bias = float(seed * 10 + (step == 10))
                reward = -2 * bias
                if variant == "layer-norm":
                    bias -= 5
                    reward += 10
                for critic in ("q1", "q2", "min"):
                    rows.append(
                        {
                            "algorithm": "td3",
                            "variant": variant,
                            "seed": seed,
                            "step": step,
                            "critic": critic,
                            "bias": bias if critic == "q1" else 999.0,
                            "mae": abs(bias),
                            "q_mean": bias - 10,
                            "return_mean": -10.0,
                            "reward": reward,
                            "completed_episodes": 10,
                            "capped_episodes": 0,
                            "completion_rate": 1.0,
                            "score_horizon": 1000,
                            "evaluation_seed": 10000,
                            "evaluation_episodes": 10,
                            "mc_max_steps": 10000,
                        }
                    )
            results[("TD3", variant, seed)] = rows
    return results
