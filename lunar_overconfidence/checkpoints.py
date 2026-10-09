from dataclasses import asdict
from pathlib import Path

import torch

from lunar_overconfidence.config import DDPGConfig, TD3Config
from lunar_overconfidence.networks import (
    ContinuousDeterministicActor,
    ContinuousQNetwork,
)


def save_checkpoint(run_dir, algorithm, cfg, step, actor, critics, obs_dim, action_dim):
    """Save online weights and configuration for evaluation and return the file path"""
    directory = Path(run_dir) / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step-{step:09d}.pt"
    torch.save(
        {
            "algorithm": algorithm,
            "config": asdict(cfg),
            "step": step,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "actor": actor.state_dict(),
            "critics": [critic.state_dict() for critic in critics],
        },
        path,
    )
    return path


def load_checkpoint(path):
    """Return checkpoint metadata, config, actor, and critics on the CPU"""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    Config = {"ddpg": DDPGConfig, "td3": TD3Config}[payload["algorithm"]]
    cfg = Config(**payload["config"])
    actor = ContinuousDeterministicActor(
        payload["obs_dim"],
        cfg.actor_hidden,
        payload["action_dim"],
        layer_norm=cfg.actor_layer_norm,
    )
    actor.load_state_dict(payload["actor"])
    actor.eval()
    critics = []
    for state in payload["critics"]:
        critic = ContinuousQNetwork(
            payload["obs_dim"],
            cfg.critic_hidden,
            payload["action_dim"],
            layer_norm=cfg.layer_norm,
        )
        critic.load_state_dict(state)
        critic.eval()
        critics.append(critic)
    return payload, cfg, actor, critics
