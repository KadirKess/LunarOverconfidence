import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
import rl_mind.envs  # noqa: F401
import torch

from .checkpoints import load_checkpoint


def discounted_returns(rewards, gamma):
    """Return the discounted suffix beginning with each action's reward"""
    returns = np.empty(len(rewards), dtype=np.float64)
    value = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        value = float(rewards[index]) + gamma * value
        returns[index] = value
    return returns


def monte_carlo_evaluation(
    env,
    actor,
    critics,
    gamma,
    n_episodes=100,
    seed=10_000,
    max_steps=10_000,
    score_horizon=None,
):
    """Measure episode-weighted Q errors using complete policy trajectories

    Pass one ordinary Gymnasium environment without an external time limit
    Capped trajectories have no complete Monte Carlo reference and get NaN errors
    score_horizon restricts scored rewards and sampled state-action pairs
    Rewards after that horizon still contribute to earlier discounted returns
    """
    if n_episodes <= 0 or max_steps <= 0 or not critics:
        raise ValueError("Need positive episode/step counts and at least one critic")
    labels = ["q1"] if len(critics) == 1 else ["q1", "q2", "min"]
    if len(critics) not in (1, 2):
        raise ValueError("Expected one DDPG critic or two TD3 critics")
    shape = (n_episodes, len(labels))
    result = {name: np.full(shape, np.nan) for name in ("bias", "mae", "q_mean")}
    result.update(
        critic_labels=np.array(labels),
        reward=np.zeros(n_episodes),
        return_mean=np.full(n_episodes, np.nan),
        length=np.zeros(n_episodes, dtype=int),
        completed=np.zeros(n_episodes, dtype=bool),
        episode_seed=np.arange(seed, seed + n_episodes),
    )
    modules = [actor, *critics]
    modes = [module.training for module in modules]
    for module in modules:
        module.eval()
    try:
        with torch.no_grad():
            for episode in range(n_episodes):
                obs, _ = env.reset(seed=seed + episode)
                rewards, predictions = [], []
                for timestep in range(max_steps):
                    observation = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    action = actor.act(observation)
                    values = [critic(observation, action).item() for critic in critics]
                    if len(critics) == 2:
                        values.append(min(values))
                    predictions.append(values)
                    obs, reward, terminated, truncated, _ = env.step(
                        action[0].cpu().numpy()
                    )
                    rewards.append(float(reward))
                    if score_horizon is None or timestep < score_horizon:
                        result["reward"][episode] += float(reward)
                    if terminated or truncated:
                        result["completed"][episode] = bool(terminated)
                        break
                result["length"][episode] = len(rewards)
                if not result["completed"][episode]:
                    # Missing continuation is unknown, and excluding episodes can skew the estimate
                    continue
                # Compute continuation returns before restricting the measured states
                returns = discounted_returns(rewards, gamma)[:score_horizon]
                q_values = np.asarray(predictions)[:score_horizon]
                # Each prediction and suffix share the same state and executed action
                errors = q_values - returns[:, None]
                result["bias"][episode] = errors.mean(axis=0)
                result["mae"][episode] = np.abs(errors).mean(axis=0)
                result["q_mean"][episode] = q_values.mean(axis=0)
                result["return_mean"][episode] = returns.mean()
    finally:
        for module, mode in zip(modules, modes):
            module.train(mode)
    return result


def evaluate_run(run_dir, n_episodes=100, seed=10_000, max_steps=10_000):
    """Evaluate all checkpoints, save per-episode arrays and summaries, and return CSV rows"""
    run_dir = Path(run_dir)
    paths = sorted((run_dir / "checkpoints").glob("step-*.pt"))
    if not paths:
        raise ValueError(f"No checkpoints found in {run_dir}")
    output_dir = run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        payload, cfg, actor, critics = load_checkpoint(path)
        wrapped_env = gym.make(cfg.env_name)
        try:
            # Use continuation rewards for the first benchmark-horizon states only
            # Training bootstraps through truncation, so stopping returns there would disagree
            result = monte_carlo_evaluation(
                wrapped_env.unwrapped,
                actor,
                critics,
                cfg.gamma,
                n_episodes=n_episodes,
                seed=seed,
                max_steps=max_steps,
                score_horizon=wrapped_env.spec.max_episode_steps,
            )
        finally:
            wrapped_env.close()
        np.savez(output_dir / f"step-{payload['step']:09d}.npz", **result)
        complete = result["completed"]
        for index, label in enumerate(result["critic_labels"]):

            def mean(name, result=result, complete=complete, index=index):
                values = result[name][complete, index]
                return float(values.mean()) if len(values) else float("nan")

            rows.append(
                {
                    "algorithm": payload["algorithm"],
                    "variant": cfg.tag,
                    "seed": cfg.seed,
                    "step": payload["step"],
                    "critic": str(label),
                    "bias": mean("bias"),
                    "mae": mean("mae"),
                    "q_mean": mean("q_mean"),
                    "return_mean": float(result["return_mean"][complete].mean())
                    if complete.any()
                    else float("nan"),
                    "reward": float(result["reward"].mean()),
                    "completed_episodes": int(complete.sum()),
                    "capped_episodes": int((~complete).sum()),
                    "completion_rate": float(complete.mean()),
                    "score_horizon": wrapped_env.spec.max_episode_steps,
                    "evaluation_seed": seed,
                    "evaluation_episodes": n_episodes,
                    "mc_max_steps": max_steps,
                }
            )
        # Completion rates identify when bias describes a selected subset of episodes
        print(
            f"{run_dir.name} step={payload['step']}: "
            f"{complete.sum()}/{n_episodes} complete MC trajectories"
        )
    with (output_dir / "bias.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
