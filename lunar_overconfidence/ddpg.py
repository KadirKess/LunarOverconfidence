import copy
from pathlib import Path

import rl_mind.envs  # noqa: F401
import torch
import torch.nn.functional as F
from rl_mind.collectors import TransitionCollector
from rl_mind.data import ReplayBuffer
from rl_mind.env import VecEnv
from rl_mind.evaluation import Evaluator
from rl_mind.nn import soft_update
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from lunar_overconfidence.checkpoints import save_checkpoint
from lunar_overconfidence.networks import (
    ContinuousDeterministicActor,
    ContinuousQNetwork,
    GaussianNoise,
)


def run_ddpg(cfg, run_dir, log_losses=True):
    """Train DDPG, saving actor-critic pairs at evaluation boundaries"""
    torch.manual_seed(cfg.seed)
    env = VecEnv(cfg.env_name, cfg.n_envs, seed=cfg.seed, same_step_reset=True)

    actor = ContinuousDeterministicActor(
        env.observation_dim,
        cfg.actor_hidden,
        env.action_dim,
        layer_norm=cfg.actor_layer_norm,
    )
    critic = ContinuousQNetwork(
        env.observation_dim,
        cfg.critic_hidden,
        env.action_dim,
        layer_norm=cfg.layer_norm,
    )
    target_critic = copy.deepcopy(critic)
    target_actor = copy.deepcopy(actor)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    collector = TransitionCollector(env, GaussianNoise(actor, cfg.action_noise))
    buffer = ReplayBuffer(cfg.buffer_size)
    run_dir = Path(run_dir)
    writer = SummaryWriter(run_dir) if log_losses else None
    evaluator = Evaluator(
        VecEnv(cfg.env_name, cfg.n_eval_episodes, seed=cfg.seed + 100),
        every=cfg.eval_interval,
        run_dir=run_dir,
        writer=writer,
    )

    def save(step):
        return save_checkpoint(
            run_dir,
            "ddpg",
            cfg,
            step,
            actor,
            [critic],
            env.observation_dim,
            env.action_dim,
        )

    save(0)
    pbar = tqdm(total=cfg.max_steps)
    try:
        while collector.steps < cfg.max_steps:
            # Stop collection at evaluation boundaries, including during warmup
            until_eval = cfg.eval_interval - collector.steps % cfg.eval_interval
            remaining = min(until_eval, cfg.max_steps - collector.steps)
            collect_steps = min(cfg.steps_per_update, remaining // cfg.n_envs)
            buffer.add(collector.collect(collect_steps))
            pbar.update(collector.steps - pbar.n)
            if len(buffer) >= max(cfg.learning_starts, cfg.batch_size):
                batch = buffer.sample(cfg.batch_size)

                q_values = critic(batch.obs, batch.action.value)
                # Use slowly moving targets so critic updates do not chase the live actor
                with torch.no_grad():
                    next_q_values = target_critic(
                        batch.next_obs, target_actor(batch.next_obs).value
                    )
                critic_loss = compute_critic_loss(
                    cfg.gamma, batch, q_values, next_q_values
                )

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                actor_loss = -critic(batch.obs, actor(batch.obs).value).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Move both targets gradually toward the trained networks
                soft_update(critic, target_critic, cfg.tau)
                soft_update(actor, target_actor, cfg.tau)

                if writer is not None:
                    writer.add_scalar(
                        "loss/critic", critic_loss.item(), collector.steps
                    )
                    writer.add_scalar("loss/actor", actor_loss.item(), collector.steps)
            if result := evaluator.run_if_needed(collector.steps, actor):
                save(collector.steps)
                pbar.set_description(
                    f"eval={result.mean:7.1f} best={evaluator.best_reward:7.1f}"
                )

        save(collector.steps)
    finally:
        pbar.close()
        if writer is not None:
            writer.close()
        env.gym_env.close()
        evaluator.env.gym_env.close()
    return evaluator


def compute_critic_loss(gamma, batch, q_values, next_q_values):
    """Fit Q(s, a) to a detached one-step target with inputs shaped [batch]"""
    # Bootstrap through time limits and stop only on true termination
    d = batch.terminated.float()
    target = (batch.reward + gamma * next_q_values * (1 - d)).detach()
    critic_loss = F.mse_loss(target, q_values)
    return critic_loss
