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


def run_td3(cfg, run_dir, log_losses=True):
    """Train TD3, saving the actor and both critics at evaluation boundaries"""
    torch.manual_seed(cfg.seed)
    env = VecEnv(cfg.env_name, cfg.n_envs, seed=cfg.seed, same_step_reset=True)

    actor = ContinuousDeterministicActor(
        env.observation_dim,
        cfg.actor_hidden,
        env.action_dim,
        layer_norm=cfg.actor_layer_norm,
    )
    critic_1 = ContinuousQNetwork(
        env.observation_dim,
        cfg.critic_hidden,
        env.action_dim,
        layer_norm=cfg.layer_norm,
    )
    critic_2 = ContinuousQNetwork(
        env.observation_dim,
        cfg.critic_hidden,
        env.action_dim,
        layer_norm=cfg.layer_norm,
    )

    target_critic_1 = copy.deepcopy(critic_1)
    target_critic_2 = copy.deepcopy(critic_2)
    target_actor = copy.deepcopy(actor)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    critic_optimizer_1 = torch.optim.Adam(critic_1.parameters(), lr=cfg.lr_critic)
    critic_optimizer_2 = torch.optim.Adam(critic_2.parameters(), lr=cfg.lr_critic)

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

    updates = 0

    def save(step):
        return save_checkpoint(
            run_dir,
            "td3",
            cfg,
            step,
            actor,
            [critic_1, critic_2],
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

                # Clipped double-Q targets reduce the effect of optimistic critic errors
                with torch.no_grad():
                    next_noisy_action = target_actor(batch.next_obs).value
                    noise = (
                        torch.randn_like(next_noisy_action) * cfg.target_noise
                    ).clamp(-cfg.target_noise_clip, cfg.target_noise_clip)
                    next_noisy_action = (next_noisy_action + noise).clamp(-1, 1)
                    next_q = torch.min(
                        target_critic_1(batch.next_obs, next_noisy_action),
                        target_critic_2(batch.next_obs, next_noisy_action),
                    )
                    target = batch.reward + cfg.gamma * next_q * (
                        1 - batch.terminated.float()
                    )

                critic_loss_1 = F.mse_loss(
                    target, critic_1(batch.obs, batch.action.value)
                )
                critic_loss_2 = F.mse_loss(
                    target, critic_2(batch.obs, batch.action.value)
                )

                critic_optimizer_1.zero_grad()
                critic_optimizer_2.zero_grad()
                critic_loss_1.backward()
                critic_loss_2.backward()
                critic_optimizer_1.step()
                critic_optimizer_2.step()

                # Delay policy and target updates until the critics have had more updates
                updates += 1
                if updates % cfg.policy_delay == 0:
                    # Q1 drives the actor while the minimum is only used in critic targets
                    actor_loss = -critic_1(batch.obs, actor(batch.obs).value).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    soft_update(critic_1, target_critic_1, cfg.tau)
                    soft_update(critic_2, target_critic_2, cfg.tau)
                    soft_update(actor, target_actor, cfg.tau)
                    if writer is not None:
                        writer.add_scalar(
                            "loss/actor", actor_loss.item(), collector.steps
                        )

                if writer is not None:
                    writer.add_scalar(
                        "loss/critic_1", critic_loss_1.item(), collector.steps
                    )
                    writer.add_scalar(
                        "loss/critic_2", critic_loss_2.item(), collector.steps
                    )

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
