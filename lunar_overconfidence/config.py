from dataclasses import dataclass


@dataclass(frozen=True)
class DDPGConfig:
    """Training settings with step counts in total transitions unless noted otherwise"""

    env_name: str = "LunarLanderContinuous-v3"
    seed: int = 1
    tag: str = "default"

    # max_steps and eval_interval count total transitions
    max_steps: int = 500_000
    n_envs: int = 1
    # Steps collected per environment before one update
    steps_per_update: int = 1
    learning_starts: int = 10_000

    buffer_size: int = 500_000
    batch_size: int = 256

    gamma: float = 0.99
    tau: float = 0.005
    action_noise: float = 0.1

    actor_hidden: tuple[int, ...] = (256, 256)
    critic_hidden: tuple[int, ...] = (256, 256)
    lr_actor: float = 1e-4
    lr_critic: float = 1e-4

    eval_interval: int = 25_000
    n_eval_episodes: int = 10

    # Critic hidden layers only
    layer_norm: bool = False
    actor_layer_norm: bool = False

    def __post_init__(self):
        for name in (
            "max_steps",
            "n_envs",
            "steps_per_update",
            "buffer_size",
            "batch_size",
            "eval_interval",
            "n_eval_episodes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.gamma < 1:
            raise ValueError("gamma must be in [0, 1)")
        if self.learning_starts < 0:
            raise ValueError("learning_starts must be nonnegative")
        if self.buffer_size < max(self.learning_starts, self.batch_size):
            raise ValueError("buffer_size must cover warmup and batch_size")
        if self.eval_interval % self.n_envs or self.max_steps % self.n_envs:
            raise ValueError("eval_interval and max_steps must be divisible by n_envs")


@dataclass(frozen=True)
class TD3Config(DDPGConfig):
    """DDPG settings plus TD3 policy delay and target smoothing"""

    policy_delay: int = 2
    target_noise: float = 0.2
    target_noise_clip: float = 0.5
