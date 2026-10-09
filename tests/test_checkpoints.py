import tempfile
import unittest

import torch
from torch import nn

from lunar_overconfidence.checkpoints import load_checkpoint, save_checkpoint
from lunar_overconfidence.config import DDPGConfig, TD3Config
from lunar_overconfidence.networks import (
    ContinuousDeterministicActor,
    ContinuousQNetwork,
)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_roundtrip_and_normalization(self):
        cfg = DDPGConfig(layer_norm=True, actor_hidden=(4,), critic_hidden=(4,))
        actor = ContinuousDeterministicActor(8, cfg.actor_hidden, 2)
        critic = ContinuousQNetwork(8, cfg.critic_hidden, 2, layer_norm=True)
        self.assertFalse(any(isinstance(m, nn.LayerNorm) for m in actor.modules()))
        self.assertEqual(sum(isinstance(m, nn.LayerNorm) for m in critic.modules()), 1)
        obs = torch.randn(3, 8)
        action = actor.act(obs)
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(directory, "ddpg", cfg, 25, actor, [critic], 8, 2)
            payload, loaded_cfg, loaded_actor, critics = load_checkpoint(path)
            self.assertEqual(payload["step"], 25)
            self.assertEqual(loaded_cfg, cfg)
            torch.testing.assert_close(loaded_actor.act(obs), action)
            torch.testing.assert_close(critics[0](obs, action), critic(obs, action))

    def test_normalized_actor_roundtrip(self):
        for algorithm, Config in (("ddpg", DDPGConfig), ("td3", TD3Config)):
            with (
                self.subTest(algorithm=algorithm),
                tempfile.TemporaryDirectory() as directory,
            ):
                cfg = Config(actor_layer_norm=True, actor_hidden=(4, 4))
                actor = ContinuousDeterministicActor(
                    8, cfg.actor_hidden, 2, layer_norm=True
                )
                obs = torch.randn(3, 8)
                path = save_checkpoint(directory, algorithm, cfg, 0, actor, [], 8, 2)
                _, loaded_cfg, loaded_actor, _ = load_checkpoint(path)
                self.assertTrue(loaded_cfg.actor_layer_norm)
                self.assertEqual(
                    sum(isinstance(m, nn.LayerNorm) for m in loaded_actor.modules()), 2
                )
                torch.testing.assert_close(loaded_actor.act(obs), actor.act(obs))

    def test_legacy_checkpoint_without_actor_flag(self):
        cfg = DDPGConfig(actor_hidden=(4,))
        actor = ContinuousDeterministicActor(8, cfg.actor_hidden, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(directory, "ddpg", cfg, 0, actor, [], 8, 2)
            payload = torch.load(path, weights_only=True)
            del payload["config"]["actor_layer_norm"]
            torch.save(payload, path)
            _, loaded_cfg, loaded_actor, _ = load_checkpoint(path)
            self.assertFalse(loaded_cfg.actor_layer_norm)
            obs = torch.randn(3, 8)
            torch.testing.assert_close(loaded_actor.act(obs), actor.act(obs))
