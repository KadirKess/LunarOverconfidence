import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from lunar_overconfidence.checkpoints import load_checkpoint
from lunar_overconfidence.config import DDPGConfig, TD3Config
from lunar_overconfidence.networks import (
    ContinuousDeterministicActor,
)


class TrainingTests(unittest.TestCase):
    def test_training_saves_warmup_intervals_and_final(self):
        from lunar_overconfidence import ddpg, td3

        for module, Config in ((ddpg, DDPGConfig), (td3, TD3Config)):
            with (
                self.subTest(algorithm=module.__name__),
                tempfile.TemporaryDirectory() as directory,
            ):
                cfg = replace(
                    Config(),
                    max_steps=6,
                    eval_interval=2,
                    n_envs=2,
                    learning_starts=4,
                    buffer_size=10,
                    batch_size=2,
                    n_eval_episodes=1,
                    actor_hidden=(4,),
                    critic_hidden=(4,),
                    layer_norm=True,
                    actor_layer_norm=True,
                )
                updates = []
                original = module.soft_update

                def tracked(source, target, tau, updates=updates, original=original):
                    updates.append((source, target))
                    original(source, target, tau)

                with (
                    patch.object(module, "soft_update", side_effect=tracked),
                ):
                    evaluator = (
                        module.run_ddpg(cfg, directory)
                        if module is ddpg
                        else module.run_td3(cfg, directory)
                    )
                try:
                    files = sorted((Path(directory) / "checkpoints").glob("*.pt"))
                    self.assertEqual(
                        [load_checkpoint(p)[0]["step"] for p in files], [0, 2, 4, 6]
                    )
                    self.assertEqual([r.step for r in evaluator.history], [2, 4, 6])
                    self.assertTrue(
                        any(
                            isinstance(a, ContinuousDeterministicActor)
                            for a, b in updates
                        )
                    )
                    for source, target in updates:
                        self.assertIsNot(source, target)
                    self.assertEqual(
                        len(load_checkpoint(files[-1])[3]), 1 if module is ddpg else 2
                    )
                finally:
                    evaluator.env.gym_env.close()

    def test_final_checkpoint_between_evaluations(self):
        from lunar_overconfidence import ddpg

        cfg = DDPGConfig(
            max_steps=6,
            eval_interval=4,
            learning_starts=10,
            buffer_size=10,
            batch_size=2,
            n_eval_episodes=1,
            actor_hidden=(4,),
            critic_hidden=(4,),
        )
        with tempfile.TemporaryDirectory() as directory:
            evaluator = ddpg.run_ddpg(cfg, directory, log_losses=False)
            self.assertFalse(list(Path(directory).glob("events.*")))
            evaluator.env.gym_env.close()
            files = sorted((Path(directory) / "checkpoints").glob("*.pt"))
            self.assertEqual([load_checkpoint(p)[0]["step"] for p in files], [0, 4, 6])
