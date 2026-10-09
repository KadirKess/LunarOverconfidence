import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lunar_overconfidence.checkpoints import save_checkpoint
from lunar_overconfidence.config import TD3Config
from lunar_overconfidence.evaluation import (
    discounted_returns,
    monte_carlo_evaluation,
)
from lunar_overconfidence.networks import (
    ContinuousDeterministicActor,
    ContinuousQNetwork,
)
from tests.fixtures import ConstantCritic, TwoStepEnv


class EvaluationTests(unittest.TestCase):
    def test_matched_returns_and_td3_bias(self):
        np.testing.assert_allclose(discounted_returns([1, 2], 0.5), [2, 2])
        actor = ContinuousDeterministicActor(8, (4,), 2)
        result = monte_carlo_evaluation(
            TwoStepEnv(),
            actor,
            [ConstantCritic(5), ConstantCritic(1)],
            gamma=0.5,
            n_episodes=2,
            max_steps=3,
        )
        np.testing.assert_allclose(result["bias"], [[3, -1, -1], [3, -1, -1]])
        np.testing.assert_allclose(result["mae"], [[3, 1, 1], [3, 1, 1]])
        np.testing.assert_allclose(result["reward"], [3, 3])
        self.assertTrue(actor.training)

    def test_capped_trajectory_is_not_a_complete_return(self):
        actor = ContinuousDeterministicActor(8, (4,), 2)
        result = monte_carlo_evaluation(
            TwoStepEnv(),
            actor,
            [ConstantCritic(5)],
            0.5,
            n_episodes=1,
            max_steps=1,
        )
        self.assertFalse(result["completed"][0])
        self.assertTrue(np.isnan(result["bias"]).all())

    def test_horizon_restricts_states_but_keeps_continuation_rewards(self):
        actor = ContinuousDeterministicActor(8, (4,), 2)
        result = monte_carlo_evaluation(
            TwoStepEnv(),
            actor,
            [ConstantCritic(5)],
            gamma=0.25,
            n_episodes=1,
            max_steps=3,
            score_horizon=1,
        )
        np.testing.assert_allclose(result["return_mean"], [1.5])
        np.testing.assert_allclose(result["bias"], [[3.5]])
        np.testing.assert_allclose(result["mae"], [[3.5]])
        np.testing.assert_allclose(result["reward"], [1])
        np.testing.assert_equal(result["length"], [2])

    def test_evaluate_run_saves_metrics_without_plotting(self):
        from lunar_overconfidence.evaluation import evaluate_run

        cfg = TD3Config(actor_hidden=(4,), critic_hidden=(4,), layer_norm=True)
        actor = ContinuousDeterministicActor(8, (4,), 2)
        critics = [ContinuousQNetwork(8, (4,), 2, layer_norm=True) for _ in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (0, 25):
                save_checkpoint(root, "td3", cfg, step, actor, critics, 8, 2)
            with patch(
                "lunar_overconfidence.evaluation.gym.make",
                side_effect=lambda name: TwoStepEnv(),
            ):
                rows = evaluate_run(root, n_episodes=2, max_steps=3)
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(row["completed_episodes"] == 2 for row in rows))
            self.assertTrue(all(row["reward"] == 1 for row in rows))
            self.assertTrue(all(row["completion_rate"] == 1 for row in rows))
            self.assertTrue(all(row["score_horizon"] == 1 for row in rows))
            self.assertFalse((root / "plots").exists())
            self.assertTrue((root / "analysis" / "bias.csv").is_file())
            saved = np.load(root / "analysis" / "step-000000025.npz")
            np.testing.assert_equal(saved["episode_seed"], [10_000, 10_001])
