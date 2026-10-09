import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import torch

from lunar_overconfidence.checkpoints import load_checkpoint
from lunar_overconfidence.experiments import (
    create_experiment,
    load_experiment,
    plot_experiment,
    train_experiment,
)


class ExperimentTests(unittest.TestCase):
    def test_notebook_configures_thirty_zoo_runs(self):
        notebook = json.loads(Path("experiments.ipynb").read_text())
        settings = next(
            cell
            for cell in notebook["cells"]
            if "settings" in cell["metadata"].get("tags", [])
        )
        namespace = {"torch": torch}
        threads = torch.get_num_threads()
        try:
            exec("".join(settings["source"]), namespace)  # noqa: S102
            self.assertEqual(namespace["seeds"], [1, 2, 3, 4, 5])
            with tempfile.TemporaryDirectory() as directory:
                root = create_experiment(
                    namespace["experiment_name"],
                    namespace["training"],
                    namespace["seeds"],
                    namespace["evaluation"],
                    namespace["td3_settings"],
                    output_root=directory,
                    normalization_variants=namespace["normalization_variants"],
                )
                manifest = json.loads((root / "experiment.json").read_text())
                self.assertEqual(len(manifest["runs"]), 30)
                self.assertEqual(manifest["torch_threads"], 1)
                self.assertEqual(len({run["path"] for run in manifest["runs"]}), 30)
                expected = {
                    "baseline": (False, False),
                    "layer-norm": (True, False),
                    "actor-critic-layer-norm": (True, True),
                }
                for run in manifest["runs"]:
                    cfg = run["config"]
                    self.assertEqual(
                        (cfg["layer_norm"], cfg["actor_layer_norm"]),
                        expected[run["variant"]],
                    )
                    for field, value in {
                        "lr_actor": 1e-3,
                        "lr_critic": 1e-3,
                        "gamma": 0.98,
                        "buffer_size": 200_000,
                        "batch_size": 256,
                        "learning_starts": 10_000,
                        "max_steps": 300_000,
                        "actor_hidden": [256, 256],
                        "critic_hidden": [256, 256],
                    }.items():
                        self.assertEqual(cfg[field], value)
        finally:
            torch.set_num_threads(threads)

    def test_name_config_and_overwrite_protection(self):
        evaluation = {"n_episodes": 2, "seed": 10000, "max_steps": 1000}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("lunar_overconfidence.experiments.datetime") as clock,
        ):
            clock.now.return_value = datetime(
                2026, 10, 7, 15, 30, 45, tzinfo=datetime.now().astimezone().tzinfo
            )
            root = create_experiment(
                "my experiment", {}, [1], evaluation, output_root=directory
            )
            self.assertEqual(root.name, "20261007-153045-my-experiment")
            manifest = json.loads((root / "experiment.json").read_text())
            self.assertEqual(len(manifest["runs"]), 4)
            self.assertEqual(manifest["evaluation"], evaluation)
            for run in manifest["runs"]:
                self.assertEqual(
                    run["config"]["layer_norm"], run["variant"] == "layer-norm"
                )
                self.assertFalse(run["config"]["actor_layer_norm"])
            with self.assertRaises(FileExistsError):
                create_experiment(
                    "my experiment", {}, [1], evaluation, output_root=directory
                )
            (root / "ddpg/baseline/seed-1/checkpoints").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "already has checkpoints"):
                train_experiment(root)

    def test_notebook_stages_and_reopening(self):
        notebook = json.loads(Path("experiments.ipynb").read_text())
        namespace = {}
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with (
                tempfile.TemporaryDirectory() as directory,
                patch("IPython.display.display"),
            ):
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] != "code":
                        continue
                    tag = cell["metadata"]["tags"][0]
                    source = "".join(cell["source"])
                    exec(  # noqa: S102
                        compile(source, f"experiments.ipynb:cell-{index}", "exec"),
                        namespace,
                    )
                    if tag == "imports":
                        namespace["create_experiment"] = lambda *args, **kwargs: (
                            create_experiment(*args, **kwargs, output_root=directory)
                        )
                    if tag == "settings":
                        namespace.update(
                            experiment_name="smoke", seeds=[1], log_losses=False
                        )
                        namespace["training"].update(
                            max_steps=8,
                            eval_interval=4,
                            n_envs=2,
                            learning_starts=4,
                            buffer_size=20,
                            batch_size=2,
                            n_eval_episodes=1,
                            actor_hidden=(4,),
                            critic_hidden=(4,),
                        )
                        namespace["evaluation"].update(n_episodes=2, max_steps=1000)
                root = namespace["experiment_dir"]
                manifest = json.loads((root / "experiment.json").read_text())
                for run in manifest["runs"]:
                    self.assertEqual(
                        run["config"]["layer_norm"], run["variant"] != "baseline"
                    )
                    self.assertEqual(
                        run["config"]["actor_layer_norm"],
                        run["variant"] == "actor-critic-layer-norm",
                    )
                    _, _, actor, critics = load_checkpoint(
                        root / run["path"] / "checkpoints/step-000000008.pt"
                    )
                    self.assertEqual(
                        sum(isinstance(m, torch.nn.LayerNorm) for m in actor.modules()),
                        int(run["config"]["actor_layer_norm"]),
                    )
                    for critic in critics:
                        self.assertEqual(
                            sum(
                                isinstance(m, torch.nn.LayerNorm)
                                for m in critic.modules()
                            ),
                            int(run["config"]["layer_norm"]),
                        )
                self.assertEqual(manifest["torch_threads"], 1)
                self.assertEqual(len(namespace["models"]), 6)
                self.assertEqual(len(namespace["reports"]), 3)
                for report in namespace["reports"].values():
                    self.assertEqual(len(report["normalization_pairs"]), 2)
                self.assertEqual(len(list(root.rglob("*.pdf"))), 15)
                self.assertEqual(len(list(root.rglob("step-*.pt"))), 18)
                self.assertFalse(list(root.rglob("events.*")))
                loaded = load_experiment(root)
                from lunar_overconfidence.analysis import analyze_bias_performance

                for name, (baseline, normalized) in namespace["comparisons"].items():
                    original = namespace["reports"][name]
                    reopened = analyze_bias_performance(
                        loaded, baseline=baseline, normalized=normalized
                    )
                    for section in ("models", "associations", "normalization_pairs"):
                        self.assertEqual(
                            json.dumps(reopened[section], sort_keys=True),
                            json.dumps(original[section], sort_keys=True),
                        )
                with (
                    patch("lunar_overconfidence.experiments.run_ddpg") as train,
                    patch("lunar_overconfidence.experiments.evaluate_run") as evaluate,
                ):
                    plot_experiment(loaded, reopened, root)
                    train.assert_not_called()
                    evaluate.assert_not_called()
        finally:
            torch.set_num_threads(threads)
