import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from .analysis import load_evaluations
from .config import DDPGConfig, TD3Config
from .ddpg import run_ddpg
from .evaluation import evaluate_run
from .plotting.bias_performance import plot_bias_performance, plot_normalization_effects
from .plotting.diagnostics import plot_completion_rates, plot_td3_critics
from .plotting.learning_curves import plot_learning_curves
from .td3 import run_td3


def create_experiment(
    name,
    training,
    seeds,
    evaluation,
    td3_settings=None,
    output_root="outputs",
    *,
    normalization=None,
    normalization_variants=None,
):
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip()).strip("-")
    if not name or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Choose an experiment name and distinct training seeds")
    runs = []
    if normalization is not None and normalization_variants is not None:
        raise ValueError("Use normalization or normalization_variants, not both")
    if normalization_variants is None:
        normalization = normalization or {"layer_norm": True, "actor_layer_norm": False}
        normalization_variants = {
            "baseline": {"layer_norm": False, "actor_layer_norm": False},
            "layer-norm": normalization,
        }
    if not normalization_variants:
        raise ValueError("Choose at least one normalization variant")
    for variant, flags in normalization_variants.items():
        if (
            re.fullmatch(r"[A-Za-z0-9_-]+", variant) is None
            or set(flags) != {"layer_norm", "actor_layer_norm"}
            or any(type(value) is not bool for value in flags.values())
        ):
            raise ValueError(
                "Variants need a simple name and two boolean normalization flags"
            )
    for algorithm, Config in (("ddpg", DDPGConfig), ("td3", TD3Config)):
        for variant, flags in normalization_variants.items():
            for seed in seeds:
                settings = (
                    {**training, **(td3_settings or {})}
                    if algorithm == "td3"
                    else training
                )
                cfg = Config(
                    **{
                        **settings,
                        "seed": seed,
                        "tag": variant,
                        **flags,
                    }
                )
                runs.append(
                    {
                        "algorithm": algorithm,
                        "variant": variant,
                        "seed": seed,
                        "path": f"{algorithm}/{variant}/seed-{seed}",
                        "config": asdict(cfg),
                    }
                )
    if evaluation["n_episodes"] <= 0 or evaluation["max_steps"] < 1000:
        raise ValueError(
            "Use positive evaluation episodes and an MC cap of at least 1000"
        )
    created = datetime.now().astimezone()
    directory = Path(output_root) / f"{created:%Y%m%d-%H%M%S}-{name}"
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "name": name,
        "created_at": created.isoformat(timespec="seconds"),
        "runs": runs,
        "evaluation": evaluation,
        "torch_threads": torch.get_num_threads(),
    }
    (directory / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory


def train_experiment(experiment_dir, log_losses=True):
    experiment_dir = Path(experiment_dir)
    manifest = json.loads((experiment_dir / "experiment.json").read_text())
    if any(
        (experiment_dir / run["path"] / "checkpoints").exists()
        for run in manifest["runs"]
    ):
        raise ValueError(
            "This experiment already has checkpoints; create a new experiment to train again"
        )
    models = {}
    if "torch_threads" in manifest:
        torch.set_num_threads(manifest["torch_threads"])
    for run in manifest["runs"]:
        algorithm = run["algorithm"]
        Config = DDPGConfig if algorithm == "ddpg" else TD3Config
        train = run_ddpg if algorithm == "ddpg" else run_td3
        print(f"{algorithm.upper()} / {run['variant']} / seed {run['seed']}")
        key = (algorithm, run["variant"], run["seed"])
        models[key] = train(
            Config(**run["config"]), experiment_dir / run["path"], log_losses
        )
    return models


def evaluate_experiment(experiment_dir):
    experiment_dir = Path(experiment_dir)
    manifest = json.loads((experiment_dir / "experiment.json").read_text())
    results = {}
    for run in manifest["runs"]:
        key = (run["algorithm"], run["variant"], run["seed"])
        results[key] = evaluate_run(
            experiment_dir / run["path"], **manifest["evaluation"]
        )
    return results


def load_experiment(experiment_dir):
    experiment_dir = Path(experiment_dir)
    manifest = json.loads((experiment_dir / "experiment.json").read_text())
    return load_evaluations([experiment_dir / run["path"] for run in manifest["runs"]])


def plot_experiment(
    results,
    report,
    experiment_dir,
    baseline="baseline",
    normalized="layer-norm",
    interval="seeds",
):
    output_dir = Path(experiment_dir) / "figures"
    return {
        "learning_curves": plot_learning_curves(
            results, output_dir, baseline, normalized, interval
        ),
        "bias_performance": plot_bias_performance(
            report, output_dir, baseline, normalized
        ),
        "normalization_effects": plot_normalization_effects(report, output_dir),
    }


def plot_diagnostics(
    results, experiment_dir, baseline="baseline", normalized="layer-norm"
):
    output_dir = Path(experiment_dir) / "figures"
    return {
        "td3_critics": plot_td3_critics(results, output_dir, baseline, normalized),
        "completion_rates": plot_completion_rates(
            results, output_dir, baseline, normalized
        ),
    }
