import csv
from pathlib import Path

import numpy as np
from scipy import stats


def _select_models(results, step, critic):
    if not results:
        raise ValueError("Need at least one evaluated training run")
    checkpoints = {}
    for key, rows in results.items():
        algorithm, variant, seed = key
        selected = {}
        for row in rows:
            if row["critic"] != critic:
                continue
            if (
                row["algorithm"].lower() != algorithm.lower()
                or int(row["seed"]) != seed
            ):
                raise ValueError(f"Evaluation metadata does not match run {key}")
            checkpoint = int(row["step"])
            if checkpoint in selected:
                raise ValueError(f"Duplicate {critic} checkpoint {checkpoint} in {key}")
            selected[checkpoint] = row
        if not selected:
            raise ValueError(f"No {critic} evaluations in run {key}")
        checkpoints[key] = selected
    common_steps = set.intersection(*(set(rows) for rows in checkpoints.values()))
    if not common_steps:
        raise ValueError("Runs have no shared checkpoint")
    step = max(common_steps) if step is None else step
    if step not in common_steps:
        raise ValueError(f"Checkpoint {step} is not available in every run")
    models = []
    for (algorithm, variant, seed), rows in sorted(checkpoints.items()):
        model = dict(
            rows[step],
            algorithm=algorithm.lower(),
            variant=variant,
            seed=seed,
            step=step,
        )
        models.append(model)
    for field in (
        "score_horizon",
        "evaluation_seed",
        "evaluation_episodes",
        "mc_max_steps",
    ):
        if len({row[field] for row in models}) != 1:
            raise ValueError(f"Runs use different evaluation settings for {field}")
    return step, models


def _finite(row):
    return bool(np.isfinite(row["bias"]) and np.isfinite(row["reward"]))


def _correlations(models, step, critic):
    associations = []
    for algorithm in sorted({row["algorithm"] for row in models}):
        rows = [row for row in models if row["algorithm"] == algorithm]
        groups = [("algorithm", "all", rows)]
        for variant in sorted({row["variant"] for row in rows}):
            groups.append(
                ("variant", variant, [row for row in rows if row["variant"] == variant])
            )
        for scope, variant, group in groups:
            valid = [row for row in group if _finite(row)]
            bias = np.array([row["bias"] for row in valid])
            reward = np.array([row["reward"] for row in valid])
            pearson = spearman = float("nan")
            if len(valid) >= 3 and np.ptp(bias) > 0 and np.ptp(reward) > 0:
                pearson = float(np.corrcoef(bias, reward)[0, 1])
                spearman = float(stats.spearmanr(bias, reward).statistic)
            associations.append(
                {
                    "algorithm": algorithm,
                    "scope": scope,
                    "variant": variant,
                    "step": step,
                    "critic": critic,
                    "n_models": len(valid),
                    "n_training_seeds": len({row["seed"] for row in valid}),
                    "excluded_models": len(group) - len(valid),
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )
    return associations


def _normalization_pairs(models, baseline, normalized, notes):
    lookup = {(row["algorithm"], row["variant"], row["seed"]): row for row in models}
    pairs = []
    for algorithm, seed in sorted({(row["algorithm"], row["seed"]) for row in models}):
        before = lookup.get((algorithm, baseline, seed))
        after = lookup.get((algorithm, normalized, seed))
        if before is None or after is None:
            notes.append(f"No matched normalization pair for {algorithm} seed {seed}.")
            continue
        if not _finite(before) or not _finite(after):
            notes.append(
                f"Excluded nonfinite normalization pair for {algorithm} seed {seed}."
            )
            continue
        pairs.append(
            {
                "algorithm": algorithm,
                "seed": seed,
                "step": before["step"],
                "critic": before["critic"],
                "baseline": baseline,
                "normalized": normalized,
                **{
                    f"delta_{field}": after[field] - before[field]
                    for field in ("bias", "mae", "q_mean", "return_mean", "reward")
                },
                "baseline_completion_rate": before["completion_rate"],
                "normalized_completion_rate": after["completion_rate"],
            }
        )
    return pairs


def analyze_bias_performance(
    results, step=None, critic="q1", baseline="baseline", normalized="layer-norm"
):
    """results maps (algorithm, variant, training seed) to evaluation rows"""
    if baseline == normalized:
        raise ValueError("Baseline and normalized variants must differ")
    step, models = _select_models(results, step, critic)
    notes = []
    if any(row["completion_rate"] < 1 for row in models):
        notes.append(
            "Some bias estimates exclude capped episodes, while reward includes all episodes."
        )
    associations = _correlations(models, step, critic)
    pairs = _normalization_pairs(models, baseline, normalized, notes)
    for row in associations:
        if not np.isfinite(row["pearson"]):
            notes.append(
                f"{row['algorithm']} / {row['variant']}: correlation unavailable with too few models or constant values."
            )
    return {
        "step": step,
        "critic": critic,
        "models": models,
        "associations": associations,
        "normalization_pairs": pairs,
        "notes": notes,
    }


def save_bias_performance(report, output_dir):
    """Save analysis tables and interpretation notes without generating plots"""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("models", "associations", "normalization_pairs"):
        rows = report[name]
        with (directory / f"{name}.csv").open("w", newline="") as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    (directory / "notes.txt").write_text(
        f"Training step: {report['step']}\nCritic: {report['critic']}\n\n"
        + "\n".join(report["notes"])
        + "\n"
    )
    return directory


def load_evaluations(run_dirs):
    """Read saved evaluation CSVs, preserving exact variant tags"""
    results = {}
    integer_fields = (
        "seed",
        "step",
        "completed_episodes",
        "capped_episodes",
        "evaluation_seed",
        "evaluation_episodes",
        "mc_max_steps",
    )
    float_fields = ("bias", "mae", "q_mean", "return_mean", "reward", "completion_rate")
    for directory in map(Path, run_dirs):
        with (directory / "analysis" / "bias.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"No evaluation rows in {directory}")
        for row in rows:
            for name in integer_fields:
                row[name] = int(row[name])
            for name in float_fields:
                row[name] = float(row[name])
            row["score_horizon"] = (
                int(row["score_horizon"])
                if row["score_horizon"] not in ("", "None")
                else None
            )
        keys = {(row["algorithm"], row["variant"], row["seed"]) for row in rows}
        if len(keys) != 1:
            raise ValueError(f"Mixed training runs in {directory}")
        key = keys.pop()
        if key in results:
            raise ValueError(f"Duplicate training run {key}")
        results[key] = rows
    return results


def compare_final_episodes(ddpg, td3):
    """Compare final episode rewards descriptively, without training-seed inference"""
    summaries = {}
    # Algorithm comparisons need independent training seeds, not more episodes per model
    for name, evaluator in (("DDPG", ddpg), ("TD3", td3)):
        rewards = evaluator.history[-1].rewards.numpy()
        summaries[name] = {
            "mean": float(rewards.mean()),
            "std": float(rewards.std(ddof=1)) if len(rewards) > 1 else float("nan"),
            "episodes": len(rewards),
        }
        summary = summaries[name]
        print(
            f"{name} final evaluation: mean={summary['mean']:.2f}, "
            f"episode std={summary['std']:.2f}, n={summary['episodes']}"
        )
    return summaries


def final_score(evaluator, last_k: int = 3) -> float:
    """Average the last few evaluation means for one training seed"""
    if not evaluator.history:
        raise ValueError("No evaluation results; train for at least one eval_interval")
    return float(np.mean([r.mean for r in evaluator.history[-last_k:]]))


def compare_seed_scores(
    results: dict, variants: dict, algorithms: dict, seeds: list[int]
):
    """Print Welch comparisons of independent training-seed scores by variant"""
    print(f"{'variant':<12} {'DDPG':>16} {'TD3':>16} {'p-value':>8}")
    # Use one score per trained model because its episodes and checkpoints are correlated
    for variant in variants:
        scores = {
            algo: np.array([final_score(results[(algo, variant, s)]) for s in seeds])
            for algo in algorithms
        }
        welch = stats.ttest_ind(scores["DDPG"], scores["TD3"], equal_var=False)

        def fmt(values):
            return f"{values.mean():7.1f} +/- {values.std(ddof=1):5.1f}"

        print(
            f"{variant:<12} {fmt(scores['DDPG']):>16} "
            f"{fmt(scores['TD3']):>16} {welch.pvalue:8.3f}"
        )
