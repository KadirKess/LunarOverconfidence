import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .style import COLORS, STYLE, save_figure, setup_axis, variant_legend


def draw_seed_curves(
    ax, runs, metric, color, interval="seeds", critic="q1", linestyle="-"
):
    """Draw recorded checkpoints without smoothing or filling missing data"""
    steps = sorted(
        {int(row["step"]) for rows in runs for row in rows if row["critic"] == critic}
    )
    curves = []
    for rows in runs:
        values = {
            int(row["step"]): float(row[metric])
            for row in rows
            if row["critic"] == critic
        }
        curves.append([values.get(step, np.nan) for step in steps])
    curves = np.asarray(curves, dtype=float)
    if not len(curves) or not len(steps):
        return
    if interval not in ("seeds", "ci95"):
        raise ValueError("interval must be seeds or ci95")
    if interval == "seeds":
        for curve in curves:
            ax.plot(
                steps, curve, color=color, alpha=0.28, linewidth=1, linestyle=linestyle
            )
    means, widths = [], []
    for column in curves.T:
        finite = column[np.isfinite(column)]
        means.append(finite.mean() if len(finite) else np.nan)
        widths.append(
            stats.sem(finite) * stats.t.ppf(0.975, len(finite) - 1)
            if len(finite) > 1
            else np.nan
        )
    means = np.asarray(means)
    if interval == "ci95":
        widths = np.asarray(widths)
        ax.fill_between(steps, means - widths, means + widths, color=color, alpha=0.12)
    ax.plot(
        steps,
        means,
        color=color,
        linewidth=2.3,
        marker="o",
        markersize=4,
        linestyle=linestyle,
    )


def plot_learning_curves(
    results, output_dir, baseline="baseline", normalized="layer-norm", interval="seeds"
):
    algorithms = sorted({algo.lower() for algo, _, _ in results})
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            2,
            len(algorithms),
            figsize=(5.1 * len(algorithms), 6.6),
            squeeze=False,
            sharex="col",
            sharey="row",
        )
        for column, algorithm in enumerate(algorithms):
            axes[0, column].set_title(algorithm.upper(), pad=10)
            counts = []
            for variant, color in (
                (baseline, COLORS["baseline"]),
                (normalized, COLORS["layer-norm"]),
            ):
                runs = [
                    rows
                    for (algo, tag, _), rows in results.items()
                    if algo.lower() == algorithm and tag == variant
                ]
                counts.append(len(runs))
                for index, metric in enumerate(("reward", "bias")):
                    draw_seed_curves(axes[index, column], runs, metric, color, interval)
            axes[0, column].text(
                0.02,
                0.04,
                f"Training seeds per displayed variant: {counts[0]} / {counts[1]}",
                transform=axes[0, column].transAxes,
                fontsize=8,
                color="#53616D",
            )
            setup_axis(axes[0, column], zero=True)
            setup_axis(axes[1, column], steps=True, zero=True)
        axes[0, 0].set_ylabel("Mean episode reward")
        axes[1, 0].set_ylabel("Signed Q1 bias\nQ prediction - discounted return")
        fig.suptitle("Learning across normalization conditions", y=0.995, fontsize=15)
        variant_legend(fig, baseline, normalized, bbox_to_anchor=(0.5, 0.955))
        detail = (
            "Thin lines: individual training seeds. Thick lines: seed means."
            if interval == "seeds"
            else "Lines: seed means. Bands: 95% Student-t confidence intervals across training seeds."
        )
        fig.text(
            0.5,
            0.015,
            detail + " No smoothing. Bias uses complete trajectories.",
            ha="center",
            fontsize=8,
        )
        fig.subplots_adjust(top=0.83, bottom=0.13, hspace=0.25, wspace=0.12)
        return save_figure(fig, output_dir, "learning-curves")


def plot_evaluations(
    output_dir="outputs/plots", filename="evaluations.png", **evaluators
):
    """Preview reward histories from training, with episode standard deviations"""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for index, (name, evaluator) in enumerate(evaluators.items()):
            steps = [result.step for result in evaluator.history]
            means = np.array([result.mean for result in evaluator.history])
            stds = np.array(
                [float(result.rewards.std()) for result in evaluator.history]
            )
            color = list(COLORS.values())[index % 2]
            ax.plot(
                steps,
                means,
                color=color,
                label=name,
                linewidth=2,
                marker="o",
                markersize=3,
            )
            ax.fill_between(steps, means - stds, means + stds, color=color, alpha=0.12)
        setup_axis(ax, steps=True)
        ax.set_ylabel("Mean episode reward")
        ax.legend(frameon=False)
        fig.text(
            0.5,
            0.01,
            "Band: episode standard deviation for this trained model.",
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        return save_figure(fig, output_dir, filename.rsplit(".", 1)[0])["png"]
