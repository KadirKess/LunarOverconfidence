import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, STYLE, save_figure, setup_axis, variant_label, variant_legend


def plot_bias_performance(
    report, output_dir, baseline="baseline", normalized="layer-norm"
):
    algorithms = sorted({row["algorithm"] for row in report["models"]})
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            1,
            len(algorithms),
            figsize=(5.1 * len(algorithms), 4.8),
            squeeze=False,
            sharey=False,
        )
        for ax, algorithm in zip(axes[0], algorithms):
            rows = [row for row in report["models"] if row["algorithm"] == algorithm]
            excluded = 0
            for row in rows:
                if row["variant"] not in (baseline, normalized):
                    continue
                if not np.isfinite(row["bias"]) or not np.isfinite(row["reward"]):
                    excluded += 1
                    continue
                color = COLORS[
                    "baseline" if row["variant"] == baseline else "layer-norm"
                ]
                ax.scatter(
                    row["bias"],
                    row["reward"],
                    s=65,
                    color=color,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=3,
                )
                ax.annotate(
                    f"S{row['seed']}",
                    (row["bias"], row["reward"]),
                    xytext=(6, 7 if row["seed"] % 2 else -13),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                )
            setup_axis(ax, zero=True)
            ax.axvline(0, color="#7D8892", linewidth=0.9, linestyle="--")
            ax.set_title(algorithm.upper(), pad=10)
            ax.set_xlabel(f"Signed {report['critic'].upper()} bias")
            ax.set_ylabel("Mean episode reward")
            ax.margins(x=0.18, y=0.18)
            if excluded:
                ax.text(
                    0.02,
                    0.02,
                    f"{excluded} nonfinite models excluded",
                    transform=ax.transAxes,
                    fontsize=8,
                )
        axes[0, 0].set_ylabel("Mean episode reward")
        fig.suptitle(
            f"Bias and performance at {report['step'] / 1000:g}k training steps",
            y=0.995,
            fontsize=15,
        )
        variant_legend(fig, baseline, normalized, bbox_to_anchor=(0.5, 0.95))
        fig.text(
            0.5,
            0.025,
            "One point per model. S = training seed. Panels use separate scales. Association does not establish causation.",
            ha="center",
            fontsize=8,
        )
        fig.subplots_adjust(top=0.78, bottom=0.18, wspace=0.12)
        return save_figure(fig, output_dir, "bias-versus-performance")


def plot_normalization_effects(report, output_dir):
    pairs = report["normalization_pairs"]
    height = max(4.4, 2.2 + len(pairs) * 0.32)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10.2, height), sharey=True)
        positions = np.arange(len(pairs))
        labels = [f"{row['algorithm'].upper()}  S{row['seed']}" for row in pairs]
        for ax, metric, title, direction in zip(
            axes,
            ("delta_bias", "delta_reward"),
            ("Change in signed bias", "Change in episode reward"),
            ("Negative = lower signed bias", "Positive = higher reward"),
        ):
            values = [row[metric] for row in pairs]
            ax.hlines(positions, 0, values, color="#B7C0C8", linewidth=1.5)
            ax.scatter(values, positions, color=COLORS["layer-norm"], s=60, zorder=3)
            ax.axvline(0, color="#7D8892", linewidth=0.9, linestyle="--")
            ax.set_title(title, pad=12)
            ax.set_xlabel("Second condition - first condition")
            ax.set_yticks(positions, labels)
            setup_axis(ax)
            ax.margins(x=0.2)
            ax.text(
                0.5, -0.23, direction, transform=ax.transAxes, ha="center", fontsize=8
            )
            if not pairs:
                ax.text(
                    0.5,
                    0.5,
                    "No complete matched seed pairs",
                    ha="center",
                    transform=ax.transAxes,
                )
        axes[0].invert_yaxis()
        fig.suptitle(
            f"Paired normalization effects at {report['step'] / 1000:g}k training steps",
            y=0.995,
            fontsize=15,
        )
        if pairs:
            fig.text(
                0.5,
                0.9,
                f"{variant_label(pairs[0]['normalized'])} minus {variant_label(pairs[0]['baseline'])}",
                ha="center",
                fontsize=9,
            )
        fig.text(
            0.5,
            0.015,
            "Each row compares the same algorithm and training seed. Values are differences, not confidence intervals.",
            ha="center",
            fontsize=8,
        )
        fig.subplots_adjust(top=0.8, bottom=0.24, left=0.13, wspace=0.14)
        return save_figure(fig, output_dir, "normalization-effects")
