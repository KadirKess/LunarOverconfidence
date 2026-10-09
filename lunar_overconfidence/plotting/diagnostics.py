import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .learning_curves import draw_seed_curves
from .style import COLORS, STYLE, save_figure, setup_axis, variant_label, variant_legend


def plot_td3_critics(results, output_dir, baseline="baseline", normalized="layer-norm"):
    td3 = {key: rows for key, rows in results.items() if key[0].lower() == "td3"}
    if not td3:
        return None
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), sharey=True)
        styles = {"q1": "-", "q2": "--", "min": ":"}
        for ax, variant, color, title in zip(
            axes,
            (baseline, normalized),
            COLORS.values(),
            (variant_label(baseline), variant_label(normalized)),
        ):
            runs = [rows for (_, tag, _), rows in td3.items() if tag == variant]
            for critic, linestyle in styles.items():
                draw_seed_curves(
                    ax, runs, "bias", color, critic=critic, linestyle=linestyle
                )
            setup_axis(ax, steps=True, zero=True)
            ax.set_title(title, pad=10)
        axes[0].set_ylabel("Signed critic bias")
        fig.suptitle("TD3 critic diagnostics", y=0.995, fontsize=15)
        fig.legend(
            handles=[
                Line2D([0], [0], color="#53616D", linestyle=style, label=label.upper())
                for label, style in styles.items()
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=3,
            frameon=False,
        )
        fig.text(
            0.5,
            0.025,
            "Thin lines: individual seeds. Thick lines: seed means. Critics can overlap closely.",
            ha="center",
            fontsize=8,
        )
        fig.subplots_adjust(top=0.78, bottom=0.18, wspace=0.12)
        return save_figure(fig, output_dir, "td3-critics")


def plot_completion_rates(
    results, output_dir, baseline="baseline", normalized="layer-norm"
):
    algorithms = sorted({key[0].lower() for key in results})
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            1,
            len(algorithms),
            figsize=(5.1 * len(algorithms), 4),
            squeeze=False,
            sharey=True,
        )
        for ax, algorithm in zip(axes[0], algorithms):
            for variant, color in (
                (baseline, COLORS["baseline"]),
                (normalized, COLORS["layer-norm"]),
            ):
                runs = [
                    rows
                    for (algo, tag, _), rows in results.items()
                    if algo.lower() == algorithm and tag == variant
                ]
                draw_seed_curves(ax, runs, "completion_rate", color)
            setup_axis(ax, steps=True)
            ax.set_ylim(-0.02, 1.05)
            ax.set_title(algorithm.upper())
        axes[0, 0].set_ylabel("Fraction of complete MC trajectories")
        fig.suptitle("Monte Carlo completion", y=0.995, fontsize=15)
        variant_legend(fig, baseline, normalized, bbox_to_anchor=(0.5, 0.95))
        fig.text(
            0.5,
            0.025,
            "Below 1: bias describes only completed episodes. Colors match the main learning curves.",
            ha="center",
            fontsize=8,
        )
        fig.subplots_adjust(top=0.74, bottom=0.22, wspace=0.12)
        return save_figure(fig, output_dir, "completion-rates")
