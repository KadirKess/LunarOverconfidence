from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

COLORS = {"baseline": "#25658A", "layer-norm": "#D47732"}
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#A5ABB1",
    "text.color": "#26313B",
    "axes.labelcolor": "#26313B",
    "xtick.color": "#53616D",
    "ytick.color": "#53616D",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
}


def setup_axis(ax, steps=False, zero=False):
    ax.grid(axis="y", color="#E2E6E9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)
    if zero:
        ax.axhline(0, color="#7D8892", linewidth=0.9, linestyle="--", zorder=1)
    if steps:
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value / 1000:g}k" if value else "0")
        )
        ax.set_xlabel("Training environment steps")


def variant_label(variant):
    return {
        "baseline": "No normalization",
        "layer-norm": "Critic normalization",
        "actor-critic-layer-norm": "Actor + critic normalization",
    }.get(variant, variant)


def variant_legend(fig, baseline="baseline", normalized="layer-norm", **kwargs):
    handles = [
        Line2D([0], [0], color=COLORS[name], linewidth=2.4, label=label)
        for name, label in (
            ("baseline", variant_label(baseline)),
            ("layer-norm", variant_label(normalized)),
        )
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, **kwargs)


def save_figure(fig, output_dir, name):
    """Export the same figure as a report PDF and a preview PNG"""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    try:
        for suffix in ("png", "pdf"):
            path = directory / f"{name}.{suffix}"
            fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.18)
            paths[suffix] = path
    finally:
        plt.close(fig)
    return paths
