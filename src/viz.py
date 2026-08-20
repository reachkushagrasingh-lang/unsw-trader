"""A small, consistent figure style so every exhibit shares one visual language.

Using a coherent custom palette/type system (rather than matplotlib defaults) is
one of the presentation-band signals in the brief. Extend this to make it your own.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = ["#1b3a5b", "#2e8b8b", "#e08a3c", "#a83e5b", "#6b7a8f", "#3c8c5a"]
GRID = "#dfe3e8"


def apply_style():
    mpl.rcParams.update({
        "figure.figsize": (9, 5), "figure.dpi": 120,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 10, "font.size": 10,
        "legend.frameon": False,
    })


def savefig(fig, path, caption_footer=None):
    if caption_footer:
        fig.text(0.01, 0.01, caption_footer, fontsize=7, color="#6b7a8f")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)