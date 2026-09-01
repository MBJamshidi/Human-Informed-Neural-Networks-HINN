"""Shared Nature-standard matplotlib styling for the real-world figure suite."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# professional scientific palette
TEAL = "#127475"     # deep teal   -> HINN (full)
VIOLET = "#7B6CB0"   # soft violet -> Data-only
CORAL = "#E0795B"    # subdued coral -> Human-only
GREY = "#4D4D4D"
INK = "#1A1A1A"

PALETTE = {"HINN (full)": TEAL, "Data-only": VIOLET, "Human-only": CORAL}


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.labelsize": 12, "axes.labelweight": "bold",
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "legend.frameon": True,
        "legend.edgecolor": "0.8", "legend.framealpha": 0.95,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GREY, "axes.linewidth": 1.0,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.2,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   [saved] {path}")
