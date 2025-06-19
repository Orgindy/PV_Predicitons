import os
import matplotlib.pyplot as plt

STYLE_PARAMS = {
    "figure.figsize": (10, 6),
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
}


def apply_plot_style():
    """Apply global matplotlib style settings."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(STYLE_PARAMS)


def finalize_plot(filename: str | None = None):
    """Grid, layout, save to figures directory, and show."""
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if filename:
        os.makedirs("figures", exist_ok=True)
        path = os.path.join("figures", filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {path}")
    plt.show()
