import os
import matplotlib.pyplot as plt


def apply_plot_style():
    """Set global matplotlib style for consistent plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
    })


def plot_with_style(figsize=(10, 6)):
    """Create a styled figure and axes."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, alpha=0.3)
    return fig, ax


def save_fig(filename, fig=None):
    """Save figure to the figures directory with standard parameters."""
    os.makedirs('figures', exist_ok=True)
    path = os.path.join('figures', filename)
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    return path
