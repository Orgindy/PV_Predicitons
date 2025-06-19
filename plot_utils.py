import os
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')


def apply_standard_plot_style(ax, title=None, xlabel=None, ylabel=None, grid=True):
    ax.tick_params(labelsize=12)
    ax.set_xlabel(xlabel or "", fontsize=14)
    ax.set_ylabel(ylabel or "", fontsize=14)
    if title:
        ax.set_title(title, fontsize=16)
    if grid:
        ax.grid(True, linestyle='--', alpha=0.5)


def save_figure(fig, filename, dpi=300, folder='figures'):
    os.makedirs(folder, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{folder}/{filename}", dpi=dpi)
    plt.close(fig)
