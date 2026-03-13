import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from ..typing import ChannelLabelImage2D, Image2D

def plot_hist(
    data: np.ndarray | torch.Tensor,
    ax: mpl.axes.Axes | None = None
    ) -> None:
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if ax is None:
        ax = plt.gca()
        show = True
    else:
        show = False
    ax.hist(data.flatten(), bins=50, color='gray')
    if show:
        plt.show()

def plot_slice(
    data: Image2D,
    ax: mpl.axes.Axes | None = None,
    labels: ChannelLabelImage2D | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    ) -> None:
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if ax is None:
        ax = plt.gca()
        show = True
    else:
        show = False

    # Plot slice.
    ax.imshow(data.T, cmap='gray')

    # Plot labels.
    if labels is not None:
        palette = sns.color_palette('colorblind', len(labels))
        for i, l in enumerate(labels):
            cmap = mpl.colors.ListedColormap(((1, 1, 1, 0), palette[i]))
            ax.imshow(l.T, alpha=0.3, cmap=cmap)
            ax.contour(l.T, colors=[palette[i]], levels=[.5], linestyles='solid')

    # Hide axis spines.
    for p in ['right', 'top', 'bottom', 'left']:
        ax.spines[p].set_visible(False)

    # Add text.
    if title is not None:
        ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if show:
        plt.show()
