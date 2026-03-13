from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from typing import Literal

from ..typing import (
    Affine3D, ChannelLabelImage2D, ChannelLabelImage3D, Image2D, Image3D,
    Point3D, Points3D,
)
from .conversion import to_numpy
from .matrix import affine_spacing

def plot_hist(
    data: np.ndarray | torch.Tensor,
    ax: mpl.axes.Axes | None = None,
    bins: int = 50,
    log_scale: bool = False,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    ) -> mpl.axes.Axes:
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if ax is None:
        ax = plt.gca()
        show = True
    else:
        show = False
    ax.hist(data.flatten(), bins=bins, color='gray')
    if log_scale:
        ax.set_yscale('log')

    # Add text.
    if title is not None:
        ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if show:
        plt.show()

    return ax

def plot_slice(
    data: Image2D,
    alpha: float = 0.3,
    ax: mpl.axes.Axes | None = None,
    cmap: str = 'gray',
    labels: ChannelLabelImage2D | None = None,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    ) -> mpl.axes.Axes:
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
    ax.imshow(data.T, cmap=cmap, vmin=vmin, vmax=vmax)

    # Plot labels.
    if labels is not None:
        palette = sns.color_palette('colorblind', len(labels))
        for i, l in enumerate(labels):
            cmap_label = mpl.colors.ListedColormap(((1, 1, 1, 0), palette[i]))
            ax.imshow(l.T, alpha=alpha, cmap=cmap_label)
            ax.contour(l.T, colors=[palette[i]], levels=[.5], linestyles='solid')

    # Hide axis spines and ticks.
    for p in ['right', 'top', 'bottom', 'left']:
        ax.spines[p].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add text.
    if title is not None:
        ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if show:
        plt.show()

    return ax

_VIEW_NAMES = ['Sagittal', 'Coronal', 'Axial']

def _get_view_slice(
    data: np.ndarray,
    view: int,
    idx: int,
    ) -> np.ndarray:
    slicing: list[int | slice] = [slice(None)] * 3
    slicing[view] = idx
    return data[tuple(slicing)]

def _resolve_idx(
    shape: tuple[int, ...],
    view: int,
    idx: int | float,
    affine: np.ndarray | None = None,
    centre: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    ) -> int:
    n = shape[view]

    # Centre on label FOV midpoint.
    if centre is not None:
        if centre.ndim >= 2:
            # Label volume(s): find midpoint of foreground along this axis.
            fg = centre.any(axis=0) if centre.ndim == 4 else centre
            coords = np.argwhere(fg)
            if len(coords) > 0:
                return int(coords[:, view].mean())
        else:
            # Point in world or voxel coordinates.
            pt = centre.flatten()
            if affine is not None:
                spacing = np.abs(np.diag(affine)[:3])
                origin = affine[:3, -1]
                vox = (pt - origin) / spacing
                return int(np.clip(np.round(vox[view]), 0, n - 1))
            return int(np.clip(np.round(pt[view]), 0, n - 1))

    # Fractional index.
    if isinstance(idx, float) and 0 <= idx <= 1:
        return int(np.round(idx * (n - 1)))

    return int(np.clip(idx, 0, n - 1))

def _get_view_aspect(
    view: int,
    affine: np.ndarray | None,
    ) -> float | str:
    if affine is None:
        return None
    spacing = affine_spacing(affine)
    axes = [i for i in range(3) if i != view]
    aspect = float(spacing[axes[0]] / spacing[axes[1]])
    print(f'Aspect for view {view} with spacing {spacing}: {aspect}')
    return aspect

def plot_volume(
    data: Image3D,
    affine: Affine3D | None = None,
    centre: ChannelLabelImage3D | Point3D | None = None,
    cmap: str = 'gray',
    dose: Image3D | None = None,
    dose_alpha_min: float = 0.3,
    dose_alpha_max: float = 1.0,
    dose_cmap: str = 'turbo',
    dose_cmap_trunc: float = 0.15,
    figsize: tuple[float, float] = (16, 6),
    idx: int | float = 0.5,
    labels: ChannelLabelImage3D | None = None,
    label_alpha: float = 0.3,
    points: Points3D | None = None,
    show_title: bool = True,
    view: int | list[int] | Literal['all'] = 'all',
    vmin: float | None = None,
    vmax: float | None = None,
    ) -> np.ndarray:
    # Convert inputs to numpy.
    data = to_numpy(data)
    affine = to_numpy(affine)
    dose = to_numpy(dose)
    labels = to_numpy(labels)
    points = to_numpy(points)
    centre = to_numpy(centre)

    # Resolve views.
    views = list(range(3)) if view == 'all' else (view if isinstance(view, list) else [view])

    palette = sns.color_palette('colorblind', 20)

    fig, axs = plt.subplots(1, len(views), figsize=figsize, squeeze=False)
    axs = axs[0]

    for col_ax, v in zip(axs, views):
        resolved_idx = _resolve_idx(data.shape, v, idx, affine=affine, centre=centre, labels=labels)
        image = _get_view_slice(data, v, resolved_idx)
        aspect = _get_view_aspect(v, affine)

        # The two non-view axes: first is displayed on x, second on y.
        col_ax.imshow(image.T, aspect=aspect, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)

        # Dose overlay.
        if dose is not None:
            dose_slice = _get_view_slice(dose, v, resolved_idx)
            base_cmap = plt.get_cmap(dose_cmap)
            trunc_cmap = mpl.colors.LinearSegmentedColormap.from_list(
                f'{base_cmap.name}_truncated',
                base_cmap(np.linspace(dose_cmap_trunc, 1.0, 256)),
            )
            colours = trunc_cmap(np.arange(trunc_cmap.N))
            colours[0, -1] = 0
            colours[1:, -1] = np.linspace(dose_alpha_min, dose_alpha_max, trunc_cmap.N - 1)
            alpha_cmap = mpl.colors.ListedColormap(colours)
            col_ax.imshow(dose_slice.T, aspect=aspect, cmap=alpha_cmap, origin='lower')

        # Label overlays.
        if labels is not None:
            for j, lab in enumerate(labels):
                label_slice = _get_view_slice(lab, v, resolved_idx)
                cmap_label = mpl.colors.ListedColormap(((1, 1, 1, 0), palette[j]))
                col_ax.imshow(label_slice.T, alpha=label_alpha, aspect=aspect, cmap=cmap_label, origin='lower')
                col_ax.contour(label_slice.T, colors=[palette[j]], levels=[0.5], linestyles='solid')

        # Point overlays.
        if points is not None:
            view_axes = [i for i in range(3) if i != v]
            if affine is not None:
                spacing = np.abs(np.diag(affine)[:3])
                origin = affine[:3, -1]
            for pt in points:
                if affine is not None:
                    vox = (pt - origin) / spacing
                else:
                    vox = pt
                # Only plot points near the current slice.
                if abs(vox[v] - resolved_idx) < 1.0:
                    col_ax.scatter(vox[view_axes[0]], vox[view_axes[1]], c='red', marker='x', s=40, zorder=5)

        # Title.
        if show_title:
            title = f'{_VIEW_NAMES[v]}, slice {resolved_idx}'
            if affine is not None:
                spacing = np.abs(np.diag(affine)[:3])
                origin_val = affine[:3, -1]
                world_pos = resolved_idx * spacing[v] + origin_val[v]
                title += f' ({world_pos:.1f}mm)'
            col_ax.set_title(title)

        # Hide spines and ticks.
        for p in ['right', 'top', 'bottom', 'left']:
            col_ax.spines[p].set_visible(False)
        col_ax.set_xticks([])
        col_ax.set_yticks([])

    plt.tight_layout()
    plt.show()
    return axs
