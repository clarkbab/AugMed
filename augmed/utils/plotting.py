import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from typing import Literal

from ..typing import AffineMatrix3D, AffineMatrix3DArray, BatchLabelImage3DArray, ChannelLabelImage2D, ChannelLabelImage3D, Image2D, Image3D, Number, Orientation, Points, Points3D, Points3DArray, Size3DArray
from .assertions import assert_orientation
from .conversion import to_numpy
from .geometry import com, foreground_fov_centre, fov, to_image_coords
from .logging import logger
from .matrix import affine_origin, affine_spacing

VIEWS = ['Sagittal', 'Coronal', 'Axial']

def _get_view_aspect(
    view: int,
    affine: np.ndarray | None,
    ) -> float | None:
    if affine is None:
        return None
    spacing = affine_spacing(affine)
    axes = [i for i in range(3) if i != view]
    aspect = float(spacing[axes[1]] / spacing[axes[0]])
    return aspect

def _get_view_origin(
    view: int,
    orientation: Orientation = 'LPS',
    ) -> tuple[Literal['lower', 'upper'], Literal['lower', 'upper']]:
    assert_orientation(orientation)
    if view == 0:
        origin_x = 'lower' if orientation[1] == 'P' else 'upper'
        origin_y = 'lower' if orientation[2] == 'S' else 'upper'
    elif view == 1:
        origin_x = 'lower' if orientation[0] == 'L' else 'upper'
        origin_y = 'lower' if orientation[2] == 'S' else 'upper'
    else:
        origin_x = 'lower' if orientation[0] == 'L' else 'upper'
        origin_y = 'upper' if orientation[1] == 'P' else 'lower'

    return (origin_x, origin_y)

def _get_view_slice(
    view: int,
    data: np.ndarray,
    idx: int,
    ) -> np.ndarray:
    slicing: list[int | slice] = [slice(None)] * 3
    slicing[view] = idx
    return data[tuple(slicing)]

def _get_view_xy(
    view: int,
    values: tuple | np.ndarray,
    ) -> tuple:
    axes = [i for i in range(3) if i != view]
    return values[axes[0]], values[axes[1]]

def plot_hist(
    data: np.ndarray | torch.Tensor,
    ax: mpl.axes.Axes | None = None,
    bins: int = 50,
    log_scale: bool = False,
    min: Number | None = None,
    max: Number | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    ) -> mpl.axes.Axes:
    data = to_numpy(data)
    if ax is None:
        ax = plt.gca()
        show = True
    else:
        show = False
    okwargs = {}
    if min is not None and max is not None:
        okwargs['range'] = (min, max)
    ax.hist(data.flatten(), bins=bins, color='gray', **okwargs)
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
    show_hist: bool = False,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    x_label: str | None = None,
    x_origin: Literal['lower', 'upper'] | None = 'lower',
    y_label: str | None = None,
    y_origin: Literal['lower', 'upper'] | None = 'upper',
    ) -> mpl.axes.Axes:
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if ax is None:
        if show_hist:
            _, axs = plt.subplots(1, 2, figsize=(12, 6))
        else:
            axs = [plt.gca()]
        show = True
    else:
        axs = [ax]
        show = False

    # Plot slice.
    axs[0].imshow(data.T, cmap=cmap, origin=y_origin, vmax=vmax, vmin=vmin)

    # Plot labels.
    if labels is not None:
        palette = sns.color_palette('colorblind', len(labels))
        for i, l in enumerate(labels):
            cmap_label = mpl.colors.ListedColormap(((1, 1, 1, 0), palette[i]))
            axs[0].imshow(l.T, alpha=alpha, cmap=cmap_label)
            axs[0].contour(l.T, colors=[palette[i]], levels=[.5], linestyles='solid')

    # Add histogram.
    if show_hist:
        axs[1].hist(data.flatten(), bins=50, color='gray')

    # Hide axis spines and ticks.
    for p in ['right', 'top', 'bottom', 'left']:
        axs[0].spines[p].set_visible(False)
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    # Add text.
    if title is not None:
        axs[0].set_title(title)
    if x_label is not None:
        axs[0].set_xlabel(x_label)
    if y_label is not None:
        axs[0].set_ylabel(y_label)

    if show:
        plt.show()

    return axs[0]

def plot_volume(
    data: Image3D,
    affine: AffineMatrix3D | None = None,
    cmap: str = 'gray',
    dose: Image3D | None = None,
    dose_alpha_min: float = 0.3,
    dose_alpha_max: float = 1.0,
    dose_cmap: str = 'turbo',
    dose_cmap_trunc: float = 0.15,
    figsize: tuple[float, float] = (16, 6),
    idx: int | float | str | None = None,
    labels: ChannelLabelImage3D | None = None,
    centre_method: Literal['com', 'fov'] = 'com',
    orientation: Orientation = 'LPS',
    label_alpha: float = 0.3,
    points: Points3D | None = None,
    points_colour: str = 'yellow',
    show_point_idxs: bool = False,
    show_title: bool = True,
    use_image_coords: bool = False,
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
    # Check for empty points array - could be filtered by the transform.
    if points is not None:
        if points.shape[0] == 0:
            logger.warn("Points array is empty. No points will be plotted.")
            points = None
            if isinstance(idx, str) and idx.startswith('points:'):
                idx = None
        else:
            assert points.shape[1] == 3, f"Expected points to have shape (N, 3) but got {points.shape}."

    # Resolve views.
    views = list(range(3)) if view == 'all' else (view if isinstance(view, list) else [view])

    palette = sns.color_palette('colorblind', 20)

    fig, axs = plt.subplots(1, len(views), figsize=figsize, squeeze=False)
    axs = axs[0]

    for col_ax, v in zip(axs, views):
        resolved_idx = _get_view_idx(v, data.shape, affine=affine, centre_method=centre_method, idx=idx, labels=labels, points=points)
        image = _get_view_slice(v, data, resolved_idx)
        aspect = _get_view_aspect(v, affine)
        origin_x, origin_y = _get_view_origin(v, orientation=orientation)

        # The two non-view axes: first is displayed on x, second on y.
        col_ax.imshow(image.T, aspect=aspect, cmap=cmap, origin=origin_y, vmax=vmax, vmin=vmin)
        if origin_x == 'upper':
            col_ax.invert_xaxis()

        # Dose overlay.
        if dose is not None:
            dose_slice = _get_view_slice(v, dose, resolved_idx)
            base_cmap = plt.get_cmap(dose_cmap)
            trunc_cmap = mpl.colors.LinearSegmentedColormap.from_list(
                f'{base_cmap.name}_truncated',
                base_cmap(np.linspace(dose_cmap_trunc, 1.0, 256)),
            )
            colours = trunc_cmap(np.arange(trunc_cmap.N))
            colours[0, -1] = 0
            colours[1:, -1] = np.linspace(dose_alpha_min, dose_alpha_max, trunc_cmap.N - 1)
            alpha_cmap = mpl.colors.ListedColormap(colours)
            col_ax.imshow(dose_slice.T, aspect=aspect, cmap=alpha_cmap, origin=origin_y)

        # Label overlays.
        if labels is not None:
            for j, lab in enumerate(labels):
                label_slice = _get_view_slice(v, lab, resolved_idx)
                cmap_label = mpl.colors.ListedColormap(((1, 1, 1, 0), palette[j]))
                col_ax.imshow(label_slice.T, alpha=label_alpha, aspect=aspect, cmap=cmap_label, origin=origin_y)
                col_ax.contour(label_slice.T, colors=[palette[j]], levels=[0.5], linestyles='solid')

        # Point overlays.
        if points is not None:
            view_axes = [i for i in range(3) if i != v]
            if affine is not None:
                spacing = affine_spacing(affine)
                origin = affine_origin(affine)
            if points_colour == 'gradient' and len(points) > 1:
                points_cmap = mpl.colors.LinearSegmentedColormap.from_list('warm_bright', ['#FFE600', '#FF8C00', '#FF3300', '#FF0066'])
                points_colours = [points_cmap(i / (len(points) - 1)) for i in range(len(points))]
            else:
                points_colours = ['yellow'] * len(points)
            for pi, p in enumerate(points):
                vox = (p - origin) / spacing if affine is not None else p
                col_ax.scatter(vox[view_axes[0]], vox[view_axes[1]], c=[points_colours[pi]], marker='o', s=20, zorder=5)
                if show_point_idxs:
                    col_ax.annotate(str(pi), (vox[view_axes[0]], vox[view_axes[1]]),
                        color=points_colours[pi], fontsize=8,
                        textcoords='offset points', xytext=(5, 5), zorder=5)

        # Coordinate ticks.
        size_x, size_y = _get_view_xy(v, data.shape)
        x_tick_spacing = np.unique(np.diff(col_ax.get_xticks()))[0]
        x_ticks = np.arange(0, size_x, x_tick_spacing)
        y_tick_spacing = np.unique(np.diff(col_ax.get_yticks()))[0]
        y_ticks = np.arange(0, size_y, y_tick_spacing)

        if not use_image_coords and affine is not None:
            s = affine_spacing(affine)
            o = affine_origin(affine)
            sx, sy = _get_view_xy(v, s)
            ox, oy = _get_view_xy(v, o)
            col_ax.set_xticks(x_ticks)
            col_ax.set_xticklabels([f'{t * sx + ox:.1f}' for t in x_ticks])
            col_ax.set_yticks(y_ticks)
            col_ax.set_yticklabels([f'{t * sy + oy:.1f}' for t in y_ticks])
        else:
            col_ax.set_xticks(x_ticks)
            col_ax.set_yticks(y_ticks)

        # Title.
        if show_title:
            title = f'{VIEWS[v]}, slice {resolved_idx}'
            if affine is not None:
                s = affine_spacing(affine)
                o = affine_origin(affine)
                world_pos = resolved_idx * s[v] + o[v]
                title += f' ({world_pos:.1f}mm)'
            col_ax.set_title(title)

        # Hide spines.
        for p in ['right', 'top', 'bottom', 'left']:
            col_ax.spines[p].set_visible(False)

    plt.tight_layout()
    plt.show()

def _get_view_idx(
    view: int,
    size: Size3DArray,
    affine: AffineMatrix3DArray | None = None,
    idx: int | float | str | None = None,
    labels: BatchLabelImage3DArray | None = None,
    centre_method: Literal['com', 'fov'] = 'com',
    points: Points3DArray | None = None,
    ) -> int:
    # Default to middle slice.
    if idx is None:
        idx = 'p:0.5'

    # World coords.
    if isinstance(idx, (int, float)) and not isinstance(idx, bool):
        if affine is not None:
            idx = to_image_coords(idx, affine)
        return int(np.clip(np.round(idx), 0, size[view] - 1))

    # String prefixes.
    if not isinstance(idx, str):
        raise ValueError(f"Invalid idx: {idx}. Expected int, float, str, or None.")

    source, value = idx.split(':')

    # Proportion of field-of-view.
    if source == 'p':
        p = float(value)
        return int(np.clip(np.round(p * (size[view] - 1)), 0, size[view] - 1))

    # Image coords.
    if source == 'i':
        return int(np.clip(int(value), 0, size[view] - 1))

    # Label channels.
    if source == 'labels':
        if labels is None:
            raise ValueError(f"idx='{idx}' but no labels were provided.")
        if centre_method == 'com':
            centre = com(labels[int(value)], affine=affine)
        elif centre_method == 'fov':
            centre = foreground_fov_centre(labels[int(value)], affine=affine)
        else:
            raise ValueError(f"Unknown centre_method '{centre_method}'. Expected 'com' or 'fov'.")

    # Points.
    elif source == 'points':
        if points is None:
            raise ValueError(f"idx='{idx}' but no points were provided.")
        centre = points[int(value)]

    else:
        raise ValueError(f"Unknown idx prefix '{source}'. Expected 'p', 'i', 'labels', or 'points'.")

    # Convert world coords to voxel coords.
    if affine is not None:
        centre = to_image_coords(centre, affine)

    return int(np.clip(np.round(centre[view]), 0, size[view] - 1))
