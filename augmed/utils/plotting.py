import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from typing import List, Literal, Tuple

from ..defaults import get_orientation
from ..typing import AffineMatrix, AffineMatrix2D, AffineMatrix3D, AffineMatrix3DArray, BatchBox2D, BatchBox3D, BatchLabelImage, BatchLabelImage2D, BatchLabelImage3D, BatchLabelImage3DArray, BatchPoints, BatchPoints2D, BatchPoints3D, Box2D, Box3D, Image, Image2D, Image2DArray, Image3D, Image3DArray, LabelImage, LabelImage2D, LabelImage3D, Number, Orientation2D, Orientation3D, Point, Point2D, Point3D, Points, Points2D, Points3D, Size3DArray, View
from .args import alias_kwargs, arg_default, arg_to_list
from .assertions import assert_image_shapes, assert_orientation
from .conversion import to_numpy
from .geometry import affine_origin, affine_spacing, centre_of_mass, foreground_fov, foreground_fov_centre, fov, to_image_coords
from .logging import logger

VIEWS = ['Sagittal', 'Coronal', 'Axial']

# CT windowing presets: (width, level).
Window = str | tuple[Number, Number]
WINDOW_PRESETS = {
    'bone': (1800, 400),
    'brain': (80, 40),
    'liver': (150, 30),
    'lung': (1500, -600),
    'mediastinum': (350, 50),
    'tissue': (400, 50),
}

def __get_origin_2d(
    orientation: Orientation2D,
    ) -> Tuple[Literal['lower', 'upper'], Literal['lower', 'upper']]:
    assert_orientation(orientation, 2)
    origin_x = 'lower' if orientation[0] == 'L' else 'upper'
    origin_y = 'lower' if orientation[1] == 'S' else 'upper'
    return (origin_x, origin_y)

def __get_view_aspect(
    view: View,
    affine: AffineMatrix3DArray | None,
    ) -> float | None:
    if affine is None:
        return None
    spacing = affine_spacing(affine)
    axes = [i for i in range(3) if i != view]
    aspect = float(spacing[axes[1]] / spacing[axes[0]])
    return aspect

def __get_view_idx(
    view: View,
    size: Size3DArray,
    affine: AffineMatrix3DArray | None = None,
    idx: int | float | str | Point3D | None = None,
    labels: BatchLabelImage3DArray | None = None,
    label_names: List[str] | None = None,
    centre_method: Literal['com', 'fov'] = 'com',
    points: List[np.ndarray] | None = None,
    point_names: List[List[str]] | None = None,
    ) -> int:
    # Default to middle slice.
    if idx is None:
        idx = 'f:0.5'

    # Point3D - a 3D world coordinate (tuple, list, np.ndarray, or torch.Tensor).
    if isinstance(idx, (tuple, list, np.ndarray, torch.Tensor)) and not isinstance(idx, bool):
        idx = np.asarray(idx).flatten()
        if len(idx) != 3:
            raise ValueError(f"Expected a 3-element point for idx but got {len(idx)} elements.")
        if affine is not None:
            idx = to_image_coords(idx, affine)
        return int(np.clip(np.round(idx[view]), 0, size[view] - 1))

    # Scalar world coords.
    if isinstance(idx, (int, float)) and not isinstance(idx, bool):
        if affine is not None:
            idx = to_image_coords(idx, affine)
        return int(np.clip(np.round(idx), 0, size[view] - 1))

    # String prefixes.
    if not isinstance(idx, str):
        raise ValueError(f"Invalid idx: {idx}. Expected int, float, str, Point3D, or None.")

    source, value = idx.split(':')

    # Proportion of field-of-view.
    if source == 'f':
        p = float(value)
        return int(np.clip(np.round(p * (size[view] - 1)), 0, size[view] - 1))

    # Image coords.
    if source == 'i':
        return int(np.clip(int(value), 0, size[view] - 1))

    # Label channels - by index (e.g. "labels:0") or name (e.g. "labels:Brainstem").
    if source in ('l', 'label', 'labels'):
        if labels is None:
            raise ValueError(f"idx='{idx}' but no labels were provided.")

        # Resolve label index from name or integer string.
        if value.isdigit():
            label_idx = int(value)
        else:
            if label_names is None:
                raise ValueError(f"idx='{idx}' uses a label name but no 'label_names' were provided.")
            if value not in label_names:
                raise ValueError(f"Label name '{value}' not found in label_names: {label_names}.")
            label_idx = label_names.index(value)

        if centre_method == 'com':
            centre = centre_of_mass(labels[label_idx], affine=affine)
        elif centre_method == 'fov':
            centre = foreground_fov_centre(labels[label_idx], affine=affine)
        else:
            raise ValueError(f"Unknown centre_method '{centre_method}'. Expected 'com' or 'fov'.")

    # Points.
    elif source in ('p', 'point', 'points'):
        if points is None:
            raise ValueError(f"idx='{idx}' but no points were provided.")
        value_parts = value.split(':')
        if len(value_parts) == 1:
            batch_idx, point_ref = 0, value_parts[0]
        else:
            batch_idx, point_ref = int(value_parts[0]), value_parts[1]
        batch = points[batch_idx]
        if point_ref.lstrip('-').isdigit():
            centre = batch[int(point_ref)]
        else:
            batch_names = point_names[batch_idx] if point_names is not None else None
            if batch_names is None:
                raise ValueError(f"idx='{idx}' uses a point name but no 'point_names' were provided.")
            if point_ref not in batch_names:
                raise ValueError(f"Point name '{point_ref}' not found in point_names: {batch_names}.")
            centre = batch[batch_names.index(point_ref)]

    else:
        raise ValueError(f"Unknown idx prefix '{source}'. Expected 'f', 'i', 'l', 'label', 'labels', 'p', 'point', or 'points'.")

    # Convert world coords to voxel coords.
    if affine is not None:
        centre = to_image_coords(centre, affine)

    return int(np.clip(np.round(centre[view]), 0, size[view] - 1))

def __get_view_origin(
    view: View,
    orientation: Orientation3D,
    ) -> Tuple[Literal['lower', 'upper'], Literal['lower', 'upper']]:
    assert_orientation(orientation, 3)
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

def __get_view_slice(
    view: View,
    data: Image3DArray,
    idx: int,
    ) -> Image2DArray:
    slicing: list[int | slice] = [slice(None)] * 3
    slicing[view] = idx
    return data[tuple(slicing)]

def __get_view_xy(
    view: View,
    values: Tuple | np.ndarray,
    ) -> Tuple:
    axes = [i for i in range(3) if i != view]
    return values[axes[0]], values[axes[1]]

def plot_hist(
    data: Image,
    ax: mpl.axes.Axes | None = None,
    bins: int = 50,
    log_scale: bool = False,
    min: Number | None = None,
    max: Number | None = None,
    return_axis: bool = False,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    ) -> mpl.axes.Axes | None:
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

    if return_axis:
        return ax

@alias_kwargs(
    ('a', 'affine'),
    ('b', 'box'),
    ('c', 'crop'),
    ('cm', 'crop_margin'),
    ('l', 'labels'),
    ('ln', 'label_names'),
    ('o', 'orientation'),
    ('p', 'points'),
    ('sl', 'show_labels'),
    ('spn', 'show_point_names'),
    ('w', 'window'),
)
def plot_slice(
    data: Image2D | None,
    affine: AffineMatrix2D | None = None,
    alpha: Number = 0.3,
    aspect: Number | None = None,
    ax: mpl.axes.Axes | None = None,
    box: Box2D | BatchBox2D | str | List[str] | None = None,
    cmap: str = 'gray',
    crop: Box2D | Point2D | str | None = None,
    crop_margin: Number = 100.0,
    labels: LabelImage2D | BatchLabelImage2D | None = None,
    label_names: str | List[str] | None = None,
    figsize: Tuple[Number, Number] = (8, 8),
    orientation: Orientation2D | None = None,
    point_names: str | List[str] | None = None,
    points: Point2D | Points2D | BatchPoints2D | List[Points2D] | None = None,
    points_colour: str = 'yellow',
    return_axis: bool = False,
    show_labels: bool = True,
    show_point_idxs: bool = False,
    show_point_names: bool = False,
    title: str | None = None,
    title_fontsize: float = 10,
    use_image_coords: bool = False,
    vmin: Number | None = None,
    vmax: Number | None = None,
    window: Window | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    ) -> mpl.axes.Axes | None:
    if data is None:
        assert labels is not None, "Labels must be provided if data is None."
        data = np.zeros(labels.shape[-2:])
    orientation = arg_default(orientation, orientation, get_orientation(2))

    # Convert inputs to numpy.
    data = to_numpy(data)
    data = data.squeeze()  # Remove any singleton dimensions.
    assert_image_shapes(data, 2, batch_allowed=False, channel_allowed=False, suggest_dim=False)
    affine = to_numpy(affine)

    # Resolve labels.
    labels = __resolve_labels(labels, dim=2)

    # Resolve window to vmin/vmax.
    vmin, vmax = __resolve_window(window, vmin, vmax, data=data, label_names=label_names, labels=labels)

    # Resolve points and point names.
    points, point_names = __resolve_points(points, affine=affine, point_names=point_names)

    # Resolve crop.
    crop_box = __resolve_crop(crop, crop_margin, data.shape, affine=affine, label_names=label_names, labels=labels, point_names=point_names, points=points)

    # Resolve boxes for overlay.
    boxes = __resolve_boxes(box, data.shape, affine=affine, label_names=label_names, labels=labels, point_names=point_names, points=points)

    if ax is None:
        _, axs = plt.subplots(1, 1, figsize=figsize)
        axs = [axs]
        show = True
    else:
        axs = [ax]
        show = False

    # Aspect ratio from affine.
    if aspect is None and affine is not None:
        spacing = affine_spacing(affine)
        aspect = float(spacing[1] / spacing[0])

    # Plot slice.
    if crop_box is not None:
        data = data[crop_box[0, 0]:crop_box[1, 0] + 1, crop_box[0, 1]:crop_box[1, 1] + 1]
    origin_x, origin_y = __get_origin_2d(orientation)
    axs[0].imshow(data.T, aspect=aspect, cmap=cmap, origin=origin_y, vmax=vmax, vmin=vmin)
    if origin_x == 'upper':
        axs[0].invert_xaxis()

    # Plot labels.
    if show_labels and labels is not None:
        label_names_list = arg_to_list(label_names, str) if label_names is not None else None
        label_palette = sns.color_palette('colorblind', len(labels))
        for i, l in enumerate(labels):
            if crop_box is not None:
                l = l[crop_box[0, 0]:crop_box[1, 0] + 1, crop_box[0, 1]:crop_box[1, 1] + 1]
            l_bin = (l > 0).astype(float)
            cmap_label = mpl.colors.ListedColormap(((1, 1, 1, 0), label_palette[i]))
            axs[0].imshow(l_bin.T, alpha=alpha, cmap=cmap_label, origin=origin_y)
            axs[0].contour(l_bin.T, colors=[label_palette[i]], levels=[0.5], linestyles='solid')

        # Add legend.
        if label_names_list is not None:
            handles = [mpl.patches.Patch(facecolor=label_palette[i], label=label_names_list[i]) for i in range(len(labels)) if i < len(label_names_list)]
            axs[0].legend(fontsize='small', framealpha=0.7, handles=handles, loc='upper right')

    # Plot points.
    if points is not None:
        n_batches = len(points)
        batch_palette = sns.color_palette('colorblind', n_batches)
        for bi, batch in enumerate(points):
            batch_colour = batch_palette[bi]
            if points_colour == 'gradient' and len(batch) > 1:
                r, g, b = batch_colour[:3]
                light = (r + (1 - r) * 0.7, g + (1 - g) * 0.7, b + (1 - b) * 0.7)
                points_cmap = mpl.colors.LinearSegmentedColormap.from_list('batch_grad', [light, batch_colour])
                p_colours = [points_cmap(i / (len(batch) - 1)) for i in range(len(batch))]
            else:
                p_colours = [batch_colour] * len(batch)
            for pi, p in enumerate(batch):
                if crop_box is not None:
                    p = p - crop_box[0]
                axs[0].scatter(p[0], p[1], c=[p_colours[pi]], marker='o', s=20, zorder=5)
                if show_point_names and points is not None:
                    axs[0].annotate(points[bi][pi], (p[0], p[1]),
                        color=p_colours[pi], fontsize=8,
                        textcoords='offset points', xytext=(5, 5), zorder=5)
                elif show_point_idxs:
                    axs[0].annotate(str(pi), (p[0], p[1]),
                        color=p_colours[pi], fontsize=8,
                        textcoords='offset points', xytext=(5, 5), zorder=5)

    # Box overlays.
    if boxes is not None:
        box_palette = sns.color_palette('colorblind', len(boxes))
        for bi, b in enumerate(boxes):
            width = b[1][0] - b[0][0]
            height = b[1][1] - b[0][1]
            rect = mpl.patches.Rectangle((b[0][0], b[0][1]), width, height, edgecolor=box_palette[bi % len(box_palette)], facecolor='none', linestyle='dotted', linewidth=2, zorder=10)
            axs[0].add_patch(rect)

    # Coordinate ticks.
    size = data.shape
    x_tick_spacing = np.unique(np.diff(axs[0].get_xticks()))[0]
    x_ticks = np.arange(0, size[0], x_tick_spacing)
    y_tick_spacing = np.unique(np.diff(axs[0].get_yticks()))[0]
    y_ticks = np.arange(0, size[1], y_tick_spacing)
    axs[0].set_xticks(x_ticks)
    axs[0].set_yticks(y_ticks)

    # Create tick labels.
    if crop_box is not None:
        x_ticks = x_ticks + crop_box[0, 0]
        y_ticks = y_ticks + crop_box[0, 1]
    if not use_image_coords and affine is not None:
        spacing = affine_spacing(affine)
        origin = affine_origin(affine)
        x_ticks = x_ticks * spacing[0] + origin[0]
        y_ticks = y_ticks * spacing[1] + origin[1]
    axs[0].set_xticklabels([f'{t:.1f}' for t in x_ticks])
    axs[0].set_yticklabels([f'{t:.1f}' for t in y_ticks])

    # Hide axis spines.
    for p in ['right', 'top', 'bottom', 'left']:
        axs[0].spines[p].set_visible(False)

    # Add text.
    if title is not None:
        axs[0].set_title(title, fontsize=title_fontsize)
    if x_label is not None:
        axs[0].set_xlabel(x_label)
    if y_label is not None:
        axs[0].set_ylabel(y_label)

    if show:
        plt.show()

    if return_axis:
        return axs[0]

@alias_kwargs(
    ('a', 'affine'),
    ('b', 'box'),
    ('c', 'crop'),
    ('ch', 'crosshairs'),
    ('cm', 'crop_margin'),
    ('d', 'dose'),
    ('i', 'idx'),
    ('l', 'labels'),
    ('ln', 'label_names'),
    ('p', 'points'),
    ('scc', 'show_crosshairs_coords'),
    ('sl', 'show_labels'),
    ('spn', 'show_point_names'),
)
def plot_volume(
    data: Image3D | None,
    affine: AffineMatrix3D | None = None,
    ax: mpl.axes.Axes | List[mpl.axes.Axes] | None = None,
    box: Box3D | BatchBox3D | str | List[str] | None = None,
    cmap: str = 'gray',
    crosshairs: Point3D | str | None = None,
    crosshairs_colour: str = 'yellow',
    crop: Box3D | Point3D | str | None = None,
    crop_margin: int = 100,
    dose: Image3D | None = None,
    dose_alpha_min: float = 0.3,
    dose_alpha_max: float = 1.0,
    dose_cmap: str = 'turbo',
    dose_cmap_trunc: float = 0.15,
    figsize: tuple[Number, Number] = (16, 6),
    idx: int | float | str | Point3D | None = None,
    labels: LabelImage3D | BatchLabelImage3D | None = None,
    label_names: str | List[str] | None = None,
    centre_method: Literal['com', 'fov'] = 'com',
    orientation: Orientation3D | None = None,
    label_alpha: float = 0.3,
    point_names: str | List[str] | None = None,
    points: Point3D | Points3D | BatchPoints3D | List[Points3D] | None = None,
    points_colour: str = 'yellow',
    return_axis: bool = False,
    show_crosshairs_coords: bool = True,
    show_labels: bool = True,
    show_point_idxs: bool = False,
    show_point_names: bool = False,
    show_points: bool = True,
    show_title: bool = True,
    title_fontsize: float = 10,
    use_image_coords: bool = False,
    view: View | List[View] | Literal['all'] = 'all',
    vmin: float | None = None,
    vmax: float | None = None,
    window: Window | None = None,
    ) -> np.ndarray | None:
    if data is None:
        assert labels is not None, "Labels must be provided if data is None."
        data = np.zeros(labels.shape[-3:])
    orientation = arg_default(orientation, orientation, get_orientation(3))

    # Convert inputs to numpy.
    data = to_numpy(data)
    data = data.squeeze()  # Remove any singleton dimensions.
    assert_image_shapes(data, 3, batch_allowed=False, channel_allowed=False, suggest_dim=False)
    affine = to_numpy(affine)
    dose = to_numpy(dose)

    # Resolve labels.
    labels = __resolve_labels(labels, dim=3)

    # Resolve window to vmin/vmax.
    vmin, vmax = __resolve_window(window, vmin, vmax, data=data, label_names=label_names, labels=labels)

    # Resolve points and point names.
    points, point_names = __resolve_points(points, affine=affine, point_names=point_names)
    if points is None and isinstance(idx, str) and idx.startswith(('p:', 'point:', 'points:')):
        idx = None

    # Resolve crop.
    crop_box = __resolve_crop(crop, crop_margin, data.shape, affine=affine, label_names=label_names, labels=labels, point_names=point_names, points=points)

    # Resolve boxes for overlay.
    boxes = __resolve_boxes(box, data.shape, affine=affine, label_names=label_names, labels=labels, point_names=point_names, points=points)

    # Resolve crosshairs to image coords.
    crosshairs_vox = __resolve_crosshairs(crosshairs, data.shape, affine=affine, centre_method=centre_method, label_names=label_names, labels=labels, point_names=point_names, points=points)

    # Resolve views.
    views = list(range(3)) if view == 'all' else (view if isinstance(view, list) else [view])

    if ax is not None:
        axs = arg_to_list(ax, mpl.axes.Axes)
        assert len(axs) == len(views), f"Expected {len(views)} axes but got {len(axs)}."
        show = False
    else:
        _, axs_grid = plt.subplots(1, len(views), figsize=figsize, squeeze=False)
        axs = list(axs_grid[0])
        show = True

    for col_ax, v in zip(axs, views):
        resolved_idx = __get_view_idx(v, data.shape, affine=affine, centre_method=centre_method, idx=idx, label_names=label_names, labels=labels, point_names=point_names, points=points)
        x_axis, y_axis = __get_view_xy(v, list(range(3)))
        view_crop_box = crop_box[:, [x_axis, y_axis]] if crop_box is not None else None
        image = __get_view_slice(v, data, resolved_idx)
        if view_crop_box is not None:
            image = image[view_crop_box[0, 0]:view_crop_box[1, 0] + 1, view_crop_box[0, 1]:view_crop_box[1, 1] + 1]
        aspect = __get_view_aspect(v, affine)
        origin_x, origin_y = __get_view_origin(v, orientation)

        # The two non-view axes: first is displayed on x, second on y.
        col_ax.imshow(image.T, aspect=aspect, cmap=cmap, origin=origin_y, vmax=vmax, vmin=vmin)
        if origin_x == 'upper':
            col_ax.invert_xaxis()

        # Dose overlay.
        if dose is not None:
            dose_slice = __get_view_slice(v, dose, resolved_idx)
            if view_crop_box is not None:
                dose_slice = dose_slice[view_crop_box[0, 0]:view_crop_box[1, 0] + 1, view_crop_box[0, 1]:view_crop_box[1, 1] + 1]
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
        if show_labels and labels is not None:
            label_names_list = arg_to_list(label_names, str) if label_names is not None else None
            label_palette = sns.color_palette('colorblind', len(labels))
            for j, lab in enumerate(labels):
                label_slice = __get_view_slice(v, lab, resolved_idx)
                if view_crop_box is not None:
                    label_slice = label_slice[view_crop_box[0, 0]:view_crop_box[1, 0] + 1, view_crop_box[0, 1]:view_crop_box[1, 1] + 1]
                label_bin = (label_slice > 0).astype(float)
                cmap_label = mpl.colors.ListedColormap(((1, 1, 1, 0), label_palette[j]))
                col_ax.imshow(label_bin.T, alpha=label_alpha, aspect=aspect, cmap=cmap_label, origin=origin_y)
                col_ax.contour(label_bin.T, colors=[label_palette[j]], levels=[0.5], linestyles='solid')

            # Add legend on first view only.
            if label_names_list is not None and v == views[0]:
                handles = [mpl.patches.Patch(facecolor=label_palette[j], label=label_names_list[j]) for j in range(len(labels)) if j < len(label_names_list)]
                col_ax.legend(fontsize='small', framealpha=0.7, handles=handles, loc='upper right')

        # Box overlays.
        if boxes is not None:
            boxes2d = boxes[:, :, [x_axis, y_axis]]
            box_palette = sns.color_palette('colorblind', len(boxes2d))
            for bi, b in enumerate(boxes2d):
                width = b[1][0] - b[0][0]
                height = b[1][1] - b[0][1]
                rect = mpl.patches.Rectangle((b[0][0], b[0][1]), width, height, edgecolor=box_palette[bi % len(box_palette)], facecolor='none', linestyle='dotted', linewidth=2, zorder=10)
                col_ax.add_patch(rect)

        # Point overlays.
        if show_points and points is not None:
            n_batches = len(points)
            batch_palette = sns.color_palette('colorblind', n_batches)
            for bi, (batch, batch_names) in enumerate(zip(points, point_names)):
                batch_colour = batch_palette[bi]
                if points_colour == 'gradient' and len(batch) > 1:
                    r, g, b = batch_colour[:3]
                    light = (r + (1 - r) * 0.7, g + (1 - g) * 0.7, b + (1 - b) * 0.7)
                    points_cmap = mpl.colors.LinearSegmentedColormap.from_list('batch_grad', [light, batch_colour])
                    p_colours = [points_cmap(i / (len(batch) - 1)) for i in range(len(batch))]
                else:
                    p_colours = [batch_colour] * len(batch)
                for pi, p in enumerate(batch):
                    p_x, p_y = __get_view_xy(v, p)
                    if view_crop_box is not None:
                        p_x -= view_crop_box[0, 0]
                        p_y -= view_crop_box[0, 1]
                    col_ax.scatter(p_x, p_y, c=[p_colours[pi]], marker='o', s=20, zorder=5)
                    if show_point_names and batch_names is not None:
                        col_ax.annotate(batch_names[pi], (p_x, p_y),
                            color=p_colours[pi], fontsize=8,
                            textcoords='offset points', xytext=(5, 5), zorder=5)
                    elif show_point_idxs:
                        col_ax.annotate(str(pi), (p_x, p_y),
                            color=p_colours[pi], fontsize=8,
                            textcoords='offset points', xytext=(5, 5), zorder=5)

        # Crosshairs.
        if crosshairs_vox is not None:
            ch_x, ch_y = __get_view_xy(v, crosshairs_vox)
            if view_crop_box is not None:
                ch_x -= view_crop_box[0, 0]
                ch_y -= view_crop_box[0, 1]
            col_ax.axvline(color=crosshairs_colour, linestyle='dashed', linewidth=0.5, x=ch_x)
            col_ax.axhline(color=crosshairs_colour, linestyle='dashed', linewidth=0.5, y=ch_y)
            if show_crosshairs_coords:
                if not use_image_coords and affine is not None:
                    s = affine_spacing(affine)
                    o = affine_origin(affine)
                    ch_x_world, ch_y_world = __get_view_xy(v, crosshairs_vox * s + o)
                    ch_label = f'({ch_x_world:.1f}, {ch_y_world:.1f})'
                else:
                    ch_label = f'({ch_x}, {ch_y})'
                col_ax.text(ch_x + 10, ch_y - 10, ch_label, color=crosshairs_colour, fontsize=8)

        # Coordinate ticks.
        size_x, size_y = image.shape
        x_tick_spacing = np.unique(np.diff(col_ax.get_xticks()))[0]
        x_ticks = np.arange(0, size_x, x_tick_spacing)
        y_tick_spacing = np.unique(np.diff(col_ax.get_yticks()))[0]
        y_ticks = np.arange(0, size_y, y_tick_spacing)
        col_ax.set_xticks(x_ticks)
        col_ax.set_yticks(y_ticks)

        # Create tick labels.
        if view_crop_box is not None:
            x_ticks = x_ticks + view_crop_box[0, 0]
            y_ticks = y_ticks + view_crop_box[0, 1]
        if not use_image_coords and affine is not None:
            s = affine_spacing(affine)
            o = affine_origin(affine)
            sx, sy = __get_view_xy(v, s)
            ox, oy = __get_view_xy(v, o)
            x_ticks = x_ticks * sx + ox
            y_ticks = y_ticks * sy + oy
        col_ax.set_xticklabels([f'{t:.1f}' for t in x_ticks])
        col_ax.set_yticklabels([f'{t:.1f}' for t in y_ticks])

        # Title.
        if show_title:
            title = f'{VIEWS[v]}, slice {resolved_idx}'
            if affine is not None:
                s = affine_spacing(affine)
                o = affine_origin(affine)
                world_pos = resolved_idx * s[v] + o[v]
                title += f' ({world_pos:.1f}mm)'
            col_ax.set_title(title, fontsize=title_fontsize)

        # Hide spines.
        for p in ['right', 'top', 'bottom', 'left']:
            col_ax.spines[p].set_visible(False)

    if show:
        plt.tight_layout()
        plt.show()

    if return_axis:
        return axs

def __resolve_boxes(
    box: np.ndarray | str | List[str] | None,
    size: tuple,
    affine: AffineMatrix | None = None,
    labels: BatchLabelImage | None = None,
    label_names: List[str] | None = None,
    points: List[np.ndarray] | None = None,
    point_names: List[List[str]] | None = None,
    ) -> np.ndarray | None:
    if box is None:
        return None

    # Resolve foreground fov for label name strings.
    if isinstance(box, (str, list)):
        if labels is None:
            raise ValueError("If box is a label name, labels must be provided.")
        label_names_list = arg_to_list(label_names, str) if label_names is not None else None
        region_ids = arg_to_list(box, str)
        boxes = []
        for r in region_ids:
            if label_names_list is None:
                raise ValueError("label_names must be provided for string box region IDs.")
            if r not in label_names_list:
                raise ValueError(f"Region '{r}' not found in label_names: {label_names_list}.")
            idx = label_names_list.index(r)
            fov_b = foreground_fov(labels[idx])
            if fov_b is None:
                continue
            boxes.append(to_numpy(fov_b))
        if not boxes:
            return None
        return np.stack(boxes)

    boxes = np.array(box) if not isinstance(box, np.ndarray) else box.copy()

    # Convert single box to batch.
    if boxes.ndim == 2:
        boxes = boxes[np.newaxis]

    # Convert to image coords.
    if affine is not None:
        for i in range(len(boxes)):
            boxes[i] = to_image_coords(boxes[i], affine)
            boxes[i] = np.clip(boxes[i], 0, np.array(size) - 1)

    return boxes

def __resolve_crop(
    crop: np.ndarray | Point2D | Point3D | str | None,
    crop_margin: int,
    size: tuple,
    affine: AffineMatrix | None = None,
    labels: BatchLabelImage | None = None,
    label_names: List[str] | None = None,
    points: List[np.ndarray] | None = None,
    point_names: List[List[str]] | None = None,
    ) -> np.ndarray | None:
    if crop is None:
        return None

    size = np.array(size)
    dim = len(size)

    if affine is not None:
        spacing = affine_spacing(affine)
        margin_vox = np.full(dim, crop_margin, dtype=np.float32) / spacing
    else:
        margin_vox = np.full(dim, crop_margin, dtype=np.float32)

    apply_margin = True

    if isinstance(crop, str):
        source, value = crop.split(':', 1)

        if source in ('l', 'label', 'labels'):
            if labels is None:
                raise ValueError(f"crop='{crop}' but no labels were provided.")
            if value.lstrip('-').isdigit():
                label_idx = int(value)
            else:
                if label_names is None:
                    raise ValueError(f"crop='{crop}' uses a label name but no 'label_names' were provided.")
                label_names_list = arg_to_list(label_names, str)
                if value not in label_names_list:
                    raise ValueError(f"Label name '{value}' not found in label_names: {label_names_list}.")
                label_idx = label_names_list.index(value)
            fov_box = foreground_fov(labels[label_idx])
            if fov_box is None:
                return None
            box_vox = to_numpy(fov_box).astype(np.float32)

        elif source in ('p', 'point', 'points'):
            if points is None:
                raise ValueError(f"crop='{crop}' but no points were provided.")
            value_parts = value.split(':')
            if len(value_parts) == 1:
                batch_idx, point_ref = 0, value_parts[0]
            else:
                batch_idx, point_ref = int(value_parts[0]), value_parts[1]
            batch = points[batch_idx]
            if point_ref.lstrip('-').isdigit():
                point_vox = batch[int(point_ref)]
            else:
                batch_names = point_names[batch_idx] if point_names is not None else None
                if batch_names is None:
                    raise ValueError(f"crop='{crop}' uses a point name but no 'point_names' were provided.")
                if point_ref not in batch_names:
                    raise ValueError(f"Point name '{point_ref}' not found in point_names: {batch_names}.")
                point_vox = batch[batch_names.index(point_ref)]
            box_vox = np.stack([point_vox, point_vox])

        else:
            raise ValueError(f"Unknown crop prefix '{source}'. Expected 'l'/'label'/'labels' or 'p'/'point'/'points'.")

    elif isinstance(crop, np.ndarray) and crop.ndim == 2:
        # Box - crop directly, no margin applied.
        if affine is not None:
            origin = affine_origin(affine)
            spacing = affine_spacing(affine)
            box_vox = (crop - origin) / spacing
        else:
            box_vox = crop.astype(np.float32)
        apply_margin = False

    else:
        # Point - create a box of crop_margin around the point.
        point = to_numpy(crop).astype(np.float32)
        if affine is not None:
            origin = affine_origin(affine)
            spacing = affine_spacing(affine)
            point_vox = (point - origin) / spacing
        else:
            point_vox = point
        box_vox = np.stack([point_vox, point_vox])

    if apply_margin:
        box_vox[0] -= margin_vox
        box_vox[1] += margin_vox

    # Clip to data bounds and convert to int.
    box_vox = np.stack([
        np.clip(np.floor(box_vox[0]).astype(np.int32), 0, size - 1),
        np.clip(np.ceil(box_vox[1]).astype(np.int32), 0, size - 1),
    ])
    return box_vox

def __resolve_crosshairs(
    crosshairs: np.ndarray | Point3D | str | None,
    size: tuple,
    affine: AffineMatrix | None = None,
    centre_method: Literal['com', 'fov'] = 'com',
    labels: BatchLabelImage | None = None,
    label_names: List[str] | None = None,
    points: List[np.ndarray] | None = None,
    point_names: List[List[str]] | None = None,
    ) -> np.ndarray | None:
    if crosshairs is None:
        return None
    size = np.array(size)
    if isinstance(crosshairs, str):
        source, value = crosshairs.split(':', 1)
        if source in ('l', 'label', 'labels'):
            if labels is None:
                raise ValueError(f"crosshairs='{crosshairs}' but no labels were provided.")
            if value.lstrip('-').isdigit():
                label_idx = int(value)
            else:
                if label_names is None:
                    raise ValueError(f"crosshairs='{crosshairs}' uses a label name but no 'label_names' were provided.")
                label_names_list = arg_to_list(label_names, str)
                if value not in label_names_list:
                    raise ValueError(f"Label name '{value}' not found in label_names: {label_names_list}.")
                label_idx = label_names_list.index(value)
            if centre_method == 'com':
                point = centre_of_mass(labels[label_idx], affine=affine)
            else:
                point = foreground_fov_centre(labels[label_idx], affine=affine)
            if affine is not None:
                point = to_image_coords(point, affine)
        elif source in ('p', 'point', 'points'):
            if points is None:
                raise ValueError(f"crosshairs='{crosshairs}' but no points were provided.")
            value_parts = value.split(':')
            if len(value_parts) == 1:
                batch_idx, point_ref = 0, value_parts[0]
            else:
                batch_idx, point_ref = int(value_parts[0]), value_parts[1]
            batch = points[batch_idx]
            if point_ref.lstrip('-').isdigit():
                point = batch[int(point_ref)]
            else:
                batch_names = point_names[batch_idx] if point_names is not None else None
                if batch_names is None:
                    raise ValueError(f"crosshairs='{crosshairs}' uses a point name but no 'point_names' were provided.")
                if point_ref not in batch_names:
                    raise ValueError(f"Point name '{point_ref}' not found in point_names: {batch_names}.")
                point = batch[batch_names.index(point_ref)]
        else:
            raise ValueError(f"Unknown crosshairs prefix '{source}'. Expected 'l'/'label'/'labels' or 'p'/'point'/'points'.")
    else:
        point = to_numpy(crosshairs).astype(float)
        if affine is not None:
            point = to_image_coords(point, affine)
    return np.clip(np.round(point), 0, size - 1).astype(int)

def __resolve_labels(
    labels: LabelImage | BatchLabelImage | None,
    dim: int,
    ) -> BatchLabelImage | None:
    if labels is None:
        return None
    labels = to_numpy(labels)
    # Normalise to batch form (B, ...).
    if labels.ndim == dim:
        labels = labels[np.newaxis]
    if labels.dtype != bool:
        if labels.min() < 0 or labels.max() > 1:
            logger.warn(f"Labels values are outside the range [0, 1]. Got min={labels.min():.3f}, max={labels.max():.3f}. Performing minmax norm.")
            labels_min, labels_max = labels.min(), labels.max()
            if labels_max > labels_min:
                labels = (labels - labels_min) / (labels_max - labels_min)
    return labels

def __resolve_points(
    points: Point | Points | BatchPoints | List[Points] | None,
    affine: AffineMatrix | None = None,
    point_names: str | List[str] | List[List[str]] | None = None,
    # Returns a list of points (instead of a BatchPoints array) because this allows different batch sizes.
    ) -> Tuple[List[Points] | None, List[List[str]] | None]:
    if points is None:
        return None, None

    # Normalise points to List[Points].
    if not isinstance(points, list):
        points = to_numpy(points)
        # Normalise to (B, N, dim): single point (dim,) → (1, 1, dim), unbatched (N, dim) → (1, N, dim).
        if points.ndim == 1:
            points = [points[np.newaxis]]
        elif points.ndim == 2:
            points = [points]
        elif points.ndim == 3:
            points = [p for p in points]

    # Convert each batch to numpy and normalise shape: (dim,) → (1, dim).
    points = [to_numpy(p) for p in points]
    points = [p[np.newaxis] if p.ndim == 1 else p for p in points]

    # Convert from world to image coords.
    if affine is not None:
        spacing = affine_spacing(affine)
        origin = affine_origin(affine)
        points = [(p - origin) / spacing for p in points]

    # Resolve point names per batch. If provided, applied uniformly
    # (each batch must have the same length).
    if point_names is not None:
        # Convert to List[List[str]].
        if isinstance(point_names, str):    # point_names = str
            point_names = [[point_names]] * len(points)
        elif isinstance(point_names, list):
            if isinstance(point_names[0], str):   # point_names = List[str]
                # Broadast to all points.
                point_names = [point_names] * len(points)
            elif isinstance(point_names[0], list):  # point_names = List[List[str]]
                pass

        # Check that point and point names have the same number of elements.
        for p, n in zip(points, point_names):
            if len(p) != len(n):
                raise ValueError(f"Number of points ({len(p)}) and point names ({len(n)}) must match for each batch.")

    return points, point_names

def __resolve_window(
    window: Window | None,
    vmin: float | None = None,
    vmax: float | None = None,
    affine: AffineMatrix | None = None,
    data: Image | None = None,
    labels: BatchLabelImage | None = None,
    label_names: List[str] | None = None,
    label_margin: float = 100,
    ) -> tuple[float | None, float | None]:
    if window is not None:
        assert vmin is None, "vmin must be None if window is specified."
        assert vmax is None, "vmax must be None if window is specified."
    if window is None:
        return vmin, vmax
    if isinstance(window, str):
        if window in WINDOW_PRESETS:
            width, level = WINDOW_PRESETS[window]
        else:
            source, value = window.split(':', 1)
            if source in ('l', 'label', 'labels'):
                if labels is None:
                    raise ValueError(f"window='{window}' but no labels were provided.")
                if value.lstrip('-').isdigit():
                    label_idx = int(value)
                else:
                    if label_names is None:
                        raise ValueError(f"window='{window}' uses a label name but no 'label_names' were provided.")
                    label_names_list = arg_to_list(label_names, str)
                    if value not in label_names_list:
                        raise ValueError(f"Label name '{value}' not found in label_names: {label_names_list}.")
                    label_idx = label_names_list.index(value)
                fov_box = foreground_fov(labels[label_idx])
                if fov_box is None:
                    return vmin, vmax
                fov_box = to_numpy(fov_box)
                margin_vox = (np.full(data.ndim, label_margin) / affine_spacing(affine)).astype(int) if affine is not None else np.full(data.ndim, label_margin, dtype=int)
                slices = tuple(slice(max(0, int(fov_box[0, i]) - margin_vox[i]), min(data.shape[i], int(fov_box[1, i]) + margin_vox[i] + 1)) for i in range(data.ndim))
                fov_data = data[slices]
                width = float(fov_data.max() - fov_data.min())
                level = float((fov_data.max() + fov_data.min()) / 2)
            else:
                raise ValueError(f"Unknown window preset '{window}'. Expected one of {list(WINDOW_PRESETS.keys())}, or 'l:<idx>'.")
    else:
        width, level = window
    vmin = level - width / 2
    vmax = level + width / 2
    return vmin, vmax
