from augmed.typing import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import *

from mymi.typing import *
from mymi.utils import *

from .plotting import get_view_aspect, get_idx, get_view_origin, get_view_slice, get_view_xy, get_v_min_max, plot_landmarks_data

def plot_slice(
    data: Image2D,
    labels: ChannelLabelImage2D | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    ) -> None:
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Plot slice.
    plt.imshow(data.T, cmap='gray')

    # Plot labels.
    if labels is not None:
        palette = sns.color_palette('colorblind', len(labels))
        for i, l in enumerate(labels):
            cmap = mpl.colors.ListedColormap(((1, 1, 1, 0), palette[i]))
            plt.imshow(l.T, alpha=0.3, cmap=cmap)
            plt.contour(l.T, colors=[palette[i]], levels=[.5], linestyles='solid')

    # Hide axis spines.
    ax = plt.gca()
    for p in ['right', 'top', 'bottom', 'left']:
        ax.spines[p].set_visible(False)

    # Add text.
    if title is not None:
        ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    plt.show()

@alias_kwargs(('uwc', 'use_world_coords'))
def plot_volume(
    data: Union[ImageArray, ImageTensor, DirPath, FilePath, List[Union[ImageArray, ImageTensor]]],
    affine: Affine | None = None,
    centre: Optional[Union[LandmarkID, LandmarkSeries, Literal['dose'], RegionArray, RegionID]] = None,
    centre_other: Optional[Union[LandmarkID, LandmarkSeries, Literal['dose'], RegionArray, RegionID]] = None,
    dose_alpha_min: float = 0.3,
    dose_alpha_max: float = 1.0,
    dose_cmap: str = 'turbo',
    dose_cmap_trunc: float = 0.15,
    dose_affine: Affine | None = None,
    dose_data: Optional[Union[ImageArray, ImageTensor, DirPath, FilePath]] = None,
    figsize: Tuple[float, float] = (16, 6),
    idx: Union[int, float, List[Union[int, float]]] = 0.5,
    # If single or list, broadcast to all images. If list of lists, leave alone.
    labels: FilePath | LabelVolume | LabelVolumeBatch | None = None,
    landmark: LandmarkIDs = 'all',
    landmarks_data: Optional[Union[LandmarksFrame, PointsArray, PointsTensor, List[Union[LandmarksFrame, PointsArray, PointsTensor]]]] = None,    # Should be in patient coordinates.
    modality: Literal['ct', 'dose'] = 'ct',
    # Our plotting follows standard radiological convention and assumes input data is in LPS+ orientation.
    orientation: str = 'LPS',
    other_landmarks_data: Optional[Union[LandmarksFrame, PointsArray, PointsTensor, List[Union[LandmarksFrame, PointsArray, PointsTensor]]]] = None,    # Should be in patient coordinates.
    show_axis_ticks: bool = True,
    show_axis_tick_labels: bool = True,
    show_dose: bool = True,
    show_landmarks: bool = True,    # Can pass 'landmarks_data' for 'centre' but don't want plotted.
    show_title: bool = True,
    transpose: bool = False,
    use_world_coords: bool = True,
    view: Union[int, List[int]] = 'all',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    window: Optional[Union[str, Tuple[float, float]]] = 'tissue',
    **kwargs) -> None:
    if isinstance(data, (DirPath, FilePath)):
        if os.path.isdir(data):
            data, affine = from_ct_dicom(data)
        elif data.endswith('.nii') or data.endswith('.nii.gz'):
            data, affine = load_nifti(data)
        elif data.endswith('.npy'):
            data, affine = load_numpy(data)
        elif data.endswith('.nrrd'):
            data, affine = load_nrrd(data)
        elif data.endswith('.mha') or data.endswith('.mhd'):
            data, affine = sitk_load_volume(data)
        else:
            raise ValueError(f'Unsupported file type: {data}')
    data = arg_to_list(data, (np.ndarray, torch.Tensor))
    dose_data = arg_to_list(dose_data, (None, np.ndarray, torch.Tensor))
    affines = arg_to_list(affine, (None, np.ndarray), broadcast=len(data))
    idxs = arg_to_list(idx, (int, float, str), broadcast=len(data))
    centres = arg_to_list(centre, (None, int, float, str), broadcast=len(data))
    centre_others = arg_to_list(centre_other, (None, int, float, str), broadcast=len(data))
    if isinstance(labels, str) and os.path.isfile(labels):
        if labels.endswith('.nii') or labels.endswith('.nii.gz'):
            labels, _ = load_nifti(labels)
    labels = arg_to_list(labels, (None, np.ndarray, torch.Tensor))   # From single element to list.
    landmarks_datas = arg_to_list(landmarks_data, (None, pd.DataFrame, np.ndarray, torch.Tensor), broadcast=len(data))
    show_doses = arg_to_list(show_dose, (bool), broadcast=len(data))
    show_landmarkses = arg_to_list(show_landmarks, (bool), broadcast=len(data))
    other_landmarks_datas = arg_to_list(other_landmarks_data, (None, pd.DataFrame, np.ndarray, torch.Tensor), broadcast=len(data))
    assert len(landmarks_datas) == len(data)
    assert len(other_landmarks_datas) == len(data)
    assert len(affines) == len(data)
    palette = sns.color_palette('colorblind', 20)
    views = arg_to_list(view, int, literals={ 'all': list(range(3)) })

    # Convert tensors.
    # TODO: Ensure np data is passed - too hard to handle tensors also.
    for i in range(len(data)):
        if isinstance(data[i], torch.Tensor):
            data[i] = data[i].cpu().numpy()
    for i in range(len(landmarks_datas)):
        if isinstance(landmarks_datas[i], torch.Tensor):
            landmarks_datas[i] = landmarks_datas[i].cpu().numpy()
    for i in range(len(other_landmarks_datas)):
        if isinstance(other_landmarks_datas[i], torch.Tensor):
            other_landmarks_datas[i] = other_landmarks_datas[i].cpu().numpy()
    for i in range(len(affines)):
        if isinstance(affines[i], torch.Tensor):
            affines[i] = affines[i].cpu().numpy()

    # Plot images.
    n_rows, n_cols = (len(views), len(data)) if transpose else (len(data), len(views))
    _, axs = plt.subplots(n_rows, n_cols, figsize=(figsize[0], len(data) * figsize[1]), squeeze=False)
    if transpose:
        axs = axs.T

    for i, (row_axs, d, aff, dd, idx, c, oc, l, lms, olms, sd, sl) in enumerate(zip(axs, data, affines, dose_data, idxs, centres, centre_others, labels, landmarks_datas, other_landmarks_datas, show_doses, show_landmarkses)):
        logging.info(f"Plotting image {i+1}/{len(data)}: with size={d.shape}, idx={idx}, centre={c}, affine={aff}.")

        # Rescale RGB image to range [0, 1).
        n_dims = len(d.shape)
        if n_dims == 4:
            d = (d - d.min()) / (d.max() - d.min())

        for col_ax, v in zip(row_axs, views):
            resolved_idx = get_idx(d.shape, v, affine=aff, centre=c, centre_other=oc, idx=idx, landmarks_data=lms, landmarks_data_other=olms, label_data=l)
            image, resolved_idx = get_view_slice(v, d, resolved_idx)
            aspect = get_view_aspect(v, affine)
            origin = get_view_origin(v, orientation=orientation)
            vmin, vmax = get_v_min_max(data=d, vmin=vmin, vmax=vmax, window=window)
            if modality == 'ct':
                cmap='gray'
            elif modality == 'dose':
                cmap='viridis'
            col_ax.imshow(image, aspect=aspect, cmap=cmap, origin=origin[1], vmin=vmin, vmax=vmax)
            if origin[0] == 'upper':
                col_ax.invert_xaxis()
            if show_title:
                col_ax.set_title(f'{get_axis_name(v)} view, slice {resolved_idx}')
            if not show_axis_ticks:
                col_ax.set_xticks([])
                col_ax.set_yticks([])
            if not show_axis_tick_labels:
                col_ax.set_xticklabels([])
                col_ax.set_yticklabels([])

            # Plot dose.
            if sd and dd is not None:
                # Resample dose to image grid if necessary.
                if (ds is not None and ds != s) or (do is not None and do != o):
                    rs_params = dict(
                        origin=do if do is not None else o,
                        output_origin=o,
                        output_size=d.shape,
                        output_spacing=s,
                        spacing=ds if ds is not None else s,
                    )
                    dd = resample(dd, **rs_params)

                dose_image, resolved_idx = get_view_slice(v, dd, resolved_idx)

                # Create colormap with varying alpha - so 0 Gray is transparent.
                dose_cmap = plt.get_cmap(dose_cmap)

                # Remove purple from the colourmap.
                dose_cmap = mpl.colors.LinearSegmentedColormap.from_list(
                    f'{dose_cmap.name}_truncated',
                    dose_cmap(np.linspace(dose_cmap_trunc, 1.0, 256))
                )

                dose_colours = dose_cmap(np.arange(dose_cmap.N))
                dose_colours[0, -1] = 0
                dose_colours[1:, -1] = np.linspace(dose_alpha_min, dose_alpha_max, dose_cmap.N - 1)
                dose_cmap = mpl.colors.ListedColormap(dose_colours)

                # Plot dose slice.
                if show_dose:
                    imax = col_ax.imshow(dose_image, aspect=aspect, cmap=dose_cmap, origin=origin)
                    # if show_dose_legend:
                    #     cbar = plt.colorbar(imax, fraction=dose_colourbar_size, pad=dose_colourbar_pad)
                    #     cbar.set_label(label='Dose [Gray]', size=fontsize)
                    #     cbar.ax.tick_params(labelsize=fontsize)

            # Plot regions.
            if l is not None:
                for j, li in enumerate(l):
                    cmap = mpl.colors.ListedColormap(((1, 1, 1, 0), palette[j]))
                    assert len(li.shape) == 3
                    region_slice, _ = get_view_slice(v, li, resolved_idx)
                    col_ax.imshow(region_slice, alpha=0.3, aspect=aspect, cmap=cmap, origin=origin[1])
                    col_ax.contour(region_slice, colors=[palette[j]], levels=[.5], linestyles='solid')

            if use_world_coords:  # Change axis tick labels to show patient coordinates.
                size_x, size_y = get_view_xy(v, d.shape)
                s = affine_spacing(aff)
                o = affine_origin(aff)
                sx, sy = get_view_xy(v, s)
                ox, oy = get_view_xy(v, o)

                if show_axis_ticks:
                    x_tick_spacing = np.unique(np.diff(col_ax.get_xticks()))[0]
                    x_ticks = np.arange(0, size_x, x_tick_spacing)
                    col_ax.set_xticks(x_ticks)
                    y_tick_spacing = np.unique(np.diff(col_ax.get_yticks()))[0]
                    y_ticks = np.arange(0, size_y, y_tick_spacing)
                    col_ax.set_yticks(y_ticks)

                    if show_axis_tick_labels:
                        x_ticklabels = x_ticks * sx + ox
                        x_ticklabels = [f'{l:.1f}' for l in x_ticklabels]
                        col_ax.set_xticklabels(x_ticklabels)
                        y_ticklabels = y_ticks * sy + oy
                        y_ticklabels = [f'{l:.1f}' for l in y_ticklabels]
                        col_ax.set_yticks(y_ticks if show_axis_ticks else [])
                        col_ax.set_yticklabels(y_ticklabels)

                view_loc = resolved_idx * s[v] + o[v]
                if show_title:
                    col_ax.set_title(f'{get_axis_name(v)} view, slice {resolved_idx} ({view_loc:.1f}mm)')
            if sl and lms is not None:
                plot_landmarks_data(lms, col_ax, resolved_idx, d.shape, aff, v, landmark=landmark, **kwargs)
            if sl and olms is not None:
                plot_landmarks_data(olms, col_ax, resolved_idx, d.shape, aff, v, landmark=landmark, marker_colour='red', **kwargs)

@delegates(plot_volume)
def plot_nifti(
    filepath: str,
    spacing: Optional[Spacing3D] = None,
    origin: Optional[Point3D] = None,
    **kwargs) -> None:
    data, nspacing, norigin = load_nifti(filepath)
    spacing = nspacing if spacing is None else spacing
    origin = norigin if origin is None else origin
    plot_volume(data, origin=origin, spacing=spacing, **kwargs)

@delegates(load_numpy, plot_volume)
def plot_numpy(
    filepath: str,
    spacing: Optional[Spacing3D] = (1, 1, 1),
    origin: Optional[Point3D] = (0, 0, 0),
    **kwargs) -> None:
    data = load_numpy(filepath, **kwargs)
    plot_volume(data, origin=origin, spacing=spacing, **kwargs)

def sitk_plot_volume(
    filepath: str,
    spacing: Optional[Spacing3D] = None,
    origin: Optional[Point3D] = None,
    **kwargs) -> None:
    data, lspacing, lorigin = sitk_load_volume(filepath)
    if spacing is None:
        spacing = lspacing
    if origin is None:
        origin = lorigin
    plot_volume(data, origin=origin, spacing=spacing, **kwargs)
