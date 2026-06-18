import numpy as np
import torch
from typing import List, Tuple

from ..typing import Image, ImagesInput, Number, Orientation, Orientation2D, Orientation3D, Points, Size, SpatialDim
from .args import arg_to_list

def assert_dim(
    dim: int,
    caller: str | None = None,
    ) -> None:
    if dim not in (2, 3):
        call_str = f' (called by {caller})' if caller is not None else ''
        raise ValueError(f"Invalid dim '{dim}'{call_str}. Spatial dimensions must be 2 or 3.")

def assert_image_shapes(
    images: ImagesInput,
    dim: SpatialDim,
    batch_allowed: bool = True,
    channel_allowed: bool = True,
    suggest_dim: bool = True,
    caller: str | None = None,
    ) -> None:
    images = arg_to_list(images, (np.ndarray, torch.Tensor))
    call_str = f' (called by {caller})' if caller is not None else ''
    for i, img in enumerate(images):
        n_dims = len(img.shape)
        n_bc_dims = int(batch_allowed) + int(channel_allowed)
        possible_dims = list(range(dim, dim + n_bc_dims + 1))   # E.g. for 3D, possible dims are 3-5 (3D spatial, optional batch/channel).
        if n_dims not in possible_dims:
            if len(possible_dims) == 1:
                expected_dims_str = f"{possible_dims[0]}"
            else:
                expected_dims_str = f"{possible_dims[0]}-{possible_dims[-1]}"
            message = f"Expected {expected_dims_str} image dimensions ({dim}D spatial"
            if batch_allowed and channel_allowed:
                message += ", optional batch/channel)"
            elif batch_allowed:
                message += ", optional batch)"
            elif channel_allowed:
                message += ", optional channel)"
            message += f", got {n_dims} dimensions - shape={img.shape} for image {i}{call_str}."
            if suggest_dim:
                message += f" Set 'dim' param if {dim}D spatial is not correct."
            raise ValueError(message)

def assert_dist(
    dist: str,
    caller: str | None = None,
    ) -> None:
    if dist not in ['gaussian', 'uniform']:
        call_str = f' (called by {caller})' if caller is not None else ''
        raise ValueError(f"Distribution '{dist}' not recognised{call_str}.")

def assert_dist_std(
    dist_std,
    caller: str | None = None,
    ) -> None:
    if not isinstance(dist_std, (int, float)) or dist_std <= 0:
        call_str = f' (called by {caller})' if caller is not None else ''
        raise ValueError(f"dist_std must be a positive number{call_str}, got '{dist_std}'.")

def assert_p(
    p,
    caller: str | None = None,
    ) -> None:
    if not isinstance(p, (int, float)) or not (0 <= p <= 1):
        call_str = f' (called by {caller})' if caller is not None else ''
        raise ValueError(f"p must be a number between 0 and 1{call_str}, got '{p}'.")

def assert_fill(
    fill,
    caller: str | None = None,
    ) -> None:
    valid = {'border', 'max', 'min', 'reflection', 'zeros'}
    if not isinstance(fill, (int, float)) and fill not in valid:
        call_str = f' (called by {caller})' if caller is not None else ''
        raise ValueError(f"Fill value '{fill}' not recognised{call_str}. Must be a number or one of {valid}.")

def assert_filter_offgrid(
    filter_offgrid,
    caller: str | None = None,
    ) -> None:
    valid_axes = {0, 1, 2}
    call_str = f' (called by {caller})' if caller is not None else ''
    axes = filter_offgrid if isinstance(filter_offgrid, list) else [filter_offgrid]
    for a in axes:
        if not isinstance(a, bool) and a not in valid_axes:
            raise ValueError(f"filter_offgrid value '{a}' not recognised{call_str}. Must be a bool or axis in {valid_axes}.")

def assert_seed(
    seed,
    caller: str | None = None,
    ) -> None:
    if seed is not None and not isinstance(seed, int):
        call_str = f' (called by {caller})' if caller is not None else ''
        raise ValueError(f"Seed must be an integer or None{call_str}, got {type(seed).__name__} '{seed}'.")

def assert_interpolation(
    interpolation: str,
    caller: str | None = None,
    ) -> None:
    valid = {'bicubic', 'bilinear', 'nearest'}
    if interpolation not in valid:
        call_str = f' (called by {caller})' if caller is not None else ''
        raise ValueError(f"Interpolation '{interpolation}' not recognised{call_str}. Must be one of {valid}.")

def assert_image_sizes(
    images: ImagesInput,
    dim: SpatialDim,
    caller: str | None = None,
    ) -> None:
    # Avoid circular import.
    def __spatial_size(image: Image, dim: SpatialDim) -> Size:
        return image.shape[-dim:]
    images = arg_to_list(images, (np.ndarray, torch.Tensor))
    call_str = f' (called by {caller})' if caller is not None else ''
    size = __spatial_size(images[0], dim)
    for i, img in enumerate(images):
        if __spatial_size(img, dim) != size:
            raise ValueError(f"All images must have the same spatial size{call_str}. Expected {size}, got {__spatial_size(img, dim)} for image {i}.")

def assert_orientation(
    orientation: Orientation,
    dim: SpatialDim,
    caller: str | None = None,
    ) -> None:
    if dim == 2:
        __assert_orientation_2d(orientation, caller=caller)
    elif dim == 3:
        __assert_orientation_3d(orientation, caller=caller)

def __assert_orientation_2d(
    orientation: Orientation2D,
    caller: str | None = None,
    ) -> None:
    orientations = {'LI', 'LS', 'RI', 'RS'}
    if orientation not in orientations:
        call_str = f' (called by {caller})' if caller is not None else ''
        raise ValueError(f"Invalid orientation '{orientation}' for dim=2{call_str}. Must be one of {orientations}.")

def __assert_orientation_3d(
    orientation: Orientation3D,
    caller: str | None = None,
    ) -> None:
    orientations = {'LAI', 'LAS', 'LPI', 'LPS', 'RAI', 'RAS', 'RPI', 'RPS'}
    if orientation not in orientations:
        call_str = f' (called by {caller})' if caller is not None else ''
        raise ValueError(f"Invalid orientation '{orientation}' for dim=3{call_str}. Must be one of {orientations}.")

def assert_points_shapes(
    # TODO: Allow BatchPoints - this is a big change throughout the code.
    points: Points | List[Points],
    dim: SpatialDim,
    caller: str | None = None,
    ) -> None:
    points = arg_to_list(points, (np.ndarray, torch.Tensor))
    call_str = f' (called by {caller})' if caller is not None else ''
    for i, p in enumerate(points):
        if p.ndim != 2:
            raise ValueError(f"Expected points to have 2 dimensions - shape=(N, {dim}), got {p.ndim} dimensions - shape={p.shape} for points {i}{call_str}.")
        if p.shape[-1] != dim:
            raise ValueError(f"Expected points to be {dim}D spatial, got {p.shape[-1]}D - shape={p.shape} for points {i}{call_str}. Set 'dim' param if {dim}D spatial is not correct.")

def assert_range(
    arg: Tuple[Number],
    dim: SpatialDim,
    name: str,
    vals_per_dim: int = 2,
    caller: str | None = None,
    ) -> None:
    expected_length = dim * vals_per_dim
    call_str = f' (called by {caller})' if caller is not None else ''
    if len(arg) != expected_length:
        raise ValueError(f"Got '{name}' of length {len(arg)}, expected length {expected_length} for {dim}D spatial{call_str}. Set 'dim' param if {dim}D spatial is not correct.")
