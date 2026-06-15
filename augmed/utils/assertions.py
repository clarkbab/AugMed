import numpy as np
import torch
from typing import List, Tuple

from ..typing import Image, ImagesInput, Number, Orientation, Orientation2D, Orientation3D, Points, Size, SpatialDim
from .args import arg_to_list

def assert_dim(
    dim: SpatialDim,
    ) -> None:
    if dim not in (2, 3):
        raise ValueError(f"Invalid dim '{dim}'. Spatial dimensions must be 2 or 3.")

def assert_image_shapes(
    images: ImagesInput,
    dim: SpatialDim,
    batch_allowed: bool = True,
    channel_allowed: bool = True,
    suggest_dim: bool = True,
    ) -> None:
    images = arg_to_list(images, (np.ndarray, torch.Tensor))
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
            message += f", got {n_dims} dimensions - shape={img.shape} for image {i}."
            if suggest_dim:
                message += f" Set 'dim' param if {dim}D spatial is not correct."
            raise ValueError(message)

def assert_image_sizes(
    images: ImagesInput,
    dim: SpatialDim,
    ) -> None:
    # Avoid circular import.
    def __spatial_size(image: Image, dim: SpatialDim) -> Size:
        return image.shape[-dim:]
    images = arg_to_list(images, (np.ndarray, torch.Tensor))
    size = __spatial_size(images[0], dim)
    for i, img in enumerate(images):
        if __spatial_size(img, dim) != size:
            raise ValueError(f"All images must have the same spatial size. Expected {tuple(size)}, got {tuple(spatial_size(img, dim))} for image {i}.")

def assert_orientation(
    orientation: Orientation,
    dim: SpatialDim,
    ) -> None:
    if dim == 2:
        __assert_orientation_2d(orientation)
    elif dim == 3:
        __assert_orientation_3d(orientation)

def __assert_orientation_2d(
    orientation: Orientation2D,
    ) -> None:
    orientations = {'LI', 'LS', 'RI', 'RS'}
    if orientation not in orientations:
        raise ValueError(f"Invalid orientation '{orientation}' for dim=2. Must be one of {orientations}.")

def __assert_orientation_3d(
    orientation: Orientation3D,
    ) -> None:
    orientations = {'LAI', 'LAS', 'LPI', 'LPS', 'RAI', 'RAS', 'RPI', 'RPS'}
    if orientation not in orientations:
        raise ValueError(f"Invalid orientation '{orientation}' for dim=3. Must be one of {orientations}.")

def assert_points_shapes(
    # TODO: Allow BatchPoints - this is a big change throughout the code.
    points: Points | List[Points],
    dim: SpatialDim,
    ) -> None:
    points = arg_to_list(points, (np.ndarray, torch.Tensor))
    for i, p in enumerate(points):
        if p.ndim != 2:
            raise ValueError(f"Expected points to have 2 dimensions - shape=(N, {dim}), got {p.ndim} dimensions - shape={p.shape} for points {i}.")
        if p.shape[-1] != dim:
            raise ValueError(f"Expected points to be {dim}D spatial, got {p.shape[-1]}D - shape={p.shape} for points {i}. Set 'dim' param if {dim}D spatial is not correct.")

def assert_range(
    arg: Tuple[Number],
    dim: SpatialDim,
    name: str,
    vals_per_dim: int = 2,
    ) -> None:
    expected_length = dim * vals_per_dim
    if len(arg) != expected_length:
        raise ValueError(f"Got '{name}' of length {len(arg)}, expected length {expected_length} for {dim}D spatial. Set 'dim' param if {dim}D spatial is not correct.")
    