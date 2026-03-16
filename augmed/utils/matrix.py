import numpy as np
import torch
from typing import *

from ..typing import *
from .args import arg_to_list
from .conversion import to_numpy, to_tensor

def affine_origin(
    affine: AffineTensor,
    ) -> PointTensor:
    affine, return_type = to_tensor(affine, return_type=True)

    # Get origin.
    dim = affine.shape[0] - 1
    if dim == 2:
        origin = to_tensor([affine[0, 2], affine[1, 2]], device=affine.device, dtype=affine.dtype)
    else:
        origin = to_tensor([affine[0, 3], affine[1, 3], affine[2, 3]], device=affine.device, dtype=affine.dtype)

    # Change return type if needed.
    if return_type is np.ndarray:
        origin = to_numpy(origin)   

    return origin

def affine_spacing(
    affine: Affine,
    ) -> Affine:
    affine, return_type = to_tensor(affine, return_type=True)

    # Get spacing.
    dim = affine.shape[0] - 1
    if dim == 2:
        spacing = to_tensor([affine[0, 0], affine[1, 1]], device=affine.device, dtype=affine.dtype)
    else:
        spacing = to_tensor([affine[0, 0], affine[1, 1], affine[2, 2]], device=affine.device, dtype=affine.dtype)

    # Change return type if needed.
    if return_type is np.ndarray:
        spacing = to_numpy(spacing)

    return spacing

# Might be public facing - allow more than just Tensor types.
def create_affine(
    spacing: Spacing,
    origin: Point,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
    ) -> AffineTensor:
    dim = len(spacing)
    affine = create_eye(dim, device=device, dtype=dtype)
    if dim == 2:
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1]
        affine[0, 2] = origin[0]
        affine[1, 2] = origin[1]
    else:
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1]
        affine[2, 2] = spacing[2]
        affine[0, 3] = origin[0]
        affine[1, 3] = origin[1]
        affine[2, 3] = origin[2]
    return affine

def create_eye(
    dim: SpatialDim,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
    return torch.eye(dim + 1, device=device, dtype=dtype)

def create_rotation(
    rotation: Number | Tuple[Number] | np.ndarray | torch.Tensor,   # In radians.
    device: torch.device = torch.device('cpu'),
    dim: SpatialDim | None = None,
    dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
    if dim is None:
        assert isinstance(rotation, (tuple, list, np.ndarray, torch.Tensor)), f"'rotation' must be a tuple, list, numpy array, or torch tensor when 'dim' is None. Got {type(rotation)}."
        dim = len(rotation)
    rotation = arg_to_list(rotation, (int, float), iter_types=(np.ndarray, torch.Tensor), broadcast=dim)
    rotation = to_tensor(rotation, device=device, dtype=dtype)
    if dim == 2:
        # 2D rotation matrix.
        matrix = to_tensor([
            [torch.cos(rotation[0]), -torch.sin(rotation[0]), 0],
            [torch.sin(rotation[0]), torch.cos(rotation[0]), 0],
            [0, 0, 1]
        ], device=device, dtype=dtype)
    else:
        # 3D rotation matrix.
        rotation_x = to_tensor([
            [1, 0, 0, 0],
            [0, torch.cos(rotation[0]), -torch.sin(rotation[0]), 0],
            [0, torch.sin(rotation[0]), torch.cos(rotation[0]), 0],
            [0, 0, 0, 1]
        ], device=device, dtype=dtype)
        rotation_y = to_tensor([
            [torch.cos(rotation[1]), 0, torch.sin(rotation[1]), 0],
            [0, 1, 0, 0],
            [-torch.sin(rotation[1]), 0, torch.cos(rotation[1]), 0],
            [0, 0, 0, 1]
        ], device=device, dtype=dtype)
        rotation_z = to_tensor([
            [torch.cos(rotation[2]), -torch.sin(rotation[2]), 0, 0],
            [torch.sin(rotation[2]), torch.cos(rotation[2]), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], device=device, dtype=dtype)
        # Perform using order: (x, y, z).
        matrix = torch.linalg.multi_dot([rotation_z, rotation_y, rotation_x])
    return matrix

def create_scaling(
    scaling: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor,
    device: torch.device = torch.device('cpu'),
    dim: SpatialDim | None = None,
    dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
    if dim is None:
        assert isinstance(scaling, (tuple, list, np.ndarray, torch.Tensor)), f"'scaling' must be a tuple, list, numpy array, or torch tensor when 'dim' is None. Got {type(scaling)}."
        dim = len(scaling)
    scaling = arg_to_list(scaling, (int, float), iter_types=(np.ndarray, torch.Tensor), broadcast=dim)
    scaling = to_tensor(scaling, device=device, dtype=dtype)
    matrix = create_eye(dim, device=device, dtype=dtype)
    for i, s in enumerate(scaling):
        matrix[i, i] = s
    return matrix

def create_translation(
    translation: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor,
    device: torch.device = torch.device('cpu'),
    dim: SpatialDim | None = None,
    dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
    if dim is None:
        assert isinstance(translation, (tuple, list, np.ndarray, torch.Tensor)), f"'translation' must be a tuple, list, numpy array, or torch tensor when 'dim' is None. Got {type(translation)}."
        dim = len(translation)
    translation = arg_to_list(translation, (int, float), iter_types=(np.ndarray, torch.Tensor), broadcast=dim)
    translation = to_tensor(translation, device=device, dtype=dtype)
    matrix = create_eye(dim, device=device, dtype=dtype)
    matrix[:dim, dim] = translation
    return matrix
