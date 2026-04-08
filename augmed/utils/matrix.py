import numpy as np
import torch
from typing import Tuple

from ..typing import Number, SpatialDim
from .args import arg_to_list
from .conversion import to_tensor
from .geometry import create_eye

def create_rotation(
    rotation: Number | Tuple[Number] | np.ndarray | torch.Tensor,   # In radians.
    device: torch.device = torch.device('cpu'),
    dim: SpatialDim | None = None,
    dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
    if dim is None:
        assert isinstance(rotation, (tuple, list, np.ndarray, torch.Tensor)), f"'rotation' must be a tuple, list, numpy array, or torch tensor when 'dim' is None. Got {type(rotation)}."
        dim = len(rotation)
    rotation = arg_to_list(rotation, (int, float), broadcast=dim, iter_types=(np.ndarray, torch.Tensor))
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
    scaling = arg_to_list(scaling, (int, float), broadcast=dim, iter_types=(np.ndarray, torch.Tensor))
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
    translation = arg_to_list(translation, (int, float), broadcast=dim, iter_types=(np.ndarray, torch.Tensor))
    translation = to_tensor(translation, device=device, dtype=dtype)
    matrix = create_eye(dim, device=device, dtype=dtype)
    matrix[:dim, dim] = translation
    return matrix
