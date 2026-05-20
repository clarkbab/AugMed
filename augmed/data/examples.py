from importlib.resources import files
import numpy as np
import os
from typing import Tuple

from ..typing import AffineMatrix, AffineMatrix3D, BatchLabelImage, BatchLabelImage3D, Image, Image3D, Number, Points3D, Size
from ..utils.args import alias_kwargs
from ..utils.conversion import to_numpy
from ..utils.geometry import affine_spacing, create_affine
from ..utils.io import load_nifti, load_numpy

@alias_kwargs(
    ('s', 'size'),
    ('a', 'affine'),
    ('ss', 'square_size'),
)
def load_checkerboard(
    size: Size,
    affine: AffineMatrix | None = None,
    max: Number = 1.0,
    min: Number = 0.0,
    square_size: Tuple[Number, ...] | Number = 10.0,
    ) -> Tuple[Image, AffineMatrix, BatchLabelImage, Points3D]:
    size = to_numpy(size)
    square_size = to_numpy(square_size, broadcast=len(size))
    dim = len(size)

    # Convert square size to voxels.
    if affine is None:
        affine = to_numpy(create_affine(dim=dim))
    spacing = affine_spacing(affine)
    ss_vox = square_size / spacing

    # Create checkerboard.
    indices = np.indices(size)   # shape: (n, d0, d1, ...)
    checkerboard = sum(indices[d] // ss_vox[d] for d in range(dim))
    checkerboard = (checkerboard % 2).astype(np.float32) * (max - min) + min

    # Create label masks: index 0 = black squares (min), index 1 = white squares (max).
    labels = np.stack([checkerboard == min, checkerboard == max], axis=0).astype(np.bool_)

    # Create points at each corner of each square (in world/mm space).
    n_squares = [int(np.ceil(size[d] / ss_vox[d])) for d in range(dim)]
    corner_coords = [np.arange(n_squares[d] + 1) * ss_vox[d] for d in range(dim)]
    grids = np.meshgrid(*corner_coords, indexing='ij')
    points_vox = np.stack([g.ravel() for g in grids], axis=1).astype(np.float32)   # (N, dim)
    points_h = np.hstack([points_vox, np.ones((len(points_vox), 1), dtype=np.float32)])
    points = (affine @ points_h.T).T[:, :dim].astype(np.float32)                   # (N, dim)

    return checkerboard, affine, labels, points

def load_example_ct() -> Tuple[Image3D, AffineMatrix3D, BatchLabelImage3D, Points3D]:
    data_dir = files("augmed.data")
    filepath = os.path.join(data_dir, "example_ct.nii.gz")
    image, affine = load_nifti(filepath)
    filepath = os.path.join(data_dir, "example_labels.npz")
    labels = load_numpy(filepath)
    filepath = os.path.join(data_dir, "example_points.npz")
    points = load_numpy(filepath)

    # Convert types.
    image = image.astype(np.float32)
    affine = affine.astype(np.float32)
    labels = labels.astype(np.bool_)
    points = points.astype(np.float32)

    return image, affine, labels, points
