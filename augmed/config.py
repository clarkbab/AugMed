import os
from typing import Tuple

from .typing import Orientation, Orientation2D, Orientation3D, SpatialDim
from .utils.assertions import assert_dim, assert_orientation

DEFAULT_DIM = 3
DEFAULT_ORIENTATION_2D = 'LS'
DEFAULT_ORIENTATION_3D = 'LPS'

def get_dim() -> SpatialDim:
    return dim

def get_orientation(
    dim: SpatialDim,
    ) -> Orientation:
    assert_dim(dim)
    if dim == 2:
        return orientation_2d
    elif dim == 3:
        return orientation_3d

def init_dim() -> SpatialDim:
    dim = os.environ.get('AM_DIM')
    if dim is not None:
        try:
            dim = int(dim)
        except ValueError:
            raise ValueError(f"AM_DIM environment variable must be an integer (2 or 3), got '{dim}'.")
        if dim not in (2, 3):
            raise ValueError(f"AM_DIM environment variable must be 2 or 3, got {dim}.")
        return dim
    return DEFAULT_DIM

def init_orientation() -> Tuple[Orientation2D, Orientation3D]:
    o2d = os.environ.get('AM_ORIENT_2D')
    if o2d is not None:
        assert_orientation(o2d, 2)
    else:
        o2d = DEFAULT_ORIENTATION_2D
    o3d = os.environ.get('AM_ORIENT_3D')
    if o3d is not None:
        assert_orientation(o3d, 3)
    else:
        o3d = DEFAULT_ORIENTATION_3D
    return o2d, o3d

def set_dim(
    d: SpatialDim,
    ) -> None:
    if d not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {d}.")
    global dim
    dim = d

def set_orientation(
    o: Orientation,
    dim: SpatialDim,
    ) -> None:
    assert_orientation(o, dim)
    global orientation_2d, orientation_3d
    if dim == 2:
        orientation_2d = o
    elif dim == 3:
        orientation_3d = o

dim = init_dim()
orientation_2d, orientation_3d = init_orientation()
