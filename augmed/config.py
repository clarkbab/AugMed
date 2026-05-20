import os

from .typing import Orientation, SpatialDim
from .utils.assertions import assert_orientation

DEFAULT_DIM = 3
DEFAULT_ORIENTATION = 'LPS'

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

def init_orientation() -> Orientation:
    o = os.environ.get('AM_ORIENT')
    if o is not None:
        assert_orientation(o, dim)
        return o
    return DEFAULT_ORIENTATION

# Global variables.
dim = init_dim()
orientation = init_orientation()

def get_dim() -> SpatialDim:
    return dim

def get_orientation() -> Orientation:
    return orientation

def set_dim(
    d: SpatialDim,
    ) -> None:
    if d not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {d}.")
    global dim
    dim = d

def set_orientation(
    o: Orientation,
    ) -> None:
    assert_orientation(o, dim)
    global orientation
    orientation = o
