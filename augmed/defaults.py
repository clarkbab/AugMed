import ast
import os
from typing import List, Literal, Tuple

from .typing import Number, Orientation, Orientation2D, Orientation3D, SpatialAxis, SpatialDim
from .utils.assertions import assert_dim, assert_dist, assert_fill, assert_interpolation, assert_dist_std, assert_orientation, assert_p, assert_seed

DEFAULT_DIM = 3
DEFAULT_DIST = 'uniform'
DEFAULT_DIST_STD = 3.0
DEFAULT_FILL = 'min'
DEFAULT_FILTER_OFFGRID = True
DEFAULT_INTERPOLATION = 'bilinear'
DEFAULT_ORIENTATION_2D = 'LS'
DEFAULT_ORIENTATION_3D = 'LPS'
DEFAULT_P = 1.0
DEFAULT_SEED = None
DEFAULT_VERBOSE = False

def get_dim() -> SpatialDim:
    return dim

def get_dist() -> str:
    return dist

def get_dist_std() -> float:
    return dist_std

def get_p() -> float:
    return p

def get_seed() -> int | None:
    return seed

def get_fill() -> Number | Literal['border', 'max', 'min', 'reflection', 'zeros']:
    return fill

def get_filter_offgrid() -> bool | SpatialAxis | List[SpatialAxis]:
    return filter_offgrid

def get_interpolation() -> Literal['bicubic', 'bilinear', 'nearest']:
    return interpolation

def get_orientation(
    dim: SpatialDim,
    ) -> Orientation:
    assert_dim(dim)
    if dim == 2:
        return orientation_2d
    elif dim == 3:
        return orientation_3d

def get_verbose() -> bool:
    return verbose

def init_dim() -> SpatialDim:
    dim = os.environ.get('AM_DIM')
    if dim is not None:
        try:
            dim = int(dim)
        except ValueError:
            raise ValueError(f"AM_DIM environment variable must be an integer (2 or 3), got '{dim}'.")
        assert_dim(dim, caller='AM_DIM environment variable')
        return dim
    return DEFAULT_DIM

def init_dist() -> str:
    dist = os.environ.get('AM_DIST')
    if dist is not None:
        assert_dist(dist, caller='AM_DIST environment variable')
        return dist
    return DEFAULT_DIST

def init_dist_std() -> float:
    n = os.environ.get('AM_DIST_STD')
    if n is not None:
        try:
            n = float(n)
        except ValueError:
            raise ValueError(f"AM_DIST_STD environment variable must be a positive number, got '{n}'.")
        assert_dist_std(n, caller='AM_DIST_STD environment variable')
        return n
    return DEFAULT_DIST_STD

def init_p() -> float:
    p_str = os.environ.get('AM_P')
    if p_str is not None:
        try:
            p = float(p_str)
        except ValueError:
            raise ValueError(f"AM_P environment variable must be a number between 0 and 1, got '{p_str}'.")
        assert_p(p, caller='AM_P environment variable')
        return p
    return DEFAULT_P

def init_seed() -> int | None:
    s = os.environ.get('AM_SEED')
    if s is not None:
        try:
            s = int(s)
        except ValueError:
            raise ValueError(f"AM_SEED environment variable must be an integer, got '{s}'.")
        assert_seed(s, caller='AM_SEED environment variable')
        return s
    return DEFAULT_SEED

def init_fill() -> Number | Literal['border', 'max', 'min', 'reflection', 'zeros']:
    f = os.environ.get('AM_FILL')
    if f is not None:
        try:
            return float(f)
        except ValueError:
            assert_fill(f, caller='AM_FILL environment variable')
            return f
    return DEFAULT_FILL

def init_filter_offgrid() -> bool | SpatialAxis | List[SpatialAxis]:
    fo = os.environ.get('AM_FILTER_OFFGRID')
    if fo is not None:
        if fo.lower() == 'true':
            return True
        elif fo.lower() == 'false':
            return False
        try:
            parsed = ast.literal_eval(fo)
            if isinstance(parsed, list):
                return [int(x) for x in parsed]
            return int(parsed)
        except (ValueError, SyntaxError):
            raise ValueError(f"AM_FILTER_OFFGRID environment variable must be 'true', 'false', an axis (0, 1, or 2), or a list like '[0,1]', got '{fo}'.")
    return DEFAULT_FILTER_OFFGRID

def init_interpolation() -> Literal['bicubic', 'bilinear', 'nearest']:
    interp = os.environ.get('AM_INTERPOLATION')
    if interp is not None:
        assert_interpolation(interp, caller='AM_INTERPOLATION environment variable')
        return interp
    return DEFAULT_INTERPOLATION

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

def init_verbose() -> bool:
    v = os.environ.get('AM_VERBOSE')
    if v is not None:
        if v.lower() == 'true':
            return True
        elif v.lower() == 'false':
            return False
        raise ValueError(f"AM_VERBOSE environment variable must be 'true' or 'false', got '{v}'.")
    return DEFAULT_VERBOSE

def set_dim(
    d: SpatialDim,
    ) -> None:
    assert_dim(d, caller='set_dim')
    global dim
    dim = d

def set_dist(
    d: str,
    ) -> None:
    assert_dist(d, caller='set_dist')
    global dist
    dist = d

def set_dist_std(
    n: float,
    ) -> None:
    assert_dist_std(n, caller='set_dist_std')
    global dist_std
    dist_std = n

def set_p(
    p_val: float,
    ) -> None:
    assert_p(p_val, caller='set_p')
    global p
    p = p_val

def set_seed(
    s: int | None,
    ) -> None:
    assert_seed(s, caller='set_seed')
    global seed
    seed = s

def set_fill(
    f: Number | Literal['border', 'max', 'min', 'reflection', 'zeros'],
    ) -> None:
    assert_fill(f, caller='set_fill')
    global fill
    fill = f

def set_filter_offgrid(
    fo: bool | SpatialAxis | List[SpatialAxis],
    ) -> None:
    global filter_offgrid
    filter_offgrid = fo

def set_interpolation(
    i: Literal['bicubic', 'bilinear', 'nearest'],
    ) -> None:
    assert_interpolation(i, caller='set_interpolation')
    global interpolation
    interpolation = i

def set_orientation(
    o: Orientation,
    dim: SpatialDim,
    ) -> None:
    assert_orientation(o, dim, caller='set_orientation')
    global orientation_2d, orientation_3d
    if dim == 2:
        orientation_2d = o
    elif dim == 3:
        orientation_3d = o

def set_verbose(
    v: bool,
    ) -> None:
    global verbose
    verbose = v

dim = init_dim()
dist = init_dist()
fill = init_fill()
filter_offgrid = init_filter_offgrid()
interpolation = init_interpolation()
dist_std = init_dist_std()
orientation_2d, orientation_3d = init_orientation()
p = init_p()
seed = init_seed()
verbose = init_verbose()
