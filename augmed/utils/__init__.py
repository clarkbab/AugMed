from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .args import arg_to_list, bubble_args
    from .conversion import to_list, to_numpy, to_tensor
    from .geometry import affine_origin, affine_spacing, centre_of_mass, create_affine, foreground_fov, foreground_fov_centre, foreground_fov_width, fov, fov_centre, fov_width
    from .plotting import plot_hist, plot_slice, plot_volume

__all__ = [
    'arg_to_list', 'bubble_args',
    'to_list', 'to_numpy', 'to_tensor',
    'affine_origin', 'affine_spacing', 'centre_of_mass', 'create_affine', 'foreground_fov', 'foreground_fov_centre', 'foreground_fov_width', 'fov', 'fov_centre', 'fov_width',
    'plot_hist', 'plot_slice', 'plot_volume',
]

ARGS_IMPORTS = ['arg_to_list', 'bubble_args']
CONVERSION_IMPORTS = ['to_list', 'to_numpy', 'to_tensor']
GEOMETRY_IMPORTS = ['affine_origin', 'affine_spacing', 'centre_of_mass', 'create_affine', 'foreground_fov', 'foreground_fov_centre', 'foreground_fov_width', 'fov', 'fov_centre', 'fov_width']
PLOTTING_IMPORTS = ['plot_hist', 'plot_slice', 'plot_volume']

def __getattr__(name):
    if name in ARGS_IMPORTS:
        from . import args
        return getattr(args, name)
    if name in CONVERSION_IMPORTS:
        from . import conversion
        return getattr(conversion, name)
    if name in GEOMETRY_IMPORTS:
        from . import geometry
        return getattr(geometry, name)
    if name in PLOTTING_IMPORTS:
        from . import plotting
        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
