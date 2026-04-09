from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conversion import to_numpy, to_tensor
    from .geometry import affine_origin, affine_spacing, create_affine
    from .plotting import plot_slice, plot_volume

__all__ = [
    'to_numpy', 'to_tensor',
    'affine_origin', 'affine_spacing', 'create_affine',
    'plot_slice', 'plot_volume',
]

CONVERSION_IMPORTS = ['to_numpy', 'to_tensor']
GEOMETRY_IMPORTS = ['affine_origin', 'affine_spacing', 'create_affine']
PLOTTING_IMPORTS = ['plot_slice', 'plot_volume']

def __getattr__(name):
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
