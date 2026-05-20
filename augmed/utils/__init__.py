from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

LAZY_IMPORTS = {
    'args': ['alias_kwargs', 'arg_default', 'arg_to_list', 'bubble_args', 'expand_range_arg', 'get_inner_args'],
    'assertions': ['assert_dim', 'assert_image_shapes', 'assert_image_sizes', 'assert_orientation', 'assert_points_shapes'],
    'conversion': ['to_list', 'to_numpy', 'to_return_format', 'to_tensor', 'to_tuple'],
    'debug': ['from_desc'],
    'geometry': [
        'affine_origin', 'affine_spacing', 'centre_of_mass', 'create_affine',
        'foreground_fov', 'foreground_fov_centre', 'foreground_fov_width',
        'fov', 'fov_centre', 'fov_width', 'spatial_size',
        'to_image_coords', 'to_world_coords',
    ],
    'grid': ['grid_points', 'grid_sample'],
    'io': ['load_nifti', 'load_nrrd', 'load_numpy', 'make_serialisable', 'save_json', 'save_yaml'],
    'logging': ['logger'],
    'maths': ['round'],
    'matrix': ['create_rotation', 'create_scaling', 'create_translation'],
    'plotting': ['plot_hist', 'plot_slice', 'plot_volume'],
    'points': ['filter_points'],
    'python': ['get_group_device', 'is_generic', 'is_windows', 'isinstance_generic', 'version', 'wrap_quotes'],
}

__all__ = [attr for attrs in LAZY_IMPORTS.values() for attr in attrs]

if TYPE_CHECKING:
    for module, attrs in LAZY_IMPORTS.items():
        for attr in attrs:
            exec(f"from .{module} import {attr}")

def __getattr__(name):
    for module, attrs in LAZY_IMPORTS.items():
        if name in attrs:
            return getattr(importlib.import_module(f"{__name__}.{module}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
