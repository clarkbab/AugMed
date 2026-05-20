from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

LAZY_IMPORTS = {
    'affine': ['Affine', 'RandomAffine'],
    'flip': ['Flip', 'RandomFlip'],
    'rotate': ['RandomRotate', 'Rotate'],
    'scale': ['RandomScale', 'Scale'],
    'shear': ['RandomShear', 'Shear'],
    'translate': ['RandomTranslate', 'Translate'],
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
