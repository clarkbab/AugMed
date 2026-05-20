from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

LAZY_IMPORTS = {
    'crop': ['Crop', 'RandomCrop'],
    'grid': ['GridTransform', 'RandomGridTransform'],
    'pad': ['Pad', 'RandomPad'],
    'resize': ['RandomResize', 'Resize'],
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
