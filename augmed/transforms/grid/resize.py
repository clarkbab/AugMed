from __future__ import annotations

import torch

from ...typing import SamplingGridTensor, Size, Spacing
from ...utils.args import alias_kwargs, expand_range_arg
from ...utils.conversion import to_tensor, to_tuple
from ...utils.geometry import affine_origin, affine_spacing, create_affine
from ..identity import Identity
from .grid import GridTransform, RandomGridTransform

# This is a grid (not spatial) transform, so it shouldn't change the position of objects in the world.
# 1. If we change the spacing, size should change to preserve geometry.
# 2. If we change the size, spacing should change to preserve geometry.
# 3. If we change both size/spacing, geometry is screwed. Should not allow.

class Resize(GridTransform):
    @alias_kwargs([
        ('sp', 'spacing'),
        ('sz', 'size'),
    ])
    def __init__(
        self,
        size: int | Size | None = None,
        spacing: float | Spacing | None = None,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        assert (size is not None or spacing is not None) and not (size is not None and spacing is not None), "Exactly one of 'size' or 'spacing' must be specified."
        self.__size = to_tensor(size, broadcast=self._dim, dtype=torch.int32) if size is not None else None
        self.__spacing = to_tensor(spacing, broadcast=self._dim) if spacing is not None else None
        super().set_params(
            self.__class__.__name__,
            size=self.__size,
            spacing=self.__spacing,
        )

    def __str__(self) -> str:
        return super().__str__(
            self.__class__.__name__,
            size=to_tuple(self.__size) if self.__size is not None else None,
            spacing=to_tuple(self.__spacing, decimals=3) if self.__spacing is not None else None,
        )

    def transform_grid(
        self,
        grid: SamplingGridTensor,
        **kwargs,
        ) -> SamplingGridTensor:
        size, affine = grid
        if self.__size is not None:
            size_t = self.__size.to(size.device)
            if affine is not None:
                spacing = affine_spacing(affine)
                spacing_t = (spacing * size / self.__size.to(size.device))
                origin_t = affine_origin(affine)
        else:
            spacing_t = self.__spacing.to(size.device)
            if affine is not None:
                spacing = affine_spacing(affine)
                size_t = (size * spacing / self.__spacing.to(size.device)).type(torch.int32)
                origin_t = affine_origin(affine)

        if affine is not None:
            affine_t = create_affine(spacing_t, origin_t, device=size.device)
        else:
            affine_t = None

        return size_t, affine_t

class RandomResize(RandomGridTransform):
    @alias_kwargs([
        ('sp', 'spacing'),
        ('sz', 'size'),
    ])
    def __init__(
        self,
        size: int | Size | None = None,
        spacing: float | Spacing | None = None,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        assert (size is not None or spacing is not None) and not (size is not None and spacing is not None), "Exactly one of 'size' or 'spacing' must be specified."
        self.__size = size
        self.__spacing = spacing
        super().set_params(
            self.__class__.__name__,
            size=self.__size,
            spacing=self.__spacing,
        )

    def freeze(self) -> Resize:
        # Expand the range args.
        # We do this now because 'set_dim' could be called after RandomResize.__init__.
        if self.__size is not None:
            size_range = expand_range_arg(self.__size, dim=self._dim)
            assert len(size_range) == 2 * self._dim, f"Expected 'size' of length {2 * self._dim} for dim={self._dim}, got length {len(size_range)}."
            size_range = to_tensor(size_range, dtype=torch.int32).reshape(self._dim, 2)
            spacing_range = None
        else:
            spacing_range = expand_range_arg(self.__spacing, dim=self._dim)
            assert len(spacing_range) == 2 * self._dim, f"Expected 'spacing_range' of length {2 * self._dim} for dim={self._dim}, got length {len(spacing_range)}."
            spacing_range = to_tensor(spacing_range).reshape(self._dim, 2)
            size_range = None

        # Draw the resize parameters.
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        if size_range is not None:
            draw = to_tensor(self._rng.random(self._dim))
            size_draw = (draw * (size_range[:, 1] - size_range[:, 0]) + size_range[:, 0]).type(torch.int32)
            spacing_draw = None
        else:
            draw = to_tensor(self._rng.random(self._dim))
            spacing_draw = (draw * (spacing_range[:, 1] - spacing_range[:, 0]) + spacing_range[:, 0])
            size_draw = None

        params = dict(
            size=size_draw,
            spacing=spacing_draw,
        )
        return super().freeze(Resize, params)

    def __str__(self) -> str:
        return super().__str__(
            self.__class__.__name__,
            size=self.__size,
            spacing=self.__spacing,
        )
