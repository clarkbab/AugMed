from typing import *

from ...typing import *
from ...utils.args import alias_kwargs, expand_range_arg
from ...utils.conversion import to_tensor, to_tuple
from ...utils.matrix import affine_origin, affine_spacing, create_eye, create_affine
from ..identity import Identity
from .grid import RandomGridTransform, GridTransform

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
        self._params = dict(
            dim=self._dim,
            size=self.__size,
            spacing=self.__spacing,
            type=self.__class__.__name__,
        )

    def __str__(self) -> str:
        params = dict(
            size=to_tuple(self.__size) if self.__size is not None else None,
            spacing=to_tuple(self.__spacing, decimals=3) if self.__spacing is not None else None,
        )
        return super().__str__(self.__class__.__name__, params)

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
        if size is not None:
            size_range = expand_range_arg(size, dim=self._dim)
            assert len(size_range) == 2 * self._dim, f"Expected 'size' of length {2 * self._dim} for dim={self._dim}, got length {len(size_range)}."
            self.__size = to_tensor(size_range, dtype=torch.int32).reshape(self._dim, 2)
            self.__spacing = None
        if spacing is not None:
            spacing_range = expand_range_arg(spacing, dim=self._dim)
            assert len(spacing_range) == 2 * self._dim, f"Expected 'spacing_range' of length {2 * self._dim} for dim={self._dim}, got length {len(spacing_range)}."
            self.__spacing = to_tensor(spacing_range).reshape(self._dim, 2)
            self.__size = None

        self._params = dict(
            dim=self._dim,
            p=self._p,
            size=self.__size,
            spacing=self.__spacing,
            type=self.__class__.__name__,
        )

    def freeze(self) -> 'Resize':
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        if self.__size is not None:
            draw = to_tensor(self._rng.random(self._dim))
            size_draw = (draw * (self.__size[:, 1] - self.__size[:, 0]) + self.__size[:, 0]).type(torch.int32)
            spacing_draw = None
        if self.__spacing is not None:
            draw = to_tensor(self._rng.random(self._dim))
            spacing_draw = (draw * (self.__spacing[:, 1] - self.__spacing[:, 0]) + self.__spacing[:, 0])
            size_draw = None

        params = dict(
            size=size_draw,
            spacing=spacing_draw,
        )
        return super().freeze(Resize, params)

    def __str__(self) -> str:
        params = dict(
            size=to_tuple(self.__size.flatten()) if self.__size is not None else None,
            spacing=to_tuple(self.__spacing.flatten(), decimals=3) if self.__spacing is not None else None,
        )
        return super().__str__(self.__class__.__name__, params)
