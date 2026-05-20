from __future__ import annotations

import numpy as np
import torch
from typing import Tuple

from ...typing import Number, SamplingGridTensor, Size, Spacing, SpatialDim, TransformParams
from ...utils.args import alias_kwargs, expand_range_arg
from ...utils.conversion import to_tensor, to_tuple
from ...utils.geometry import affine_origin, affine_spacing, create_affine, fov_width
from ..identity import Identity
from .grid import GridTransform, RandomGridTransform

# This is a grid (not spatial) transform, so it shouldn't change the position/scale of objects in the world.
# 1. If we change the spacing, size should change to preserve field-of-view (mm).
# 2. If we change the size, spacing should change to preserve field-of-view (mm).
# 3. If we change both size/spacing, geometry is screwed. Should not allow.

class Resize(GridTransform):
    @alias_kwargs(
        ('sp', 'spacing'),
        ('spp', 'spacing_p'),
        ('sz', 'size'),
        ('szp', 'size_p'),
    )
    def __init__(
        self,
        size: int | Size | None = None,
        size_p: float | Tuple[float, ...] | None = None,
        spacing: float | Spacing | None = None,
        spacing_p: float | Tuple[float, ...] | None = None,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        assert ((size is not None) ^ (size_p is not None)) ^ ((spacing is not None) ^ (spacing_p is not None)), "Exactly one of 'size', 'size_p', 'spacing', or 'spacing_p' must be specified."
        self.__size = to_tensor(size, broadcast=self.__dim, dtype=torch.int32) if size is not None else None
        self.__size_p = to_tensor(size_p, broadcast=self.__dim) if size_p is not None else None
        self.__spacing = to_tensor(spacing, broadcast=self.__dim) if spacing is not None else None
        self.__spacing_p = to_tensor(spacing_p, broadcast=self.__dim) if spacing_p is not None else None

    @property
    def params(self) -> TransformParams:
        return super().params(
            size=to_tuple(self.__size) if self.__size is not None else None,
            size_p=to_tuple(self.__size_p) if self.__size_p is not None else None,
            spacing=to_tuple(self.__spacing) if self.__spacing is not None else None,
            spacing_p=to_tuple(self.__spacing_p) if self.__spacing_p is not None else None,
        )

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().__str__(
            self.__class__.__name__,
            size=to_tuple(self.__size) if self.__size is not None else None,
            size_p=to_tuple(self.__size_p, dp=3) if self.__size_p is not None else None,
            spacing=to_tuple(self.__spacing, dp=3) if self.__spacing is not None else None,
            spacing_p=to_tuple(self.__spacing_p, dp=3) if self.__spacing_p is not None else None,
            subtransform=subtransform,
        )

    def transform_grid(
        self,
        grid: SamplingGridTensor,
        **kwargs,
        ) -> SamplingGridTensor:
        size, affine = grid
        if affine is None:
            affine = create_affine(device=size.device, dim=self.__dim)

        if self.__size is not None or self.__size_p is not None:
            if self.__size is not None:
                size_t = self.__size.to(size.device)
            else:
                size_t = (size * self.__size_p.to(size.device)).type(torch.int32)
            # What does a resize mean without an affine?
            # For example, we might have a 64x64 image without an affine and we want 
            # to 2x downsample to 32x32. To faithfully construct the resampling grid
            # at the end, we need to know that the spacing has halved, so we need to
            # construct the affine assuming no input affine means identify affine. 

            # Change the spacing to preserve the field-of-view (mm).
            fov_w = fov_width((size, affine))
            spacing_t = fov_w / (size_t - 1)
            origin_t = affine_origin(affine)
            affine_t = create_affine(spacing_t, origin_t, device=size.device)
        else:
            if self.__spacing is not None:
                spacing_t = self.__spacing.to(size.device)
            else:
                spacing = affine_spacing(affine)
                spacing_t = spacing * self.__spacing_p.to(size.device)

            # Change the size to preserve the field-of-view (mm).
            fov_w = fov_width((size, affine))
            size_t = fov_w / spacing_t + 1
            origin_t = affine_origin(affine)
            affine_t = create_affine(spacing_t, origin_t, device=size.device)

        return size_t, affine_t

class RandomResize(RandomGridTransform):
    @alias_kwargs(
        ('sp', 'spacing'),
        ('spp', 'spacing_p'),
        ('sz', 'size'),
        ('szp', 'size_p'),
    )
    def __init__(
        self,
        # There are two parameters that can vary for the random resize.
        # 1. size: the size in pixels/voxels of the image.
        # 2. size_p: the size as a proportion of the original size.
        # 3. spacing: the spacing of the image.
        # 4. spacing_p: the spacing as a proportion of the original spacing.
        # TODO: Allow users to specify both size and size_p (or spacing and spacing_p).
        size: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = None,
        size_p: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = None,
        spacing: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = None,
        spacing_p: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = None,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        assert ((size is not None) ^ (size_p is not None)) ^ ((spacing is not None) ^ (spacing_p is not None)), "Exactly one of 'size', 'size_p', 'spacing', or 'spacing_p' must be specified."
        self.__size = size
        self.__size_p = size_p
        self.__spacing = spacing
        self.__spacing_p = spacing_p
        self.__expand_range_args()

    def __expand_range_args(self) -> None:
        vals_per_dim = 2
        size_range = size_p_range = spacing_range = spacing_p_range = None
        if self.__size is not None:
            expanded = expand_range_arg(self.__size, check_range='<=', dim=self.__dim, vals_per_dim=vals_per_dim)
            assert len(expanded) == vals_per_dim * self.__dim, f"Expected 'size' of length {vals_per_dim * self.__dim}, got {len(expanded)}."
            size_range = to_tensor(expanded).reshape(self.__dim, 2).T
        elif self.__size_p is not None:
            expanded = expand_range_arg(self.__size_p, check_range='<=', dim=self.__dim, vals_per_dim=vals_per_dim)
            assert len(expanded) == vals_per_dim * self.__dim, f"Expected 'size_p' of length {vals_per_dim * self.__dim}, got {len(expanded)}."
            size_p_range = to_tensor(expanded).reshape(self.__dim, 2).T
        elif self.__spacing is not None:
            expanded = expand_range_arg(self.__spacing, check_range='<=', dim=self.__dim, vals_per_dim=vals_per_dim)
            assert len(expanded) == vals_per_dim * self.__dim, f"Expected 'spacing' of length {vals_per_dim * self.__dim}, got {len(expanded)}."
            spacing_range = to_tensor(expanded).reshape(self.__dim, 2).T
        elif self.__spacing_p is not None:
            expanded = expand_range_arg(self.__spacing_p, check_range='<=', dim=self.__dim, vals_per_dim=vals_per_dim)
            assert len(expanded) == vals_per_dim * self.__dim, f"Expected 'spacing_p' of length {vals_per_dim * self.__dim}, got {len(expanded)}."
            spacing_p_range = to_tensor(expanded).reshape(self.__dim, 2).T
        self.__size_range = size_range
        self.__size_p_range = size_p_range
        self.__spacing_range = spacing_range
        self.__spacing_p_range = spacing_p_range

    # These are params that could be used to recreate the transform.
    # So they should match the input form (after expansion).
    def freeze(self) -> Resize | Identity:
        should_apply = self.__rng.random(1) < self.__p
        if not should_apply:
            return Identity(dim=self.__dim)

        # Draw resize params.
        draw = to_tensor(self.__rng.random(self.__dim))
        size_draw = size_p_draw = spacing_draw = spacing_p_draw = None
        if self.__size_range is not None:
            size_draw = (draw * (self.__size_range[1] - self.__size_range[0]) + self.__size_range[0]).type(torch.int32)
        elif self.__size_p_range is not None:
            size_p_draw = draw * (self.__size_p_range[1] - self.__size_p_range[0]) + self.__size_p_range[0]
        elif self.__spacing_range is not None:
            spacing_draw = draw * (self.__spacing_range[1] - self.__spacing_range[0]) + self.__spacing_range[0]
        elif self.__spacing_p_range is not None:
            spacing_p_draw = draw * (self.__spacing_p_range[1] - self.__spacing_p_range[0]) + self.__spacing_p_range[0]

        params = dict(
            size=size_draw,
            size_p=size_p_draw,
            spacing=spacing_draw,
            spacing_p=spacing_p_draw,
        )
        return super().freeze(Resize, params)

    @property
    def params(self) -> TransformParams:
        return super().params(
            size=to_tuple(self.__size_range.T.flatten()) if self.__size_range is not None else None,
            size_p=to_tuple(self.__size_p_range.T.flatten()) if self.__size_p_range is not None else None,
            spacing=to_tuple(self.__spacing_range.T.flatten()) if self.__spacing_range is not None else None,
            spacing_p=to_tuple(self.__spacing_p_range.T.flatten()) if self.__spacing_p_range is not None else None,
        )

    def set_dim(
        self,
        dim: SpatialDim,
        ) -> None:
        super().set_dim(dim)
        self.__expand_range_args()

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().__str__(
            self.__class__.__name__,
            size=to_tuple(self.__size_range.T.flatten()) if self.__size_range is not None else None,
            size_p=to_tuple(self.__size_p_range.T.flatten(), dp=3) if self.__size_p_range is not None else None,
            spacing=to_tuple(self.__spacing_range.T.flatten(), dp=3) if self.__spacing_range is not None else None,
            spacing_p=to_tuple(self.__spacing_p_range.T.flatten(), dp=3) if self.__spacing_p_range is not None else None,
            subtransform=subtransform,
        )
