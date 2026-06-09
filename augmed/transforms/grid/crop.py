from __future__ import annotations

import numpy as np
import torch
from typing import List, Literal, Tuple

from ...typing import AffineMatrix, Indices, Number, Point, Points, SamplingGridTensor, Size, SpatialDim, TransformParams
from ...utils.args import alias_kwargs, arg_default, arg_to_list, expand_range_arg
from ...utils.assertions import assert_points_shapes, assert_range
from ...utils.conversion import to_numpy, to_return_format, to_tensor, to_tuple
from ...utils.geometry import affine_spacing, create_affine, fov, fov_centre, to_world_coords
from ...utils.python import get_group_device, wrap_quotes
from ..identity import Identity
from .grid import GridTransform, RandomGridTransform

DEFAULT_REMOVE_MM = 20.0

class Crop(GridTransform):
    @alias_kwargs(
        ('c', 'centre'),
        ('co', 'centre_offset'),
        ('m', 'margin'),
        ('r', 'remove'),
        ('s', 'size'),
    )
    def __init__(
        self,
        centre: Point | Literal['image-centre'] = 'image-centre',
        centre_offset: Number | Tuple[Number, ...] = 0.0,
        margin: Number | Tuple[Number | None, ...] | None = None,
        remove: Number | Tuple[Number | None, ...] | None = None,
        size: Size | None = None,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        remove = arg_default((margin, remove, size), remove, DEFAULT_REMOVE_MM)
        assert (margin is not None) ^ (remove is not None) ^ (size is not None), "Exactly one of 'margin', 'remove', 'size' must be specified."
        self.__centre = self.__centre_offset = self.__margin = self.__remove = self.__size = None
        if margin is not None or size is not None:
            if margin is not None:
                margin = expand_range_arg(margin, dim=self.__dim)
                assert_range(margin, self.__dim, 'margin')
                self.__margin = to_tensor(margin).reshape(self.__dim, 2)
            elif size is not None:
                size = expand_range_arg(size, dim=self.__dim, vals_per_dim=1)
                self.__size = to_tensor(size, dtype=torch.float32)
                assert len(self.__size) == self.__dim
            self.__centre = to_tuple(centre, broadcast=self.__dim)   # Tensors can't store str types.
            assert len(self.__centre) == self.__dim
            self.__centre_offset = to_tensor(centre_offset, broadcast=self.__dim)
            assert len(self.__centre_offset) == self.__dim
        elif remove is not None:
            remove = expand_range_arg(remove, dim=self.__dim)
            self.__remove = to_tensor(remove).reshape(self.__dim, 2)

    @property
    def params(self) -> TransformParams:
        return super().params(
            centre=to_tuple(self.__centre) if self.__centre != 'image-centre' else wrap_quotes('image-centre'),
            centre_offset=to_tuple(self.__centre_offset) if self.__centre_offset is not None else None,
            margin=to_tuple(self.__margin.flatten()) if self.__margin is not None else None,
            remove=to_tuple(self.__remove.flatten()) if self.__remove is not None else None,
            size=to_tuple(self.__range.flatten()) if self.__size is not None else None,
        )

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().__str__(
            self.__class__.__name__,
            centre=to_tuple(self.__centre, dp=3) if self.__centre != 'image-centre' else wrap_quotes('image-centre'),
            centre_offset=to_tuple(self.__centre_offset.flatten(), dp=3) if self.__centre_offset is not None else None,
            margin=to_tuple(self.__margin.flatten(), dp=3) if self.__margin is not None else None,
            remove=to_tuple(self.__remove.flatten(), dp=3) if self.__remove is not None else None,
            size=to_tuple(self.__size.flatten(), dp=3) if self.__size is not None else None,
            subtransform=subtransform,
        )

    # Crop should be able to pad too due to the randomised offset.
    def transform_grid(
        self,
        grid: SamplingGridTensor,
        **kwargs,
        ) -> SamplingGridTensor:
        size, affine = grid
        if self.__remove is not None:
            # Get the current FOV in image coords.
            fov_min, fov_max = fov((size, None))

            # Get the amounts to remove.
            # Could be either image/world coords depending on if affine is passed.
            remove_min = self.__remove[:, 0].to(size.device)
            remove_max = self.__remove[:, 1].to(size.device)

            # Convert to image coords.
            if affine is not None:
                remove_min /= affine_spacing(affine)
                remove_max /= affine_spacing(affine)

            # Get the new FOV.
            fov_min = torch.clamp(fov_min + remove_min, 0)
            fov_max = torch.clamp(fov_max - remove_max, max=(size - 1))
        else:
            # Get the crop centre.
            fov_c = fov_centre((size, None))
            centre = [fov_c[i] if c == 'image-centre' else c for i, c in enumerate(self.__centre)]
            centre = to_tensor(centre, device=size.device, dtype=torch.float32)

            # Apply the crop centre offset.
            centre_offset = self.__centre_offset.to(size.device)
            if affine is not None:
                centre_offset /= affine_spacing(affine)
            centre += centre_offset

            # Get the margin sizes.
            if self.__margin is not None:
                margin_min = self.__margin[:, 0].to(size.device)
                margin_max = self.__margin[:, 1].to(size.device)
            elif self.__size is not None:
                # Convert size to margin.
                margin_min = (self.__size / 2).to(size.device)
                margin_max = (self.__size / 2).to(size.device)
            if affine is not None:
                margin_min *= affine_spacing(affine)
                margin_max *= affine_spacing(affine)


            # Get the new FOV.
            # Allow crop transform to pad, for example if we choose a size and a centre offset.
            fov_min = centre - margin_min
            fov_max = centre + margin_max
            # fov_min = torch.clamp(centre - margin_min, 0)
            # fov_max = torch.clamp(centre + margin_max, max=(size - 1))

        # Get new size.
        size_t = fov_max - fov_min
        size_t = size_t.clamp(0)

        # Check result.
        if torch.any(size_t == 0):
            raise ValueError(f"{self} would create image with size zero along one or more axes (size={to_tuple(size_t)}).")

        # Get new affine.
        if affine is not None:
            # Crop doesn't change voxel spacing, but it does change the position of the 0th voxel in world coordinates.
            spacing_t = affine_spacing(affine)
            origin_t = to_world_coords(fov_min, affine)
            affine_t = create_affine(spacing_t, origin_t, device=size.device)
        else:
            affine_t = None

        # Convert types.
        size_t = size_t.round().type(torch.int32)

        return size_t, affine_t

    @alias_kwargs(
        ('a', 'affine'),
        ('fo', 'filter_offgrid'),
        ('rf', 'return_filtered'),
        ('s', 'size'),
    )
    def transform_points(
        self,
        points: Points | List[Points],
        affine: AffineMatrix | None = None,       # Required for some transforms, e.g. Rotate, to get centre of rotation.
        filter_offgrid: bool | SpatialDim | List[SpatialDim] | None = None,
        return_filtered: bool = False,
        size: Size | None = None,           # Required for filtering off-grid points.
        **kwargs,
        ) -> Points | List[Points | Indices | List[Indices]]:
        assert_points_shapes(points, self.__dim)
        points = arg_to_list(points, (np.ndarray, torch.Tensor))
        device = get_group_device(points, device=self.__device)
        return_types = [type(p) for p in points]
        points = [to_tensor(p, device=device, dtype=torch.float32) for p in points]
        size = to_tensor(size, device=device, dtype=torch.int32)
        affine = to_tensor(affine, device=device, dtype=torch.float32)
        filter_offgrid = filter_offgrid if filter_offgrid is not None else self.__filter_offgrid

        points_ts = []
        indiceses = []
        for p in points:
            points_t = p

            # Forward transformed points could end up off-screen and should be filtered.
            # However, we need to know which points are returned for loss calc for example.
            if filter_offgrid is not False:    # "is not" required here because filter_offgrid=0 is valid.
                # Get new FOV.
                assert size is not None, "Size must be provided for filtering off-grid points."
                assert affine is not None, "Affine must be provided for filtering off-grid points."
                size_t, affine_t = self.transform_grid((size, affine))

                # Crop points to the new FOV.
                fov_mm = fov((size_t, affine_t))
                if filter_offgrid is True:
                    in_fov = (points_t >= fov_mm[0]) & (points_t < fov_mm[1])
                else:
                    dims = arg_to_list(filter_offgrid, int)
                    in_fov = (points_t[:, dims] >= fov_mm[0][dims]) & (points_t[:, dims] < fov_mm[1][dims])
                to_keep = in_fov.all(axis=1)
                points_t = points_t[to_keep]
                indices = torch.where(~to_keep)[0].type(torch.int32)
                indiceses.append(indices)

            points_ts.append(points_t)

        # Convert to return format.
        other_data = []
        if filter_offgrid and return_filtered:
            indiceses = to_return_format(indiceses, return_types=return_types)
            other_data.append(indiceses)
        return to_return_format(points_ts, other_data=other_data, return_types=return_types)

class RandomCrop(RandomGridTransform):
    @alias_kwargs(
        ('c', 'centre'),
        ('co', 'centre_offset'),
        ('m', 'margin'),
        ('r', 'remove'),
        ('s', 'size'),
        ('sym', 'symmetric'),
    )
    def __init__(
        self,
        # How many ways are there to define a crop?
        # 1. Removing an amount off each axis end ('remove').
        # 2. Cropping using a centre and margin ('centre', 'margin').
        # 3. Using defined values in image/world coordinates.
        # 4. TODO: Cropping around a label centre or boundary.
        # 5. Specify an output size and vary the crop centre.
        # Must keep 'centre' and 'centre_offset' separate so we can specify image centre using 'image-centre'.
        centre: Point | Literal['image-centre'] | None = 'image-centre',
        centre_offset: Range2PerDim = 0.0,
        margin: Range4PerDim | None = None,
        remove: Range4PerDim | None = None,
        # Whilst margin gives more flexibility - the user can specify assymetric margins - size
        # is a convenience method that gives symmetric margins.
        size: Range2PerDim | None = None,
        # Cropped amounts are the same at both ends of each axis.
        # This should be configured per axis really, for example we might want want symmetry
        # along the x-axis only.
        symmetric: bool | Tuple[bool, ...] = False,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        remove = arg_default((margin, remove, size), remove, DEFAULT_REMOVE_MM)
        assert (margin is not None) ^ (remove is not None) ^ (size is not None), "Exactly one of 'margin', 'remove', 'size' must be specified."
        self.__centre = centre
        self.__centre_offset = centre_offset
        self.__margin = margin
        self.__remove = remove
        self.__size = size
        self.__symmetric = symmetric
        self.__expand_range_args()

    def __expand_range_args(self) -> None:
        self.__symmetric = to_tensor(self.__symmetric, broadcast=self.__dim)
        self.__centre_offset_range = self.__margin_range = self.__remove_range = self.__size_range = None
        if self.__margin is not None or self.__size is not None:
            if self.__margin is not None:
                vals_per_dim = 4
                margin_range = expand_range_arg(self.__margin, check_range='<=', dim=self.__dim, vals_per_dim=vals_per_dim)
                assert_range(margin_range, self.__dim, 'margin', vals_per_dim=vals_per_dim)
                for i, s in enumerate(self.__symmetric):
                    cmr_axis_vals = margin_range[i * vals_per_dim:(i + 1) * vals_per_dim]
                    if s and (cmr_axis_vals[0] != cmr_axis_vals[2] or cmr_axis_vals[1] != cmr_axis_vals[3]):
                        raise ValueError(f"Cannot create symmetric crops for axis {i} with crop margin ranges {cmr_axis_vals}.")
                self.__margin_range = to_tensor(margin_range).reshape(self.__dim, 2, 2).T
            elif self.__size is not None:
                vals_per_dim = 2
                size_range = expand_range_arg(self.__size, check_range='<=', dim=self.__dim, vals_per_dim=2)
                assert_range(size_range, self.__dim, 'margin', vals_per_dim=vals_per_dim)
                self.__size_range = to_tensor(size_range).reshape(self.__dim, 2).T
            centre_offset_range = expand_range_arg(self.__centre_offset, check_range='<=', dim=self.__dim, negate_lower=True)
            assert_range(centre_offset_range, self.__dim, 'centre_offset')
            self.__centre_offset_range = to_tensor(centre_offset_range).reshape(self.__dim, 2).T
        elif self.__remove is not None:
            vals_per_dim = 4
            remove_range = expand_range_arg(self.__remove, check_range='<=', dim=self.__dim, vals_per_dim=vals_per_dim)
            assert_range(remove_range, self.__dim, 'remove', vals_per_dim=vals_per_dim)
            for i, s in enumerate(self.__symmetric):
                cr_axis_vals = remove_range[i * vals_per_dim:(i + 1) * vals_per_dim]
                if s and (cr_axis_vals[0] != cr_axis_vals[2] or cr_axis_vals[1] != cr_axis_vals[3]):
                    raise ValueError(f"Cannot create symmetric crops for axis {i} with crop ranges {cr_axis_vals}.")
            self.__remove_range = to_tensor(remove_range).reshape(self.__dim, 2, 2).T

    def freeze(self) -> Crop | Identity:
        should_apply = self.__rng.random(1) < self.__p
        if not should_apply:
            return Identity(dim=self.__dim)

        if self.__margin_range is not None or self.__size_range is not None:
            draw = to_tensor(self.__rng.random((2, self.__dim)))
            centre_offset_draw = to_tensor(self.__rng.random(self.__dim)) * (self.__centre_offset_range[1] - self.__centre_offset_range[0]) + self.__centre_offset_range[0]
            params = dict(
                centre=self.__centre,
                centre_offset=centre_offset_draw,
            )
            if self.__margin_range is not None:
                draw = to_tensor(self.__rng.random((2, self.__dim)))
                margin_draw = draw * (self.__margin_range[1] - self.__margin_range[0]) + self.__margin_range[0]
                sym_axes = torch.argwhere(self.__symmetric_t).flatten()
                margin_draw[1, sym_axes] = margin_draw[0, sym_axes]
                params['margin'] = margin_draw.T.flatten()
            elif self.__size_range is not None:
                draw = to_tensor(self.__rng.random(self.__dim))
                size_draw = draw * (self.__size_range[1] - self.__size_range[0]) + self.__size_range[0]
                params['size'] = size_draw.T.flatten()
        elif self.__remove_range is not None:
            draw = to_tensor(self.__rng.random((2, self.__dim)))
            remove_draw = draw * (self.__remove_range[1] - self.__remove_range[0]) + self.__remove_range[0]
            sym_axes = torch.argwhere(self.__symmetric_t).flatten()
            remove_draw[1, sym_axes] = remove_draw[0, sym_axes]
            params = dict(
                remove=remove_draw.T.flatten(),
            )
        return super().freeze(Crop, params)

    @property
    def params(self) -> TransformParams:
        return super().params(
            centre=self.__centre,
            centre_offset=to_tuple(self.__centre_offset_range.T.flatten()) if self.__centre_offset_range is not None else None,
            margin=to_tuple(self.__margin_range.T.flatten()) if self.__margin_range is not None else None,
            remove=to_tuple(self.__remove_range.T.flatten()) if self.__remove_range is not None else None,
            size=to_tuple(self.__size_range.T.flatten()) if self.__size_range is not None else None,
            symmetric=self.__symmetric,
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
            centre=to_tuple(self.__centre, dp=3) if self.__centre != 'image-centre' else wrap_quotes('image-centre'),
            centre_offset=to_tuple(self.__centre_offset_range.T.flatten(), dp=3) if self.__centre_offset_range is not None else None,
            margin=to_tuple(self.__margin_range.T.flatten(), dp=3) if self.__margin_range is not None else None,
            remove=to_tuple(self.__remove_range.T.flatten(), dp=3) if self.__remove_range is not None else None,
            size=to_tuple(self.__size_range.T.flatten(), dp=3) if self.__size_range is not None else None,
            subtransform=subtransform,
            symmetric=self.__symmetric,
        )
