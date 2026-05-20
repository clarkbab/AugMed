from __future__ import annotations

import numpy as np
import torch
from typing import List, Literal, Tuple

from ...typing import AffineMatrix, Indices, Number, Points, SamplingGridTensor, Size, SpatialDim, TransformParams
from ...utils.args import alias_kwargs, arg_to_list, expand_range_arg
from ...utils.assertions import assert_points_shapes
from ...utils.conversion import to_return_format, to_tensor, to_tuple
from ...utils.geometry import affine_spacing, create_affine, fov, fov_centre, to_world_coords
from ...utils.python import get_group_device, wrap_quotes
from ..identity import Identity
from .grid import GridTransform, RandomGridTransform

class Pad(GridTransform):
    @alias_kwargs(
        ('a', 'add'),
        ('c', 'centre'),
        ('co', 'centre_offset'),
        ('m', 'margin'),
    )
    def __init__(
        self,
        add: Number | Tuple[Number, ...] | None = None,
        centre: Number | Literal['image-centre'] | Tuple[Number | Literal['image-centre'], ...] | None = 'image-centre',
        centre_offset: Number | Tuple[Number, ...] | None = 0.0,
        margin: Number | Tuple[Number, ...] | None = None,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        assert add is not None or (centre is not None and margin is not None)
        if add is not None:
            add_range = expand_range_arg(add, dim=self.__dim)
            self.__add = to_tensor(add_range).reshape(self.__dim, 2).T
            self.__margin = None
            self.__centre = None
            self.__centre_offset = None
        else:
            margin_range = expand_range_arg(margin, dim=self.__dim)
            self.__margin = to_tensor(margin_range).reshape(self.__dim, 2).T
            self.__centre = to_tuple(centre, broadcast=self.__dim)   # Tensors can't store str types.
            assert len(self.__centre) == self.__dim
            self.__centre_offset = to_tensor(centre_offset, broadcast=self.__dim)
            assert len(self.__centre_offset) == self.__dim

    @property
    def params(self) -> TransformParams:
        return super().params(
            add=to_tuple(self.__add.T.flatten()) if self.__add is not None else None,
            centre=to_tuple(self.__centre) if self.__centre != 'image-centre' else wrap_quotes('image-centre'),
            centre_offset=to_tuple(self.__centre_offset) if self.__centre_offset is not None else None,
            margin=to_tuple(self.__margin.T.flatten()) if self.__margin is not None else None,
        )

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().__str__(
            self.__class__.__name__,
            add=to_tuple(self.__add.T.flatten(), dp=3) if self.__add is not None else None,
            centre=to_tuple(self.__centre, dp=3) if self.__centre != 'image-centre' else wrap_quotes('image-centre'),
            centre_offset=to_tuple(self.__centre_offset.flatten(), dp=3) if self.__centre_offset is not None else None,
            margin=to_tuple(self.__margin.T.flatten(), dp=3) if self.__margin is not None else None,
            subtransform=subtransform,
        )

    def transform_grid(
        self,
        grid: SamplingGridTensor,
        **kwargs,
        ) -> SamplingGridTensor:
        size, affine = grid
        if self.__add is not None:
            # Get the current FOV.
            fov_min, fov_max = fov((size, None))

            # Get the amounts to add.
            add_min = self.__add[0].to(size.device)
            add_max = self.__add[1].to(size.device)
            if affine is not None:
                add_min /= affine_spacing(affine)
                add_max /= affine_spacing(affine)

            # Get the new FOV.
            fov_min = torch.clamp(fov_min - add_min, 0)
            fov_max = torch.clamp(fov_max + add_max, max=(size - 1))
        else:
            # Get pad centre.
            fov_c = fov_centre((size, None))
            centre = [fov_c[i] if c == 'image-centre' else c for i, c in enumerate(self.__centre)]
            centre = to_tensor(centre, device=size.device)

            # Get centre offset.
            centre_offset = self.__centre_offset.to(size.device)
            if affine is not None:
                centre_offset /= affine_spacing(affine)
            centre += centre_offset

            # Get pad box.
            margin_min = self.__margin[0].to(size.device)
            margin_max = self.__margin[1].to(size.device)
            if affine is not None:
                margin_min /= affine_spacing(affine)
                margin_max /= affine_spacing(affine)

            # Truncate to true voxel coords.
            fov_min = torch.clamp(centre - margin_min, 0)
            fov_max = torch.clamp(centre + margin_max, max=(size - 1))

        # Get new size.
        size_t = fov_max - fov_min + 1
        size_t = size_t.clamp(0)

        # Get new affine.
        if affine is not None:
            # Crop doesn't change voxel spacing, but it does change the position of the 0th voxel in world coordinates.
            spacing_t = affine_spacing(affine)
            origin_t = to_world_coords(fov_min, affine)
            affine_t = create_affine(spacing_t, origin_t, device=size.device)
        else:
            affine_t = None

        # Convert types.
        size_t = size_t.type(torch.int32)

        return size_t, affine_t

    @alias_kwargs(
        ('a', 'affine'),
        ('fo', 'filter_offgrid'),
        ('rf', 'return_filtered'),
        ('rs', 'return_single'),
        ('s', 'size'),
    )
    def transform_points(
        self,
        points: Points | List[Points],
        # TODO: Alter filter_offgrid part as pad can never move points off-grid.
        affine: AffineMatrix | None = None,       # Required for some transforms, e.g. Rotate, to get centre of rotation.
        filter_offgrid: bool | SpatialDim | List[SpatialDim] | None = None,
        # grid: SamplingGrid | None = None,   # Required for filtering off-grid points and some transforms, e.g. Rotate.
        return_filtered: bool = False,
        return_single: bool = True,
        size: Size | None = None,           # Required for filtering off-grid points.
        **kwargs,
        ) -> Points | List[Points | Indices | List[Indices]]:
        assert_points_shapes(points)
        points, points_was_single = arg_to_list(points, (np.ndarray, torch.Tensor), return_matched=True)
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
                # Get new grid.
                assert size is not None, "Size must be provided for filtering off-grid points."
                assert affine is not None, "Affine must be provided for filtering off-grid points."
                size_t, affine_t = self.transform_grid((size, affine))

                # Pad points.
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
            indiceses = to_return_format(indiceses, return_single=True, return_types=return_types)
            other_data.append(indiceses)
        return to_return_format(points_ts, other_data=other_data, return_single=return_single and points_was_single, return_types=return_types)

class RandomPad(RandomGridTransform):
    @alias_kwargs(
        ('a', 'add'),
        ('c', 'centre'),
        ('co', 'centre_offset'),
        ('m', 'margin'),
        ('s', 'symmetric'),
    )
    def __init__(
        self,
        # How many ways are there to define a pad?
        # 1. Removing an amount off each axis end ('pad_remove').
        # 2. Padding using a centre and margin ('centre', 'margin').
        # 3. Using defined values in image/world coordinates.
        # 4. TODO: Padding around a label centre or boundary.
        add: Number | Tuple[Number, ...] | None = None,
        centre: Number | Literal['image-centre'] | Tuple[Number | Literal['image-centre'], ...] | None = 'image-centre',
        centre_offset: Number | Tuple[Number, ...] | None = 0.0,
        margin: Number | Tuple[Number, ...] | None = None,
        # padded amounts are the same at both ends of each axis.
        # This should be configured per axis really, for example we might want want symmetry
        # along the x-axis only.
        symmetric: bool | Tuple[bool, ...] = False,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        assert add is not None or (centre is not None and margin is not None), "Must specify either 'add' or both 'centre' and 'margin'."
        self.__add = add
        self.__centre = centre
        self.__centre_offset = centre_offset
        self.__margin = margin
        self.__symmetric = symmetric
        self.__expand_range_args()

    def __expand_range_args(self) -> None:
        symmetric = to_tensor(self.__symmetric, broadcast=self.__dim)
        add_range = margin_range = centre_offset_range = None
        if self.__add is not None:
            cr_vals_per_dim = 4
            add_range = expand_range_arg(self.__add, dim=self.__dim, vals_per_dim=cr_vals_per_dim)
            assert len(add_range) == cr_vals_per_dim * self.__dim, f"Expected 'add' of length {cr_vals_per_dim * self.__dim}, got {len(add_range)}."
            for i, s in enumerate(symmetric):
                cr_axis_vals = add_range[i * cr_vals_per_dim:(i + 1) * cr_vals_per_dim]
                if s and (cr_axis_vals[0] != cr_axis_vals[2] or cr_axis_vals[1] != cr_axis_vals[3]):
                    raise ValueError(f"Cannot create symmetric pads for axis {i} with pad ranges {cr_axis_vals}.")
            add_range = to_tensor(add_range).reshape(self.__dim, 2, 2).permute(2, 1, 0)
        else:
            cmr_vals_per_dim = 4
            margin_range = expand_range_arg(self.__margin, dim=self.__dim, vals_per_dim=cmr_vals_per_dim)
            assert len(margin_range) == cmr_vals_per_dim * self.__dim, f"Expected 'margin' of length {cmr_vals_per_dim * self.__dim}, got {len(margin_range)}."
            for i, s in enumerate(symmetric):
                cmr_axis_vals = margin_range[i * cmr_vals_per_dim:(i + 1) * cmr_vals_per_dim]
                if s and (cmr_axis_vals[0] != cmr_axis_vals[2] or cmr_axis_vals[1] != cmr_axis_vals[3]):
                    raise ValueError(f"Cannot create symmetric pads for axis {i} with pad margin ranges {cmr_axis_vals}.")
            margin_range = to_tensor(margin_range).reshape(self.__dim, 2, 2).permute(2, 1, 0)
            centre_offset_range = expand_range_arg(self.__centre_offset, dim=self.__dim, negate_lower=True)
            assert len(centre_offset_range) == 2 * self.__dim, f"Expected 'centre_offset' of length {2 * self.__dim}, got {len(centre_offset_range)}."
            centre_offset_range = to_tensor(centre_offset_range).reshape(self.__dim, 2).T
        self.__add_range = add_range
        self.__margin_range = margin_range
        self.__centre_offset_range = centre_offset_range
        self.__symmetric_t = symmetric

    def freeze(self) -> Pad | Identity:
        should_apply = self.__rng.random(1) < self.__p
        if not should_apply:
            return Identity(dim=self.__dim)

        draw = to_tensor(self.__rng.random((2, self.__dim)))
        if self.__add_range is not None:
            add_draw = draw * (self.__add_range[1] - self.__add_range[0]) + self.__add_range[0]
            sym_axes = torch.argwhere(self.__symmetric_t).flatten()
            add_draw[1, sym_axes] = add_draw[0, sym_axes]
            margin_draw = None
            centre_offset_draw = None
        else:
            add_draw = None
            margin_draw = draw * (self.__margin_range[1] - self.__margin_range[0]) + self.__margin_range[0]
            sym_axes = torch.argwhere(self.__symmetric_t).flatten()
            margin_draw[1, sym_axes] = margin_draw[0, sym_axes]
            draw = to_tensor(self.__rng.random(self.__dim))
            centre_offset_draw = draw * (self.__centre_offset_range[1] - self.__centre_offset_range[0]) + self.__centre_offset_range[0]
        params = dict(
            add=add_draw.T.flatten() if add_draw is not None else None,
            centre=self.__centre,
            centre_offset=centre_offset_draw,
            margin=margin_draw.T.flatten() if margin_draw is not None else None,
        )
        return super().freeze(Pad, params)

    @property
    def params(self) -> TransformParams:
        return super().params(
            add=to_tuple(self.__add_range.permute(2, 1, 0).flatten()) if self.__add_range is not None else None,
            centre=self.__centre,
            centre_offset=to_tuple(self.__centre_offset_range.T.flatten()) if self.__centre_offset_range is not None else None,
            margin=to_tuple(self.__margin_range.permute(2, 1, 0).flatten()) if self.__margin_range is not None else None,
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
            add=to_tuple(self.__add_range.permute(2, 1, 0).flatten(), dp=3) if self.__add_range is not None else None,
            centre=to_tuple(self.__centre, dp=3) if self.__centre != 'image-centre' else "'image-centre'",
            centre_offset=to_tuple(self.__centre_offset, dp=3) if self.__centre_offset is not None else None,
            margin=to_tuple(self.__margin, dp=3) if self.__margin is not None else None,
            subtransform=subtransform,
            symmetric=self.__symmetric,
        )
