import numpy as np
import torch
from typing import List, Literal, Tuple

from ...typing import AffineMatrix, Indices, Number, Points, SamplingGridTensor, Size
from ...utils.args import alias_kwargs, arg_to_list, expand_range_arg, to_tuple
from ...utils.conversion import to_return_format, to_tensor
from ...utils.geometry import fov, fov_centre, to_image_coords, to_world_coords
from ...utils.matrix import affine_origin, affine_spacing, create_affine
from ..identity import Identity, get_group_device
from .grid import GridTransform, RandomGridTransform

class Pad(GridTransform):
    @alias_kwargs([
        ('pa', 'add'),
        ('pc', 'centre'),
        ('pco', 'centre_offset'),
        ('pm', 'margin'),
    ])
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
            add_range = expand_range_arg(add, dim=self._dim)
            self.__add = to_tensor(add_range).reshape(self._dim, 2)
            self.__margin = None
            self.__centre = None
            self.__centre_offset = None
        else:
            margin_range = expand_range_arg(margin, dim=self._dim)
            self.__margin = to_tensor(margin_range).reshape(self._dim, 2)
            self.__centre = to_tuple(centre, broadcast=self._dim)   # Tensors can't store str types.
            assert len(self.__centre) == self._dim
            self.__centre_offset = to_tensor(centre_offset, broadcast=self._dim)
            assert len(self.__centre_offset) == self._dim

        super().set_params(
            self.__class__.__name__,
            add=self.__add,
            centre=self.__centre,
            centre_offset=self.__centre_offset,
            margin=self.__margin,
        )

    def __str__(self) -> str:
        return super().__str__(
            self.__class__.__name__,
            centre=to_tuple(self.__centre, decimals=3),
            centre_offset=to_tuple(self.__centre_offset.flatten(), decimals=3) if self.__centre_offset is not None else None,
            add=to_tuple(self.__add.flatten(), decimals=3) if self.__add is not None else None,
            margin=to_tuple(self.__margin.flatten(), decimals=3) if self.__margin is not None else None,
        )

    def transform_grid(
        self,
        grid: SamplingGridTensor,
        **kwargs,
        ) -> SamplingGridTensor:
        size, affine = grid
        if self.__add is not None:
            # Get the current FOV.
            fov_min, fov_max = fov(size, affine=None)

            # Get the amounts to add.
            add_min = self.__add[:, 0].to(size.device)
            add_max = self.__add[:, 1].to(size.device)
            if affine is not None:
                add_min /= affine_spacing(affine)
                add_max /= affine_spacing(affine)

            # Get the new FOV.
            fov_min = torch.clamp(fov_min - add_min, 0)
            fov_max = torch.clamp(fov_max + add_max, max=(size - 1))
        else:
            # Get pad centre.
            fov_c = fov_centre(size, affine=None)
            centre = [fov_c[i] if c == 'image-centre' else c for i, c in enumerate(self.__centre)]
            centre = to_tensor(centre, device=size.device)

            # Get centre offset.
            centre_offset = self.__centre_offset.to(size.device)
            if affine is not None:
                centre_offset /= affine_spacing(affine)
            centre += centre_offset

            # Get pad box.
            margin_min = self.__margin[:, 0].to(size.device)
            margin_max = self.__margin[:, 1].to(size.device)
            if affine is not None:
                margin_min /= affine_spacing(affine)
                margin_max /= affine_spacing(affine)

            # Truncate to true voxel coords.
            fov_min = torch.clamp(centre - margin_min, 0)
            fov_max = torch.clamp(centre + margin_max, max=(size - 1))

        # Get new size.
        size_t = fov_max - fov_min
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

    def transform_points(
        self,
        points: Points | List[Points],
        # Can a pad ever move points offgrid??
        affine: AffineMatrix | None = None,       # Required for some transforms, e.g. Rotate, to get centre of rotation.
        filter_offgrid: bool = True,
        # grid: SamplingGrid | None = None,   # Required for filtering off-grid points and some transforms, e.g. Rotate.
        return_filtered: bool = False,
        size: Size | None = None,           # Required for filtering off-grid points.
        **kwargs,
        ) -> Points | List[Points | Indices | List[Indices]]:
        pointses, points_was_single = arg_to_list(points, (np.ndarray, torch.Tensor), return_expanded=True)
        device = get_group_device(pointses, device=self._device)
        return_types = [type(p) for p in pointses]
        pointses = [to_tensor(p, device=device, dtype=torch.float32) for p in pointses]
        size = to_tensor(size, device=device, dtype=torch.int32)
        affine = to_tensor(affine, device=device, dtype=torch.float32)

        points_ts = []
        indiceses = []
        for p in pointses:
            points_t = p

            # Forward transformed points could end up off-screen and should be filtered.
            # However, we need to know which points are returned for loss calc for example.
            if filter_offgrid:
                # Get new grid.
                assert size is not None, "Size must be provided for filtering off-grid points."
                assert affine is not None, "Affine must be provided for filtering off-grid points."
                size_t, affine_t = self.transform_grid((size, affine))
                spacing_t = affine_spacing(affine_t)
                origin_t = affine_origin(affine_t)

                # Get pad box.
                min_mm = origin_t
                max_mm = origin_t + size_t * spacing_t

                # Pad points.
                bounds_mm = torch.stack([min_mm, max_mm]).to(device)
                to_keep = (points >= bounds_mm[0]) & (points < bounds_mm[1])
                to_keep = to_keep.all(axis=1)
                points_t = points_t[to_keep]
                indices = torch.where(~to_keep)[0].type(torch.int32)
                indiceses.append(indices)

            points_ts.append(points_t)

        # Convert to return format.
        other_data = []
        if filter_offgrid and return_filtered:
            indiceses = to_return_format(indiceses, return_single=True, return_types=return_types)
            other_data.append(indiceses)
        results = to_return_format(points_ts, other_data=other_data, return_single=points_was_single, return_types=return_types)
        return results

class RandomPad(RandomGridTransform):
    @alias_kwargs([
        ('a', 'add'),
        ('pc', 'centre'),
        ('pco', 'centre_offset'),
        ('pm', 'margin'),
        ('s', 'symmetric'),
    ])
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
        self.__symmetric = to_tensor(symmetric, broadcast=self._dim)
        if add is not None:
            # Handle pad from outside case.
            cr_vals_per_dim = 4
            add_range = expand_range_arg(add, dim=self._dim, vals_per_dim=cr_vals_per_dim)
            assert len(add_range) == cr_vals_per_dim * self._dim, f"Expected 'add' of length {cr_vals_per_dim * self._dim}, got {len(add_range)}."

            # Ensure pad ranges allow symmetry.
            for i, s in enumerate(self.__symmetric):
                cr_axis_vals = add_range[i * cr_vals_per_dim:(i + 1) * cr_vals_per_dim]
                if s and (cr_axis_vals[0] != cr_axis_vals[2] or cr_axis_vals[1] != cr_axis_vals[3]):
                    raise ValueError(f"Cannot create symmetric pads for axis {i} with pad ranges {cr_axis_vals}.")

            self.__add_range = to_tensor(add_range).reshape(self._dim, 2, 2)
            # Should we zero out things that aren't relevant?
            self.__centre = None
            self.__margin_range = None
            self.__centre_offset_range = None
        else:
            # Handle pad from centre point and margin case.
            cmr_vals_per_dim = 4
            margin_range = expand_range_arg(margin, dim=self._dim, vals_per_dim=cmr_vals_per_dim)
            assert len(margin_range) == cmr_vals_per_dim * self._dim, f"Expected 'margin' of length {cmr_vals_per_dim * self._dim}, got {len(margin_range)}."

            # Ensure pad margin ranges allow symmetry.
            for i, s in enumerate(self.__symmetric):
                cmr_axis_vals = margin_range[i * cmr_vals_per_dim:(i + 1) * cmr_vals_per_dim]
                if s and (cmr_axis_vals[0] != cmr_axis_vals[2] or cmr_axis_vals[1] != cmr_axis_vals[3]):
                    raise ValueError(f"Cannot create symmetric pads for axis {i} with pad margin ranges {cmr_axis_vals}.")

            self.__margin_range = to_tensor(margin_range).reshape(self._dim, 2, 2)
            centre = arg_to_list(centre, (int, float, str), broadcast=self._dim)
            assert len(centre) == self._dim, f"Expected 'centre' of length {self._dim}, got {len(centre)}."
            self.__centre = centre  # Can't be tensor as might have 'image-centre' str.
            centre_offset_range = expand_range_arg(centre_offset, dim=self._dim, negate_lower=True)
            assert len(centre_offset_range) == 2 * self._dim, f"Expected 'centre_offset' of length {2 * self._dim}, got {len(centre_offset_range)}."
            self.__centre_offset_range = to_tensor(centre_offset_range).reshape(self._dim, 2)
            self.__add_range = None

        super().set_params(
            self.__class__.__name__,
            add=self.__add_range,
            centre=self.__centre,
            centre_offset=self.__centre_offset_range,
            margin=self.__margin_range,
        )

    def freeze(self) -> 'Pad':
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random((self._dim, 2)))
        if self.__add_range is not None:
            add_draw = (draw * (self.__add_range[:, :, 1] - self.__add_range[:, :, 0]) + self.__add_range[:, :, 0])
            # Copy lower end of axis for symmetric pads.
            sym_axes = torch.argwhere(self.__symmetric).flatten()
            add_draw[sym_axes, 1] = add_draw[sym_axes, 0]
            margin_draw = None
            centre_offset_draw = None
        else:
            add_draw = None
            margin_draw = (draw * (self.__margin_range[:, :, 1] - self.__margin_range[:, :, 0]) + self.__margin_range[:, :, 0])
            # Copy lower end of axis for symmetric pads.
            sym_axes = torch.argwhere(self.__symmetric).flatten()
            margin_draw[sym_axes, 1] = margin_draw[sym_axes, 0]
            draw = to_tensor(self._rng.random(self._dim))
            centre_offset_draw = (draw * (self.__centre_offset_range[:, 1] - self.__centre_offset_range[:, 0]) + self.__centre_offset_range[:, 0])

        params = dict(
            add=add_draw,
            centre=self.__centre,
            centre_offset=centre_offset_draw,
            margin=margin_draw,
        )
        return super().freeze(Pad, params)

    def __str__(self) -> str:
        return super().__str__(
            self.__class__.__name__,
            add=to_tuple(self.__add_range.flatten(), decimals=3) if self.__add_range is not None else None,
            centre=to_tuple(self.__centre, decimals=3),
            centre_offset=to_tuple(self.__centre_offset_range.flatten(), decimals=3) if self.__centre_offset_range is not None else None,
            margin=to_tuple(self.__margin_range.flatten(), decimals=3) if self.__margin_range is not None else None,
        )
