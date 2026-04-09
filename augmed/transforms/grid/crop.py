from __future__ import annotations

import numpy as np
import torch
from typing import List, Literal, Tuple

from ...typing import AffineMatrix, Indices, Number, Point, Points, SamplingGridTensor, Size
from ...utils.args import alias_kwargs, arg_to_list, expand_range_arg
from ...utils.conversion import to_return_format, to_tensor, to_tuple
from ...utils.geometry import affine_origin, affine_spacing, create_affine, fov, fov_centre, to_world_coords
from ...utils.misc import get_group_device
from ..identity import Identity
from .grid import GridTransform, RandomGridTransform

class Crop(GridTransform):
    @alias_kwargs([
        ('cc', 'centre'),
        ('cco', 'centre_offset'),
        ('cm', 'margin'),
        ('cr', 'remove'),
    ])
    def __init__(
        self,
        centre: Point | Literal['image-centre'] = 'image-centre',
        centre_offset: Number | Tuple[Number, ...] = 0.0,
        margin: Number | Tuple[Number | None, ...] | None = None,
        remove: Number | Tuple[Number | None, ...] | None = 20.0,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        assert remove is not None or (centre is not None and margin is not None), "Must specify either 'remove' or both 'centre' and 'margin'."
        if remove is not None:
            remove = expand_range_arg(remove, dim=self._dim)
            self.__remove = to_tensor(remove).reshape(self._dim, 2)
            self.__margin = None
            self.__centre = None
            self.__centre_offset = None
        else:
            margin = expand_range_arg(margin, dim=self._dim)
            self.__margin = to_tensor(margin).reshape(self._dim, 2)
            self.__centre = to_tuple(centre, broadcast=self._dim)   # Tensors can't store str types.
            assert len(self.__centre) == self._dim
            self.__centre_offset = to_tensor(centre_offset, broadcast=self._dim)
            assert len(self.__centre_offset) == self._dim
            self.__remove = None

        super().set_params(
            self.__class__.__name__,
            centre=self.__centre,
            centre_offset=self.__centre_offset,
            margin=self.__margin,
            remove=self.__remove,
        )

    def __str__(self) -> str:
        return super().__str__(
            self.__class__.__name__,
            centre=to_tuple(self.__centre, decimals=3),
            centre_offset=to_tuple(self.__centre_offset.flatten(), decimals=3) if self.__centre_offset is not None else None,
            margin=to_tuple(self.__margin.flatten(), decimals=3) if self.__margin is not None else None,
            remove=to_tuple(self.__remove.flatten(), decimals=3) if self.__remove is not None else None,
        )

    def transform_grid(
        self,
        grid: SamplingGridTensor,
        **kwargs,
        ) -> SamplingGridTensor:
        size, affine = grid
        if self.__remove is not None:
            # Get the current FOV in image coords.
            fov_min, fov_max = fov(size, affine=None)

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
            fov_c = fov_centre(size, affine=None)
            centre = [fov_c[i] if c == 'image-centre' else c for i, c in enumerate(self.__centre)]
            centre = to_tensor(centre, device=size.device)

            # Apply the crop centre offset.
            centre_offset = self.__centre_offset.to(size.device)
            if affine is not None:
                centre_offset /= affine_spacing(affine)
            centre += centre_offset

            # Get the margin sizes.
            margin_min = self.__margin[:, 0].to(size.device)
            margin_max = self.__margin[:, 1].to(size.device)
            if affine is not None:
                margin_min *= affine_spacing(affine)
                margin_max *= affine_spacing(affine)

            # Get the new FOV.
            fov_min = torch.clamp(centre - margin_min, 0)
            fov_max = torch.clamp(centre + margin_max, max=(size - 1))

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
        size_t = size_t.type(torch.int32)

        return size_t, affine_t

    def transform_points(
        self,
        points: Points | List[Points],
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
                # Get new FOV.
                assert size is not None, "Size must be provided for filtering off-grid points."
                assert affine is not None, "Affine must be provided for filtering off-grid points."
                size_t, affine_t = self.transform_grid((size, affine))
                spacing_t = affine_spacing(affine_t)
                origin_t = affine_origin(affine_t)

                # Get crop box.
                min_mm = origin_t
                max_mm = origin_t + size_t * spacing_t

                # Crop points.
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

class RandomCrop(RandomGridTransform):
    @alias_kwargs([
        ('cc', 'centre'),
        ('cco', 'centre_offset'),
        ('cm', 'margin'),
        ('cr', 'remove'),
        ('s', 'symmetric'),
    ])
    def __init__(
        self,
        # How many ways are there to define a crop?
        # 1. Removing an amount off each axis end ('remove').
        # 2. Cropping using a centre and margin ('centre', 'margin').
        # 3. Using defined values in image/world coordinates.
        # 4. TODO: Cropping around a label centre or boundary.
        # Must keep 'centre' and 'centre_offset' separate so we can specify image centre using 'image-centre'.
        centre: Point | np.ndarray | torch.Tensor | Literal['image-centre'] | None = None,
        centre_offset: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 0.0,
        margin: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = None,
        remove: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = 20,       # Currently the default.
        # Cropped amounts are the same at both ends of each axis.
        # This should be configured per axis really, for example we might want want symmetry
        # along the x-axis only.
        symmetric: bool | Tuple[bool, ...] = False,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        assert remove is not None or (centre is not None and margin is not None), "Must specify either 'remove' or both 'centre' and 'margin'."
        self.__centre = centre
        self.__centre_offset = centre_offset
        self.__margin = margin
        self.__remove = remove
        self.__symmetric = symmetric
        super().set_params(
            self.__class__.__name__,
            centre=self.__centre,
            centre_offset=self.__centre_offset,
            margin=self.__margin,
            remove=self.__remove,
            symmetric=self.__symmetric,
        )

    def freeze(self) -> Crop:
        # Expand the range args.
        # We do this now because 'set_dim' could be called after RandomCrop.__init__.
        symmetric = to_tensor(self.__symmetric, broadcast=self._dim)
        if self.__remove is not None: # Cropping by remove from each axis end.
            cr_vals_per_dim = 4
            remove_range = expand_range_arg(self.__remove, dim=self._dim, vals_per_dim=cr_vals_per_dim)
            assert len(remove_range) == cr_vals_per_dim * self._dim, f"Expected 'remove' of length {cr_vals_per_dim * self._dim}, got {len(remove_range)}."

            # Ensure crop ranges allow symmetry.
            for i, s in enumerate(symmetric):
                cr_axis_vals = remove_range[i * cr_vals_per_dim:(i + 1) * cr_vals_per_dim]
                if s and (cr_axis_vals[0] != cr_axis_vals[2] or cr_axis_vals[1] != cr_axis_vals[3]):
                    raise ValueError(f"Cannot create symmetric crops for axis {i} with crop ranges {cr_axis_vals}.")

            remove_range = to_tensor(remove_range).reshape(self._dim, 2, 2)
            centre = None
            centre_offset = None
            margin = None

        else:   # Cropping using centre point and margin.
            cmr_vals_per_dim = 4
            margin_range = expand_range_arg(self.__margin, dim=self._dim, vals_per_dim=cmr_vals_per_dim)
            assert len(margin_range) == cmr_vals_per_dim * self._dim, f"Expected 'margin' of length {cmr_vals_per_dim * self._dim}, got {len(margin_range)}."

            # Ensure crop margin ranges allow symmetry.
            for i, s in enumerate(symmetric):
                cmr_axis_vals = margin_range[i * cmr_vals_per_dim:(i + 1) * cmr_vals_per_dim]
                if s and (cmr_axis_vals[0] != cmr_axis_vals[2] or cmr_axis_vals[1] != cmr_axis_vals[3]):
                    raise ValueError(f"Cannot create symmetric crops for axis {i} with crop margin ranges {cmr_axis_vals}.")

            margin = to_tensor(margin_range).reshape(self._dim, 2, 2)
            centre = arg_to_list(centre, (int, float, str), broadcast=self._dim, iter_types=(np.ndarray, torch.Tensor))
            assert len(centre) == self._dim, f"Expected 'centre' of length {self._dim}, got {len(centre)}."
            centre_offset_range = expand_range_arg(centre_offset_range, dim=self._dim, negate_lower=True)
            assert len(centre_offset_range) == 2 * self._dim, f"Expected 'centre_offset_range' of length {2 * self._dim}, got {len(centre_offset_range)}."
            centre_offset_range = to_tensor(centre_offset_range).reshape(self._dim, 2)
            remove = None

        # Draw the crop parameters.
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random((self._dim, 2)))
        if remove_range is not None:
            remove_draw = (draw * (remove_range[:, :, 1] - remove_range[:, :, 0]) + remove_range[:, :, 0])
            # Copy lower end of axis for symmetric crops.
            sym_axes = torch.argwhere(symmetric).flatten()
            remove_draw[sym_axes, 1] = remove_draw[sym_axes, 0]
            margin_draw = None
            centre_offset_draw = None
        else:
            centre_offset_draw = (draw * (centre_offset_range[:, 1] - centre_offset_range[:, 0]) + centre_offset_range[:, 0])
            margin_draw = (draw * (margin[:, :, 1] - margin[:, :, 0]) + margin[:, :, 0])
            # Copy lower end of axis for symmetric crops.
            sym_axes = torch.argwhere(symmetric).flatten()
            margin_draw[sym_axes, 1] = margin_draw[sym_axes, 0]
            draw = to_tensor(self._rng.random(self._dim))
            remove_draw = None

        params = dict(
            centre=self.__centre,
            centre_offset=centre_offset_draw,
            margin=margin_draw.flatten() if margin_draw is not None else None,
            remove=remove_draw.flatten() if remove_draw is not None else None,
        )
        return super().freeze(Crop, params)

    def __str__(self) -> str:
        return super().__str__(
            self.__class__.__name__,
            centre=self.__centre,
            centre_offset=self.__centre_offset,
            margin=self.__margin,
            remove=self.__remove,
            symmetric=self.__symmetric,
        )
