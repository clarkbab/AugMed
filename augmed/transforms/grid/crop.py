import numpy as np
import torch
from typing import List, Literal, Tuple

from ...typing import AffineMatrix, Indices, Number, Point, Points, SamplingGridTensor, Size
from ...utils.args import alias_kwargs, arg_to_list, expand_range_arg
from ...utils.conversion import to_return_format, to_tensor, to_tuple
from ...utils.geometry import fov, fov_centre
from ...utils.matrix import affine_origin, affine_spacing, create_affine
from ..identity import Identity, get_group_device
from .grid import GridTransform, RandomGridTransform

# TODO: Handle None types for crop margin - indicating no removal on that axis/end.
class RandomCrop(RandomGridTransform):
    @alias_kwargs([
        ('cc', 'crop_centre'),
        ('cco', 'crop_centre_offset'),
        ('cm', 'crop_margin'),
        ('cr', 'crop_remove'),
        ('s', 'symmetric'),
    ])
    def __init__(
        self,
        # How many ways are there to define a crop?
        # 1. Removing an amount off each axis end ('crop_remove').
        # 2. Cropping using a centre and margin ('crop_centre', 'crop_margin').
        # 3. Using defined values in image/world coordinates.
        # 4. TODO: Cropping around a label centre or boundary.
        # Must keep 'crop_centre' and 'crop_centre_offset' separate so we can specify image centre using 'image-centre'.
        crop_centre: Point | np.ndarray | torch.Tensor | Literal['image-centre'] | None = None,
        crop_centre_offset: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 0.0,
        crop_margin: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = None,
        crop_remove: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = 50,       # Currently the default.
        # Cropped amounts are the same at both ends of each axis.
        # This should be configured per axis really, for example we might want want symmetry
        # along the x-axis only.
        symmetric: bool | Tuple[bool, ...] = False,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        assert crop_remove is not None or (crop_centre is not None and crop_margin is not None), "Must specify either 'crop_remove' or both 'crop_centre' and 'crop_margin'."
        self.__symmetric = to_tensor(symmetric, broadcast=self._dim)
        if crop_remove is not None: # Cropping by remove from each axis end.
            cr_vals_per_dim = 4
            crop_remove_range = expand_range_arg(crop_remove, dim=self._dim, vals_per_dim=cr_vals_per_dim)
            assert len(crop_remove_range) == cr_vals_per_dim * self._dim, f"Expected 'crop_remove' of length {cr_vals_per_dim * self._dim}, got {len(crop_remove_range)}."

            # Ensure crop ranges allow symmetry.
            for i, s in enumerate(self.__symmetric):
                cr_axis_vals = crop_remove_range[i * cr_vals_per_dim:(i + 1) * cr_vals_per_dim]
                if s and (cr_axis_vals[0] != cr_axis_vals[2] or cr_axis_vals[1] != cr_axis_vals[3]):
                    raise ValueError(f"Cannot create symmetric crops for axis {i} with crop ranges {cr_axis_vals}.")

            self.__crop_remove = to_tensor(crop_remove_range).reshape(self._dim, 2, 2)
            self.__crop_margin = None
            self.__crop_centre = None
            self.__crop_centre_offset = None
        else:   # Cropping using centre point and margin.
            cmr_vals_per_dim = 4
            crop_margin_range = expand_range_arg(crop_margin, dim=self._dim, vals_per_dim=cmr_vals_per_dim)
            assert len(crop_margin_range) == cmr_vals_per_dim * self._dim, f"Expected 'crop_margin' of length {cmr_vals_per_dim * self._dim}, got {len(crop_margin_range)}."

            # Ensure crop margin ranges allow symmetry.
            for i, s in enumerate(self.__symmetric):
                cmr_axis_vals = crop_margin_range[i * cmr_vals_per_dim:(i + 1) * cmr_vals_per_dim]
                if s and (cmr_axis_vals[0] != cmr_axis_vals[2] or cmr_axis_vals[1] != cmr_axis_vals[3]):
                    raise ValueError(f"Cannot create symmetric crops for axis {i} with crop margin ranges {cmr_axis_vals}.")

            self.__crop_margin = to_tensor(crop_margin_range).reshape(self._dim, 2, 2)
            crop_centre = arg_to_list(crop_centre, (int, float, str), broadcast=self._dim, iter_types=(np.ndarray, torch.Tensor))
            assert len(crop_centre) == self._dim, f"Expected 'crop_centre' of length {self._dim}, got {len(crop_centre)}."
            self.__crop_centre = crop_centre  # Can't be tensor as might have 'centre' str.
            centre_offset_range = expand_range_arg(centre_offset_range, dim=self._dim, negate_lower=True)
            assert len(centre_offset_range) == 2 * self._dim, f"Expected 'centre_offset_range' of length {2 * self._dim}, got {len(centre_offset_range)}."
            self.__crop_centre_offset = to_tensor(centre_offset_range).reshape(self._dim, 2)
            self.__crop_remove = None

        self._params = dict(
            crop_centre=self.__crop_centre,
            crop_centre_offset=self.__crop_centre_offset,
            crop_margin_range=self.__crop_margin,
            crop_remove=self.__crop_remove,
            dim=self._dim,
            p=self._p,
            type=self.__class__.__name__,
        )

    def freeze(self) -> 'Crop':
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random((self._dim, 2)))
        if self.__crop_remove is not None:
            crop_remove_draw = (draw * (self.__crop_remove[:, :, 1] - self.__crop_remove[:, :, 0]) + self.__crop_remove[:, :, 0])
            # Copy lower end of axis for symmetric crops.
            sym_axes = torch.argwhere(self.__symmetric).flatten()
            crop_remove_draw[sym_axes, 1] = crop_remove_draw[sym_axes, 0]
            crop_margin_draw = None
            crop_centre_offset_draw = None
        else:
            crop_centre_offset_draw = (draw * (self.__crop_centre_offset[:, 1] - self.__crop_centre_offset[:, 0]) + self.__crop_centre_offset[:, 0])
            crop_margin_draw = (draw * (self.__crop_margin[:, :, 1] - self.__crop_margin[:, :, 0]) + self.__crop_margin[:, :, 0])
            # Copy lower end of axis for symmetric crops.
            sym_axes = torch.argwhere(self.__symmetric).flatten()
            crop_margin_draw[sym_axes, 1] = crop_margin_draw[sym_axes, 0]
            draw = to_tensor(self._rng.random(self._dim))
            crop_remove_draw = None

        params = dict(
            crop_centre=self.__crop_centre,
            crop_centre_offset=crop_centre_offset_draw,
            crop_margin=crop_margin_draw,
            crop_remove=crop_remove_draw,
        )
        return super().freeze(Crop, params)

    def __str__(self) -> str:
        params = dict(
            crop_centre=to_tuple(self.__crop_centre, decimals=3),
            crop_centre_offset=to_tuple(self.__crop_centre_offset.flatten(), decimals=3) if self.__crop_centre_offset is not None else None,
            crop_margin=to_tuple(self.__crop_margin.flatten(), decimals=3) if self.__crop_margin is not None else None,
            crop_remove=to_tuple(self.__crop_remove.flatten(), decimals=3) if self.__crop_remove is not None else None,
            symmetric=to_tuple(self.__symmetric),
        )
        return super().__str__(self.__class__.__name__, params)

class Crop(GridTransform):
    @alias_kwargs([
        ('cc', 'crop_centre'),
        ('cco', 'crop_centre_offset'),
        ('cm', 'crop_margin'),
        ('cr', 'crop_remove'),
    ])
    def __init__(
        self,
        crop_centre: Point | Literal['image-centre'] = 'image-centre',
        crop_centre_offset: Number | Tuple[Number, ...] = 0.0,
        crop_margin: Number | Tuple[Number | None, ...] | None = None,
        crop_remove: Number | Tuple[Number | None, ...] | None = 50.0,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        assert crop_remove is not None or (crop_centre is not None and crop_margin is not None), "Must specify either 'crop_remove' or both 'crop_centre' and 'crop_margin'."
        if crop_remove is not None:
            crop_remove = expand_range_arg(crop_remove, dim=self._dim)
            self.__crop_remove = to_tensor(crop_remove).reshape(self._dim, 2)
            self.__crop_margin = None
            self.__crop_centre = None
            self.__crop_centre_offset = None
        else:
            crop_margin = expand_range_arg(crop_margin, dim=self._dim)
            self.__crop_margin = to_tensor(crop_margin).reshape(self._dim, 2)
            self.__crop_centre = to_tuple(crop_centre, broadcast=self._dim)   # Tensors can't store str types.
            assert len(self.__crop_centre) == self._dim
            self.__crop_centre_offset = to_tensor(crop_centre_offset, broadcast=self._dim)
            assert len(self.__crop_centre_offset) == self._dim
            self.__crop_remove = None

        self._params = dict(
            crop_centre=self.__crop_centre,
            crop_centre_offset=self.__crop_centre_offset,
            crop_margin=self.__crop_margin,
            crop_remove=self.__crop_remove,
            dim=self._dim,
            type=self.__class__.__name__,
        )

    def __str__(self) -> str:
        params = dict(
            crop_centre=to_tuple(self.__crop_centre, decimals=3),
            crop_centre_offset=to_tuple(self.__crop_centre_offset.flatten(), decimals=3) if self.__crop_centre_offset is not None else None,
            crop_margin=to_tuple(self.__crop_margin.flatten(), decimals=3) if self.__crop_margin is not None else None,
            crop_remove=to_tuple(self.__crop_remove.flatten(), decimals=3) if self.__crop_remove is not None else None,
        )
        return super().__str__(self.__class__.__name__, params)

    def transform_grid(
        self,
        grid: SamplingGridTensor,
        **kwargs,
        ) -> SamplingGridTensor:
        size, affine = grid
        if self.__crop_remove is not None:
            # Get the current FOV.
            fov_min, fov_max = fov(size, affine=affine)

            # Get the amounts to remove.
            crop_remove_min = self.__crop_remove[:, 0].to(size.device)
            crop_remove_max = self.__crop_remove[:, 1].to(size.device)

            if affine is not None:
                # Convert FOV from mm -> vox.
                fov_min = to_image_coords(fov_min, affine)
                fov_max = to_image_coords(fov_max, affine)

                # Convert 'crop_remove' from mm -> vox.
                crop_remove_min = to_image_coords(crop_remove_min, affine)
                crop_remove_max = to_image_coords(crop_remove_max, affine)

            # Get the new FOV.
            crop_min_vox = fov_min + crop_remove_min
            crop_max_vox = fov_max - crop_remove_max
        else:
            # Get the crop centre.
            fov_c = fov_centre(size, affine=affine)
            centre = [fov_c[i] if c == 'image-centre' else c for i, c in enumerate(self.__crop_centre)]
            centre = to_tensor(centre, device=size.device)
            centre = centre + self.__crop_centre_offset.to(size.device)

            # Get the margin sizes.
            crop_margin_min = self.__crop_margin[:, 0].to(size.device)
            crop_margin_max = self.__crop_margin[:, 1].to(size.device)

            if affine is not None:
                # Convert the crop centre from mm -> vox.
                center = to_image_coords(centre, affine)

                # Convert the crop margins from mm -> vox.
                crop_margin_min = (spacing * crop_margin_min)
                crop_margin_max = (spacing * crop_margin_max)

            # Get the new FOV.
            crop_min_vox = centre - crop_margin_min
            crop_max_vox = centre + crop_margin_max

            # Truncate to true voxel coords.
            crop_min_vox = torch.clamp(crop_min_vox, 0)
            crop_max_vox = torch.clamp(crop_max_vox, max=(size - 1))

        # Get new size.
        size_t = crop_max_vox - crop_min_vox
        size_t = size_t.clamp(0)

        # Check result.
        if torch.any(size_t == 0):
            raise ValueError(f"{self} would create image with size zero along one or more axes (size={to_tuple(size_t)}).")

        # Get new affine.
        if affine is not None:
            # Crop doesn't change voxel spacing, but it does change the position of the 0th voxel in world coordinates.
            spacing_t = affine_spacing(affine)
            origin = affine_origin(affine)
            origin_t = (crop_min_vox * spacing) + origin
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
                crop_min_mm = origin_t
                crop_max_mm = origin_t + size_t * spacing_t

                # Crop points.
                crop_mm = torch.stack([crop_min_mm, crop_max_mm]).to(device)
                to_keep = (points >= crop_mm[0]) & (points < crop_mm[1])
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
