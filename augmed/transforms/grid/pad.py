from typing import *

from ...typing import *
from ...utils.args import alias_kwargs, arg_to_list, expand_range_arg
from ...utils.conversion import to_tensor
from ...utils.geometry import fov, fov_centre
from ...utils.matrix import affine_origin, affine_spacing, create_affine
from ..identity import Identity
from .grid import RandomGridTransform, GridTransform

class Pad(GridTransform):
    @alias_kwargs([
        ('pa', 'pad_add'),
        ('pc', 'pad_centre'),
        ('pco', 'pad_centre_offset'),
        ('pm', 'pad_margin'),
    ])
    def __init__(
        self,
        pad_add: Number | Tuple[Number, ...] | None = None,
        pad_centre: Number | Literal['image-centre'] | Tuple[Number | Literal['image-centre'], ...] | None = 'image-centre',
        pad_centre_offset: Number | Tuple[Number, ...] | None = 0.0,
        pad_margin: Number | Tuple[Number, ...] | None = None,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        assert pad_add is not None or (pad_centre is not None and pad_margin is not None)
        if pad_add is not None:
            pad_add_range = expand_range_arg(pad_add, dim=self._dim)
            self.__pad_add = to_tensor(pad_add_range).reshape(self._dim, 2)
            self.__pad_margin = None
            self.__pad_centre = None
            self.__pad_centre_offset = None
        else:
            pad_margin_range = expand_range_arg(pad_margin, dim=self._dim)
            self.__pad_margin = to_tensor(pad_margin_range).reshape(self._dim, 2)
            self.__pad_centre = to_tuple(pad_centre, broadcast=self._dim)   # Tensors can't store str types.
            assert len(self.__pad_centre) == self._dim
            self.__pad_centre_offset = to_tensor(pad_centre_offset, broadcast=self._dim)
            assert len(self.__pad_centre_offset) == self._dim

        self._params = dict(
            type=self.__class__.__name__,
            dim=self._dim,
            pad_add=self.__pad_add,
            pad_centre=self.__pad_centre,
            pad_centre_offset=self.__pad_centre_offset,
            pad_margin=self.__pad_margin,
        )

    def __str__(self) -> str:
        params = dict(
            pad=to_tuple(self.__pad.flatten(), decimals=3) if self.__pad is not None else None,
            pad_margin=to_tuple(self.__pad_margin.flatten(), decimals=3) if self.__pad_margin is not None else None,
            centre=to_tuple(self.__centre, decimals=3),
            centre_offset=to_tuple(self.__centre_offset.flatten(), decimals=3) if self.__centre_offset is not None else None,
        )
        return super().__str__(self.__class__.__name__, params)

    def transform_grid(
        self,
        grid: SamplingGridTensor,
        **kwargs,
        ) -> SamplingGridTensor:
        size, affine = grid
        if self.__pad_add is not None:
            # Get the current FOV.
            fov_min, fov_max = fov(size, affine=affine)

            # Get the amounts to add.
            pad_add_min = self.__pad_add[:, 0].to(size.device)
            pad_add_max = self.__pad_add[:, 1].to(size.device)

            if affine is not None:
                # Convert FOV from mm -> vox.
                spacing = affine_spacing(affine)
                origin = affine_origin(affine) 
                fov_min = torch.round((fov_min - origin) / spacing)
                fov_max = torch.round((fov_max - origin) / spacing)

                # Convert 'pad_add' from mm -> vox.
                spacing = affine_spacing(affine)
                origin = affine_origin(affine)
                pad_add_min = torch.round(pad_add_min / spacing)
                pad_add_max = torch.round(pad_add_max / spacing)

            # Get the new FOV.
            pad_min_vox = fov_min - pad_add_min
            pad_max_vox = fov_max + pad_add_max
        else:
            # Get pad centre.
            fov_c = fov_centre(size, affine=affine)
            centre_mm = [fov_c[i] if c == 'image-centre' else c for i, c in enumerate(self.__pad_centre)]
            centre_mm = to_tensor(centre_mm, device=size.device)
            centre_mm = centre_mm + self.__pad_centre_offset.to(size.device)

            # Get pad box.
            pad_margin_min_mm = self.__pad_margin[:, 0].to(size.device)
            pad_margin_max_mm = self.__pad_margin[:, 1].to(size.device)
            if not self._use_image_coords:
                # Convert from image -> patient coords.
                pad_margin_min_mm = (spacing * pad_margin_min_mm).type(torch.float32)
                pad_margin_max_mm = (spacing * pad_margin_max_mm).type(torch.float32)
            pad_min_mm = centre_mm - pad_margin_min_mm
            pad_max_mm = centre_mm + pad_margin_max_mm

            # Convert to voxels.
            pad_min_vox = torch.round((pad_min_mm - origin) / spacing).type(torch.int32)
            pad_max_vox = torch.round((pad_max_mm - origin) / spacing).type(torch.int32)

            # Truncate to true voxel coords.
            pad_min_vox = torch.clamp(pad_min_vox, 0)
            pad_max_vox = torch.clamp(pad_max_vox, max=(size - 1))

        # Get new size.
        size_t = pad_max_vox - pad_min_vox
        size_t = size_t.clamp(0)

        # Get new affine.
        if affine is not None:
            # Crop doesn't change voxel spacing, but it does change the position of the 0th voxel in world coordinates.
            spacing_t = affine_spacing(affine)
            origin = affine_origin(affine)
            origin_t = (pad_min_vox * spacing_t) + origin
            affine_t = create_affine(spacing_t, origin_t)
        else:
            affine_t = None

        # Convert types.
        size_t = size_t.type(torch.int32)

        return size_t, affine_t

    def transform_points(
        self,
        points: Points,
        # Can a pad ever move points offgrid??
        filter_offgrid: bool = True,
        grid: SamplingGrid | None = None,   # Required for 'image-centre' pad centre.
        return_filtered: bool = False,
        **kwargs,
        ) -> Points:
        if isinstance(points, np.ndarray):
            points = to_tensor(points)
            return_type = 'numpy'
        else:
            return_type = 'torch'
        size, affine = grid if grid is not None else (None, None)
        size = to_tensor(size, device=points.device, dtype=points.dtype)
        affine = to_tensor(affine, device=points.device, dtype=points.dtype)

        # Forward transformed points could end up off-screen and should be filtered.
        # However, we need to know which points are returned for loss calc for example.
        if filter_offgrid:
            assert size is not None
            assert affine is not None

            # Get new grid.
            size_t, affine_t = self.transform_grid(size, affine=affine)
            spacing_t = affine_spacing(affine_t)
            origin_t = affine_origin(affine_t)

            # Get pad box.
            pad_min_mm = origin_t
            pad_max_mm = origin_t + size_t * spacing_t

            # Pad points.
            pad_mm = torch.stack([pad_min_mm, pad_max_mm]).to(points.device)
            print(pad_mm)
            to_keep = (points >= pad_mm[0]) & (points < pad_mm[1])
            print(to_keep)
            to_keep = to_keep.all(axis=1)
            points_t = points[to_keep]
            indices = torch.where(to_keep)[0]
            if return_type == 'numpy':
                points_t, indices = points_t.numpy(), indices.numpy()
            if return_filtered:
                return points_t, indices
            else:
                return points_t
        else:
            if return_type == 'numpy':
                points = points.numpy()
            return points

class RandomPad(RandomGridTransform):
    @alias_kwargs([
        ('pa', 'pad_add'),
        ('pc', 'pad_centre'),
        ('pco', 'pad_centre_offset'),
        ('pm', 'pad_margin'),
        ('s', 'symmetric'),
    ])
    def __init__(
        self,
        # How many ways are there to define a pad?
        # 1. Removing an amount off each axis end ('pad_remove').
        # 2. Padding using a centre and margin ('pad_centre', 'pad_margin').
        # 3. Using defined values in image/world coordinates.
        # 4. TODO: Padding around a label centre or boundary.
        pad_add: Number | Tuple[Number, ...] | None = None,
        pad_centre: Number | Literal['image-centre'] | Tuple[Number | Literal['image-centre'], ...] | None = 'image-centre',
        pad_centre_offset: Number | Tuple[Number, ...] | None = 0.0,
        pad_margin: Number | Tuple[Number, ...] | None = None,
        # padded amounts are the same at both ends of each axis.
        # This should be configured per axis really, for example we might want want symmetry
        # along the x-axis only.
        symmetric: bool | Tuple[bool, ...] = False,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        assert pad_add is not None or (pad_centre is not None and pad_margin is not None), "Must specify either 'pad_add' or both 'pad_centre' and 'pad_margin'."
        self.__symmetric = to_tensor(symmetric, broadcast=self._dim)
        if pad_add is not None:
            # Handle pad from outside case.
            cr_vals_per_dim = 4
            pad_add_range = expand_range_arg(pad_add, dim=self._dim, vals_per_dim=cr_vals_per_dim)
            assert len(pad_add_range) == cr_vals_per_dim * self._dim, f"Expected 'pad_add' of length {cr_vals_per_dim * self._dim}, got {len(pad_add_range)}."

            # Ensure pad ranges allow symmetry.
            for i, s in enumerate(self.__symmetric):
                cr_axis_vals = pad_add_range[i * cr_vals_per_dim:(i + 1) * cr_vals_per_dim]
                if s and (cr_axis_vals[0] != cr_axis_vals[2] or cr_axis_vals[1] != cr_axis_vals[3]):
                    raise ValueError(f"Cannot create symmetric pads for axis {i} with pad ranges {cr_axis_vals}.")

            self.__pad_add_range = to_tensor(pad_add_range).reshape(self._dim, 2, 2)
            # Should we zero out things that aren't relevant?
            self.__pad_centre = None
            self.__pad_margin_range = None
            self.__pad_centre_offset_range = None
        else:
            # Handle pad from centre point and margin case.
            cmr_vals_per_dim = 4
            pad_margin_range = expand_range_arg(pad_margin, dim=self._dim, vals_per_dim=cmr_vals_per_dim)
            assert len(pad_margin_range) == cmr_vals_per_dim * self._dim, f"Expected 'pad_margin' of length {cmr_vals_per_dim * self._dim}, got {len(pad_margin_range)}."

            # Ensure pad margin ranges allow symmetry.
            for i, s in enumerate(self.__symmetric):
                cmr_axis_vals = pad_margin_range[i * cmr_vals_per_dim:(i + 1) * cmr_vals_per_dim]
                if s and (cmr_axis_vals[0] != cmr_axis_vals[2] or cmr_axis_vals[1] != cmr_axis_vals[3]):
                    raise ValueError(f"Cannot create symmetric pads for axis {i} with pad margin ranges {cmr_axis_vals}.")

            self.__pad_margin_range = to_tensor(pad_margin_range).reshape(self._dim, 2, 2)
            pad_centre = arg_to_list(pad_centre, (int, float, str), broadcast=self._dim)
            assert len(pad_centre) == self._dim, f"Expected 'pad_centre' of length {self._dim}, got {len(pad_centre)}."
            self.__pad_centre = pad_centre  # Can't be tensor as might have 'image-centre' str.
            pad_centre_offset_range = expand_range_arg(pad_centre_offset, dim=self._dim, negate_lower=True)
            assert len(pad_centre_offset_range) == 2 * self._dim, f"Expected 'pad_centre_offset' of length {2 * self._dim}, got {len(pad_centre_offset_range)}."
            self.__pad_centre_offset_range = to_tensor(centre_offset_range).reshape(self._dim, 2)
            self.__pad_add_range = None

        self._params = dict(
            type=self.__class__.__name__,
            p=self._p,
            dim=self._dim,
            pad_add=self.__pad_add_range,
            pad_centre=self.__pad_centre,
            pad_centre_offset=self.__pad_centre_offset_range,
            pad_margin=self.__pad_margin_range,
        )

    def freeze(self) -> 'Pad':
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random((self._dim, 2)))
        if self.__pad_add_range is not None:
            pad_add_draw = (draw * (self.__pad_add_range[:, :, 1] - self.__pad_add_range[:, :, 0]) + self.__pad_add_range[:, :, 0])
            # Copy lower end of axis for symmetric pads.
            sym_axes = torch.argwhere(self.__symmetric).flatten()
            pad_add_draw[sym_axes, 1] = pad_add_draw[sym_axes, 0]
            pad_margin_draw = None
            centre_offset_draw = None
        else:
            pad_draw = None
            pad_margin_draw = (draw * (self.__pad_margin_range[:, :, 1] - self.__pad_margin_range[:, :, 0]) + self.__pad_margin_range[:, :, 0])
            # Copy lower end of axis for symmetric pads.
            sym_axes = torch.argwhere(self.__symmetric).flatten()
            pad_margin_draw[sym_axes, 1] = pad_margin_draw[sym_axes, 0]
            draw = to_tensor(self._rng.random(self._dim))
            centre_offset_draw = (draw * (self.__pad_centre_offset_range[:, 1] - self.__pad_centre_offset_range[:, 0]) + self.__pad_centre_offset_range[:, 0])

        params = dict(
            pad=pad_draw,
            pad_margin=pad_margin_draw,
            pad_centre=self.__pad_centre,
            pad_centre_offset=centre_offset_draw,
        )
        return super().freeze(Pad, params)

    def __str__(self) -> str:
        params = dict(
            pad_add=to_tuple(self.__pad_add_range.flatten(), decimals=3) if self.__pad_add_range is not None else None,
            pad_centre=to_tuple(self.__pad_centre, decimals=3),
            pad_centre_offset=to_tuple(self.__pad_centre_offset_range.flatten(), decimals=3) if self.__pad_centre_offset_range is not None else None,
            pad_margin=to_tuple(self.__pad_margin_range.flatten(), decimals=3) if self.__pad_margin_range is not None else None,
        )
        return super().__str__(self.__class__.__name__, params)
