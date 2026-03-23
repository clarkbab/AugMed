import numpy as np
import torch
import torch
from typing import List, Literal, Tuple

from ....typing import Affine, AffineTensor, Indices, Number, Point, Points, PointsTensor, SamplingGrid, Size
from ....utils.args import alias_kwargs, arg_to_list, expand_range_arg
from ....utils.conversion import to_tensor, to_tuple
from ....utils.geometry import fov_centre
from ....utils.matrix import create_eye, create_rotation, create_scaling, create_translation
from ..spatial import RandomSpatialTransform, SpatialTransform

# Flip, Rotation, Translation (and others) should probably subclass this.
class Affine(SpatialTransform):
    @alias_kwargs([
        ('r', 'rotation'),
        ('rc', 'rotation_centre'),
        ('s', 'scaling'),
        ('sc', 'scaling_centre'),
        ('t', 'translation'),
    ])
    def __init__(
        self,
        rotation: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = None,
        rotation_centre: Point | Literal['image-centre'] = 'image-centre',
        scaling: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = None,
        scaling_centre: Point | Literal['image-centre'] = 'image-centre',
        translation: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = None,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        assert rotation is not None or scaling is not None or translation is not None, "At least one of 'rotation', 'scaling', or 'translation' must be specified for Affine transform."
        if rotation is not None:
            rotation = arg_to_list(rotation, (int, float, None), broadcast=self._dim, iter_types=(np.ndarray, torch.Tensor))
            assert len(rotation) == self._dim, f"Expected 'rotation' of length {self._dim} for dim={self._dim}, got {len(rotation)}."
            self._rotation = to_tensor(rotation)
            self._rotation_rad = torch.deg2rad(self._rotation) if rotation is not None else None
        else:
            self._rotation = None
            self._rotation_rad = None
        self._rotation_centre = 'image-centre' if rotation_centre == 'image-centre' else to_tensor(rotation_centre)
        if scaling is not None:
            scaling = arg_to_list(scaling, (int, float, None), broadcast=self._dim, iter_types=(np.ndarray, torch.Tensor))
            assert len(scaling) == self._dim, f"Expected 'scaling' of length {self._dim} for dim={self._dim}, got {len(scaling)}."
            self._scaling = to_tensor(scaling)
            if torch.any(self._scaling == 0):
                raise ValueError(f"scaling must be non-zero, got: {scaling}.")
        else:
            self._scaling = None
        self._scaling_centre = 'image-centre' if scaling_centre == 'image-centre' else to_tensor(scaling_centre)
        if translation is not None:
            translation = arg_to_list(translation, (int, float, None), broadcast=self._dim, iter_types=(np.ndarray, torch.Tensor))
            assert len(translation) == self._dim, f"Expected 'translation' of length {self._dim} for dim={self._dim}, got {len(translation)}."
            self._translation = to_tensor(translation)
        else:
            self._translation = None
        self.__create_transforms()
        self._params = dict(
            backward_rotation_matrix=self._backward_rotation_matrix,
            backward_scaling_matrix=self._backward_scaling_matrix,
            backward_translation_matrix=self._backward_translation_matrix,
            dim=self._dim,
            rotation=self._rotation,
            rotation_centre=self._rotation_centre,
            rotation_matrix=self._rotation_matrix,
            rotation_rad=self._rotation_rad,
            scaling=self._scaling,
            scaling_centre=self._scaling_centre,
            scaling_matrix=self._scaling_matrix,
            translation=self._translation,
            translation_matrix=self._translation_matrix,
            type=self.__class__.__name__,
        )
    
    # This is used for image resampling, not for point clouds.
    def backward_transform_points(
        self,
        points: PointsTensor,
        grid: SamplingGrid | None = None,   # Required for 'image-centre' rotation/scale.
        **kwargs,
        ) -> PointsTensor:
        # Get homogeneous matrix.
        matrix_a = self.get_affine_backward_transform(points.device, grid=grid)

        # Transform points.
        points_h = torch.hstack([points, torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)])  # Move to homogeneous coords.
        points_t_h = torch.linalg.multi_dot([matrix_a, points_h.T]).T
        points_t = points_t_h[:, :-1]
        return points_t

    # Defines the forward/backward transforms.
    def __create_transforms(self) -> None:
        if self._rotation is not None:
            self._rotation_matrix = create_rotation(self._rotation_rad)
            self._backward_rotation_matrix = create_rotation(-self._rotation_rad)
        else:
            self._rotation_matrix = create_eye(self._dim)
            self._backward_rotation_matrix = create_eye(self._dim)
        if self._scaling is not None:
            self._scaling_matrix = create_scaling(self._scaling)
            self._backward_scaling_matrix = create_scaling(1.0 / self._scaling)
        else:
            self._scaling_matrix = create_eye(self._dim)
            self._backward_scaling_matrix = create_eye(self._dim)
        if self._translation is not None:
            self._translation_matrix = create_translation(self._translation)
            self._backward_translation_matrix = create_translation(-self._translation)
        else:
            self._translation_matrix = create_eye(self._dim)
            self._backward_translation_matrix = create_eye(self._dim)

    def get_affine_backward_transform(
        self,
        device: torch.device,
        grid: SamplingGrid | None = None,   # Required for 'image-centre' rotation/scale.
        **kwargs,
        ) -> AffineTensor:
        # Get rotation matrices.
        if self._rotation is not None:
            # Get centre of rotation.
            if self._rotation_centre == 'image-centre':
                if grid is None:
                    raise ValueError(f"Sampling 'grid' required when performing rotation around image centre (centre='image-centre').")
                size, affine = grid
                rot_centre = fov_centre(size, affine=affine)
            else:
                rot_centre = self._rotation_centre.to(device)

            rot_centre_trans_matrix = create_translation(-rot_centre, device=device)
            inv_rot_centre_trans_matrix = create_translation(rot_centre, device=device)
        else:
            rot_centre_trans_matrix = create_eye(self._dim)
            inv_rot_centre_trans_matrix = create_eye(self._dim)

        # Get scaling matrices.
        if self._scaling is not None:
            if self._scaling_centre == 'image-centre':
                if grid is None:
                    raise ValueError(f"Sampling 'grid' required when performing rotation around image centre (centre='image-centre').")
                size, affine = grid
                scale_centre = fov_centre(size, affine=affine)
            else:
                scale_centre = self._scaling_centre.to(device)

            scale_centre_trans_matrix = create_translation(-scale_centre, device=device)
            inv_scale_centre_trans_matrix = create_translation(scale_centre, device=device)
        else:
            scale_centre_trans_matrix = create_eye(self._dim)
            inv_scale_centre_trans_matrix = create_eye(self._dim)

        # Combine matrices.
        # Inverse of the forward transform, but quicker to create than solve.
        matrix = torch.linalg.multi_dot([
            inv_rot_centre_trans_matrix,
            self._backward_rotation_matrix.to(device),
            rot_centre_trans_matrix,
            inv_scale_centre_trans_matrix,
            self._backward_scaling_matrix.to(device),
            scale_centre_trans_matrix,
            self._backward_translation_matrix.to(device), 
        ])

        return matrix

    def get_affine_transform(
        self,
        device: torch.device,   # Can't infer from 'grid' as it might be None.
        grid: SamplingGrid | None = None,   # Required for 'image-centre' rotation/scale.
        **kwargs,
        ) -> AffineTensor:
        # Get rotation matrices.
        if self._rotation is not None:
            # Get centre of rotation.
            if self._rotation_centre == 'image-centre':
                if grid is None:
                    raise ValueError(f"Sampling 'grid' required when performing rotation around image centre (centre='image-centre').")
                size, affine = grid
                rot_centre = fov_centre(size, affine=affine)
            else:
                rot_centre = self._rotation_centre.to(device)

            rot_centre_trans_matrix = create_translation(-rot_centre, device=device)
            inv_rot_centre_trans_matrix = create_translation(rot_centre, device=device)
        else:
            rot_centre_trans_matrix = create_eye(self._dim)
            inv_rot_centre_trans_matrix = create_eye(self._dim)

        # Get scaling matrices.
        if self._scaling is not None:
            if self._scaling_centre == 'image-centre':
                if grid is None:
                    raise ValueError(f"Sampling 'grid' required when performing rotation around image centre (centre='image-centre').")
                size, affine = grid
                scale_centre = fov_centre(size, affine=affine)
            else:
                scale_centre = self._scaling_centre.to(device)

            scale_centre_trans_matrix = create_translation(-scale_centre, device=device)
            inv_scale_centre_trans_matrix = create_translation(scale_centre, device=device)
        else:
            scale_centre_trans_matrix = create_eye(self._dim)
            inv_scale_centre_trans_matrix = create_eye(self._dim)

        # Combine matrices.
        # Perform using order: rotation -> scaling -> translation.
        matrix = torch.linalg.multi_dot([
            self._translation_matrix.to(device), 
            inv_scale_centre_trans_matrix,
            self._scaling_matrix.to(device),
            scale_centre_trans_matrix,
            inv_rot_centre_trans_matrix,
            self._rotation_matrix.to(device),
            rot_centre_trans_matrix
        ])

        return matrix

    def __str__(self) -> str:
        params = dict(
            rotation=to_tuple(self._rotation, decimals=3) if self._rotation is not None else None,
            rotation_centre=to_tuple(self._rotation_centre, decimals=3) if self._rotation_centre != 'image-centre' else "\"image-centre\"",
            scaling=to_tuple(self._scaling, decimals=3) if self._scaling is not None else None,
            scaling_centre=to_tuple(self._scaling_centre, decimals=3) if self._scaling_centre != 'image-centre' else "\"image-centre\"",
            translation=to_tuple(self._translation, decimals=3) if self._translation is not None else None,
        )
        return super().__str__(self.__class__.__name__, params)

    def super_freeze(
        self,
        class_name: str,
        params: dict,
        ) -> str:
        return super().freeze(class_name, params)

    def super_str(
        self,
        class_name: str,
        params: dict,
        ) -> str:
        return super().__str__(class_name, params)

    # This is for point clouds, not for image resampling. Note that this
    # requires invertibility of the back point transform, which may not be
    # be available for some transforms (e.g. folded elastic).
    def transform_points(
        self,
        points: Points | List[Points],
        affine: Affine | None = None,       # Required for some transforms, e.g. Rotate, to get centre of rotation.
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
            # Get homogeneous matrix.
            matrix_a = self.get_affine_transform(device, grid=grid)

            # Perform forward transform.
            points_h = torch.hstack([p, torch.ones((p.shape[0], 1), device=device, dtype=torch.float32)])  # Move to homogeneous coords.
            points_t_h = torch.linalg.multi_dot([matrix_a, points_h.T]).T
            points_t = points_t_h[:, :-1]

            # Forward transformed points could end up off-screen and should be filtered.
            # However, we need to know which points are returned for loss calc for example.
            if filter_offgrid:
                assert size is not None, "Size must be provided for filtering off-grid points."
                assert affine is not None, "Affine must be provided for filtering off-grid points."
                grid = torch.stack([affine[:3, 3], affine[:3, 3] + size * affine[:3, :3].diag()]).to(device)
                to_keep = (points_t >= grid[0]) & (points_t < grid[1])
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

class RandomAffine(RandomSpatialTransform):
    @alias_kwargs([
        ('r', 'rotation'),
        ('rc', 'rotation_centre'),
        ('s', 'scaling'),
        ('sc', 'scaling_centre'),
        ('t', 'translation'),
    ])
    def __init__(
        self,
        rotation: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = 15.0,
        rotation_centre: Point | Literal['image-centre'] = 'image-centre',
        scaling: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = (0.8, 1.2),
        scaling_centre: Point | Literal['image-centre'] = 'image-centre',
        translation: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = 50.0,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        if rotation is not None:
            rotation_range = expand_range_arg(rotation, dim=self._dim, negate_lower=True)
            assert len(rotation_range) == 2 * self._dim, f"Expected 'rotation' of length {2 * self._dim}, got {len(rotation_range)}."
            if isinstance(rotation_centre, tuple):
                assert len(rotation_centre) == self._dim, f"Rotate centre must have {self._dim} dimensions."
            self._rotation_range = to_tensor(rotation_range).reshape(self._dim, 2)
        else:
            self._rotation_range = None
        self._rotation_centre = to_tensor(rotation_centre) if not rotation_centre == 'image-centre' else rotation_centre
        if scaling is not None:
            scaling_range = expand_range_arg(scaling, dim=self._dim, negate_lower=False)
            assert len(scaling_range) == 2 * self._dim, f"Expected 'scaling' of length {2 * self._dim}, got {len(scaling_range)}."
            self._scaling_range = to_tensor(scaling_range).reshape(self._dim, 2)
        else:
            self._scaling_range = None
        self._scaling_centre = to_tensor(scaling_centre) if not scaling_centre == 'image-centre' else scaling_centre
        if translation is not None:
            translation_range = expand_range_arg(translation, dim=self._dim, negate_lower=True)
            assert len(translation_range) == 2 * self._dim, f"Expected 'translation' of length {2 * self._dim}, got {len(translation_range)}."
            self._translation_range = to_tensor(translation_range).reshape(self._dim, 2)
        else:
            self._translation_range = None
        self._params = dict(
            dim=self._dim,
            p=self._p,
            rotation=self._rotation_range,
            rotation_centre=self._rotation_centre,
            scaling=self._scaling_range,
            scaling_centre=self._scaling_centre,
            translation=self._translation_range,
            type=self.__class__.__name__,
        )

    def freeze(self) -> 'Affine':
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random(self._dim))
        if self._rotation_range is not None:
            rot_draw = to_tuple(draw * (self._rotation_range[:, 1] - self._rotation_range[:, 0]) + self._rotation_range[:, 0])
        else:
            rot_draw = None
        draw = to_tensor(self._rng.random(self._dim))
        if self._scaling_range is not None:
            scale_draw = to_tuple(draw * (self._scaling_range[:, 1] - self._scaling_range[:, 0]) + self._scaling_range[:, 0])
        else:
            scale_draw = None
        draw = to_tensor(self._rng.random(self._dim))
        if self._translation_range is not None:
            trans_draw = to_tuple(draw * (self._translation_range[:, 1] - self._translation_range[:, 0]) + self._translation_range[:, 0])
        else:
            trans_draw = None
        params = dict(
            rotation=rot_draw,
            rotation_centre=self._rotation_centre,
            scaling=scale_draw,
            scaling_centre=self._scaling_centre,
            translation=trans_draw,
        )
        return super().freeze(Affine, params)

    def get_affine_backward_transform(
        self,
        device: torch.device,
        **kwargs,
        ) -> torch.Tensor:
        return self.freeze().get_affine_backward_transform(device, **kwargs)

    def get_affine_transform(
        self,
        device: torch.device,
        **kwargs,
        ) -> AffineTensor:
        return self.freeze().get_affine_transform(device, **kwargs)

    def __str__(self) -> str:
        params = dict(
            rotation=to_tuple(self._rotation_range.flatten(), decimals=3) if self._rotation_range is not None else None,
            rotation_centre=to_tuple(self._rotation_centre, decimals=3) if self._rotation_centre != 'image-centre' else "\"image-centre\"",
            scaling=to_tuple(self._scaling_range.flatten(), decimals=3) if self._scaling_range is not None else None,
            scaling_centre=to_tuple(self._scaling_centre, decimals=3) if self._scaling_centre != 'image-centre' else "\"image-centre\"",
            translation=to_tuple(self._translation_range.flatten(), decimals=3) if self._translation_range is not None else None,
        )
        return super().__str__(self.__class__.__name__, params)

    def super_freeze(
        self,
        class_name: str,
        params: dict,
        ) -> str:
        return super().freeze(class_name, params)
        
    def super_str(
        self,
        class_name: str,
        params: dict,
        ) -> str:
        return super().__str__(class_name, params)
