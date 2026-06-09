from __future__ import annotations

import numpy as np
import torch
import torch
from typing import List, Literal, Tuple

from ....typing import AffineMatrix, AffineMatrixTensor, Indices, Number, Point, Points, PointsTensor, SamplingGrid, Size, SpatialDim, TransformParams
from ....utils.args import alias_kwargs, arg_default, arg_to_list, expand_range_arg
from ....utils.assertions import assert_points_shapes, assert_range
from ....utils.conversion import to_return_format, to_tensor, to_tuple
from ....utils.geometry import create_affine, fov, fov_centre, fov_width
from ....utils.matrix import create_rotation, create_scaling, create_translation
from ....utils.python import get_group_device, get_private_attr, set_private_attr, wrap_quotes
from ... import Identity
from ..spatial import RandomSpatialTransform, SpatialTransform

DEFAULT_ROTATION_RANGE = (-15.0, 15.0)
DEFAULT_SCALING_RANGE = (0.8, 1.2)
DEFAULT_TRANSLATION_P_RANGE = (-0.1, 0.1)

# Flip, Rotation, Translation (and others) should probably subclass this.
class Affine(SpatialTransform):
    @alias_kwargs(
        ('r', 'rotation'),
        ('rc', 'rotation_centre'),
        ('rco', 'rotation_centre_offset'),
        ('s', 'scaling'),
        ('sc', 'scaling_centre'),
        ('sco', 'scaling_centre_offset'),
        ('t', 'translation'),
        ('tp', 'translation_p'),
    )
    def __init__(
        self,
        rotation: Range2PerDim | None = 0.0,
        rotation_centre: Point | Literal['image-centre'] = 'image-centre',
        rotation_centre_offset: Range2PerDim = 0.0,
        scaling: Range2PerDim | None = 1.0,
        scaling_centre: Point | Literal['image-centre'] = 'image-centre',
        scaling_centre_offset: Range2PerDim = 0.0,
        # Why do we set both to None?
        # We don't want the user to have to override the default, (translation_p=0.0) just to
        # use translation.
        translation: Range2PerDim | None = None,
        translation_p: Range2PerDim | None = None,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        translation = arg_default((translation, translation_p), translation, 0.0)

        # Set rotation.
        if rotation is not None:
            rotation = arg_to_list(rotation, (int, float, None), broadcast=get_private_attr(self, '__dim'))
            assert len(rotation) == get_private_attr(self, '__dim'), f"Expected 'rotation' of length {get_private_attr(self, '__dim')} for dim={get_private_attr(self, '__dim')}, got {len(rotation)}."
            set_private_attr(self, '__rotation', to_tensor(rotation))
            set_private_attr(self, '__rotation_rad', torch.deg2rad(get_private_attr(self, '__rotation')) if rotation is not None else None)
        else:
            set_private_attr(self, '__rotation', None)
            set_private_attr(self, '__rotation_rad', None)
        set_private_attr(self, '__rotation_centre', 'image-centre' if rotation_centre == 'image-centre' else to_tensor(rotation_centre))
        set_private_attr(self, '__rotation_centre_offset', to_tensor(rotation_centre_offset, broadcast=get_private_attr(self, '__dim')))

        # Set scaling.
        if scaling is not None:
            scaling = arg_to_list(scaling, (int, float, None), broadcast=get_private_attr(self, '__dim'))
            assert len(scaling) == get_private_attr(self, '__dim'), f"Expected 'scaling' of length {get_private_attr(self, '__dim')} for dim={get_private_attr(self, '__dim')}, got {len(scaling)}."
            set_private_attr(self, '__scaling', to_tensor(scaling))
            if torch.any(get_private_attr(self, '__scaling') == 0):
                raise ValueError(f"Scaling must be non-zero, got: {scaling}.")
        else:
            set_private_attr(self, '__scaling', None)
        set_private_attr(self, '__scaling_centre', 'image-centre' if scaling_centre == 'image-centre' else to_tensor(scaling_centre))
        set_private_attr(self, '__scaling_centre_offset', to_tensor(scaling_centre_offset, broadcast=get_private_attr(self, '__dim')))

        # Set translation.
        if translation is not None:
            translation = arg_to_list(translation, (int, float, None), broadcast=get_private_attr(self, '__dim'))
            assert len(translation) == get_private_attr(self, '__dim'), f"Expected 'translation' of length {get_private_attr(self, '__dim')} for dim={get_private_attr(self, '__dim')}, got {len(translation)}."
            set_private_attr(self, '__translation', to_tensor(translation))
        else:
            set_private_attr(self, '__translation', None)
        if translation_p is not None:
            translation_p = arg_to_list(translation_p, (int, float, None), broadcast=get_private_attr(self, '__dim'))
            assert len(translation_p) == get_private_attr(self, '__dim'), f"Expected 'translation_p' of length {get_private_attr(self, '__dim')} for dim={get_private_attr(self, '__dim')}, got {len(translation_p)}."
            set_private_attr(self, '__translation_p', to_tensor(translation_p))
        else:
            set_private_attr(self, '__translation_p', None)

        self.__create_transforms()

    def back_transform_points(
        self,
        points: PointsTensor,
        grid: SamplingGrid | None = None,   # Required for 'image-centre' rotation/scale.
        **kwargs,
        ) -> PointsTensor:
        # Get homogeneous matrix.
        matrix_a = self.get_affine_back_transform(points.device, grid=grid)

        # Transform points.
        points_h = torch.hstack([points, torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)])  # Move to homogeneous coords.
        points_t_h = torch.linalg.multi_dot([matrix_a, points_h.T]).T
        points_t = points_t_h[:, :-1]
        return points_t
    
    # This is used for image resampling, not for point clouds.
    def __create_transforms(self) -> None:
        # Create rotation transforms.
        if get_private_attr(self, '__rotation') is not None:
            set_private_attr(self, '__rotation_matrix', create_rotation(get_private_attr(self, '__rotation_rad')))
            set_private_attr(self, '__backward_rotation_matrix', create_rotation(-get_private_attr(self, '__rotation_rad')))
        else:
            set_private_attr(self, '__rotation_matrix', create_affine(dim=get_private_attr(self, '__dim')))
            set_private_attr(self, '__backward_rotation_matrix', create_affine(dim=get_private_attr(self, '__dim')))

        # Create scaling transforms.
        if get_private_attr(self, '__scaling') is not None:
            set_private_attr(self, '__scaling_matrix', create_scaling(get_private_attr(self, '__scaling')))
            set_private_attr(self, '__backward_scaling_matrix', create_scaling(1.0 / get_private_attr(self, '__scaling')))
        else:
            set_private_attr(self, '__scaling_matrix', create_affine(dim=get_private_attr(self, '__dim')))
            set_private_attr(self, '__backward_scaling_matrix', create_affine(dim=get_private_attr(self, '__dim')))

        # Create translation transforms.
        if get_private_attr(self, '__translation') is not None:
            set_private_attr(self, '__translation_matrix', create_translation(get_private_attr(self, '__translation')))
            set_private_attr(self, '__backward_translation_matrix', create_translation(-get_private_attr(self, '__translation')))
        elif get_private_attr(self, '__translation_p') is not None:
            # The actual translation matrix can't be computed until we have the image
            # size.
            set_private_attr(self, '__translation_matrix', None)
            set_private_attr(self, '__backward_translation_matrix', None)
        else:
            set_private_attr(self, '__translation_matrix', create_affine(dim=get_private_attr(self, '__dim')))
            set_private_attr(self, '__backward_translation_matrix', create_affine(dim=get_private_attr(self, '__dim')))

    # Defines the forward/backward transforms.
    def get_affine_back_transform(
        self,
        device: torch.device,
        grid: SamplingGrid | None = None,   # Required for 'image-centre' rotation/scale.
        **kwargs,
        ) -> AffineMatrixTensor:
        # Get rotation matrices.
        if get_private_attr(self, '__rotation') is not None:
            # Get centre of rotation.
            if get_private_attr(self, '__rotation_centre') == 'image-centre':
                if grid is None:
                    raise ValueError(f"Sampling 'grid' required when performing rotation around image centre (centre='image-centre').")
                rot_centre = fov_centre(grid)
            else:
                rot_centre = get_private_attr(self, '__rotation_centre').to(device)
            rot_centre = rot_centre + get_private_attr(self, '__rotation_centre_offset').to(device)
            rot_centre_trans_matrix = create_translation(-rot_centre, device=device)
            inv_rot_centre_trans_matrix = create_translation(rot_centre, device=device)
        else:
            rot_centre_trans_matrix = create_affine(device=device, dim=get_private_attr(self, '__dim'))
            inv_rot_centre_trans_matrix = create_affine(device=device, dim=get_private_attr(self, '__dim'))

        # Get scaling matrices.
        if get_private_attr(self, '__scaling') is not None:
            if get_private_attr(self, '__scaling_centre') == 'image-centre':
                if grid is None:
                    raise ValueError(f"Sampling 'grid' required when performing rotation around image centre (centre='image-centre').")
                scale_centre = fov_centre(grid)
            else:
                scale_centre = get_private_attr(self, '__scaling_centre').to(device)
            scale_centre = scale_centre + get_private_attr(self, '__scaling_centre_offset').to(device)
            scale_centre_trans_matrix = create_translation(-scale_centre, device=device)
            inv_scale_centre_trans_matrix = create_translation(scale_centre, device=device)
        else:
            scale_centre_trans_matrix = create_affine(device=device, dim=get_private_attr(self, '__dim'))
            inv_scale_centre_trans_matrix = create_affine(device=device, dim=get_private_attr(self, '__dim'))

        # Get translation matrix.
        if get_private_attr(self, '__translation') is not None:
            backward_trans_matrix = get_private_attr(self, '__backward_translation_matrix').to(device)
        elif get_private_attr(self, '__translation_p') is not None:
            translation = get_private_attr(self, '__translation_p').to(device) * fov_width(grid)
            backward_trans_matrix = create_translation(-translation, device=device)

        # Combine matrices.
        # Inverse of the forward transform, but quicker to create than solve.
        return torch.linalg.multi_dot(list(reversed([
            backward_trans_matrix, 
            scale_centre_trans_matrix,
            get_private_attr(self, '__backward_scaling_matrix').to(device),
            inv_scale_centre_trans_matrix,
            rot_centre_trans_matrix,
            get_private_attr(self, '__backward_rotation_matrix').to(device),
            inv_rot_centre_trans_matrix,
        ])))

    def get_affine_transform(
        self,
        device: torch.device,   # Can't infer from 'grid' as it might be None.
        grid: SamplingGrid,   # Required for 'image-centre' rotation/scale.
        **kwargs,
        ) -> AffineMatrixTensor:
        # Get rotation matrices.
        if get_private_attr(self, '__rotation') is not None:
            # Get centre of rotation.
            if get_private_attr(self, '__rotation_centre') == 'image-centre':
                rot_centre = fov_centre(grid)
            else:
                rot_centre = get_private_attr(self, '__rotation_centre').to(device)
            rot_centre = rot_centre + get_private_attr(self, '__rotation_centre_offset').to(device)
            rot_centre_trans_matrix = create_translation(-rot_centre, device=device)
            inv_rot_centre_trans_matrix = create_translation(rot_centre, device=device)
        else:
            rot_centre_trans_matrix = create_affine(device=device, dim=get_private_attr(self, '__dim'))
            inv_rot_centre_trans_matrix = create_affine(device=device, dim=get_private_attr(self, '__dim'))

        # Get scaling matrices.
        if get_private_attr(self, '__scaling') is not None:
            if get_private_attr(self, '__scaling_centre') == 'image-centre':
                scale_centre = fov_centre(grid)
            else:
                scale_centre = get_private_attr(self, '__scaling_centre').to(device)
            scale_centre = scale_centre + get_private_attr(self, '__scaling_centre_offset').to(device)
            scale_centre_trans_matrix = create_translation(-scale_centre, device=device)
            inv_scale_centre_trans_matrix = create_translation(scale_centre, device=device)
        else:
            scale_centre_trans_matrix = create_affine(device=device, dim=get_private_attr(self, '__dim'))
            inv_scale_centre_trans_matrix = create_affine(device=device, dim=get_private_attr(self, '__dim'))

        # Get translation matrix.
        if get_private_attr(self, '__translation') is not None:
            trans_matrix = get_private_attr(self, '__translation_matrix').to(device)
        elif get_private_attr(self, '__translation_p') is not None:
            translation = get_private_attr(self, '__translation_p').to(device) * fov_width(grid)
            trans_matrix = create_translation(translation, device=device)

        # Combine matrices.
        # Perform using order: rotation -> scaling -> translation.
        # We can't add this as a transform.param because rotation/scaling centres might
        # depend on current sampling grid.
        return torch.linalg.multi_dot(list(reversed([
            rot_centre_trans_matrix,
            get_private_attr(self, '__rotation_matrix').to(device),
            inv_rot_centre_trans_matrix,
            scale_centre_trans_matrix,
            get_private_attr(self, '__scaling_matrix').to(device),
            inv_scale_centre_trans_matrix,
            trans_matrix,
        ])))

    @property
    def params(self) -> TransformParams:
        return super().params(
            backward_rotation_matrix=get_private_attr(self, '__backward_rotation_matrix'),
            backward_scaling_matrix=get_private_attr(self, '__backward_scaling_matrix'),
            backward_translation_matrix=get_private_attr(self, '__backward_translation_matrix'),
            rotation=to_tuple(get_private_attr(self, '__rotation')) if get_private_attr(self, '__rotation') is not None else None,
            rotation_centre=get_private_attr(self, '__rotation_centre'),
            rotation_centre_offset=to_tuple(get_private_attr(self, '__rotation_centre_offset')),
            rotation_matrix=get_private_attr(self, '__rotation_matrix'),
            rotation_rad=to_tuple(get_private_attr(self, '__rotation_rad')) if get_private_attr(self, '__rotation_rad') is not None else None,
            scaling=to_tuple(get_private_attr(self, '__scaling')) if get_private_attr(self, '__scaling') is not None else None,
            scaling_centre=get_private_attr(self, '__scaling_centre'),
            scaling_centre_offset=to_tuple(get_private_attr(self, '__scaling_centre_offset')),
            scaling_matrix=get_private_attr(self, '__scaling_matrix'),
            translation=to_tuple(get_private_attr(self, '__translation')) if get_private_attr(self, '__translation') is not None else None,
            translation_matrix=get_private_attr(self, '__translation_matrix'),
        )

    def __str__(self) -> str:
        return self.to_str()

    def super_freeze(
        self,
        class_name: str,
        params: dict,
        ) -> str:
        return super().freeze(class_name, params)

    # There's no super().super() in Python?
    def super_params(
        self,
        **params,
        ) -> str:
        return super().params(**params)

    def super_str(
        self,
        class_name: str,
        subtransform: bool = False,
        **params,
        ) -> str:
        return super().__str__(class_name, subtransform=subtransform, **params)
        
    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().__str__(
            self.__class__.__name__,
            rotation=to_tuple(get_private_attr(self, '__rotation'), dp=3) if get_private_attr(self, '__rotation') is not None else None,
            rotation_centre=to_tuple(get_private_attr(self, '__rotation_centre'), dp=3) if get_private_attr(self, '__rotation_centre') != 'image-centre' else wrap_quotes('image-centre'),
            rotation_centre_offset=to_tuple(get_private_attr(self, '__rotation_centre_offset'), dp=3),
            scaling=to_tuple(get_private_attr(self, '__scaling'), dp=3) if get_private_attr(self, '__scaling') is not None else None,
            scaling_centre=to_tuple(get_private_attr(self, '__scaling_centre'), dp=3) if get_private_attr(self, '__scaling_centre') != 'image-centre' else wrap_quotes('image-centre'),
            scaling_centre_offset=to_tuple(get_private_attr(self, '__scaling_centre_offset'), dp=3),
            subtransform=subtransform,
            translation=to_tuple(get_private_attr(self, '__translation'), dp=3) if get_private_attr(self, '__translation') is not None else None,
            translation_p=to_tuple(get_private_attr(self, '__translation_p'), dp=3) if get_private_attr(self, '__translation_p') is not None else None,
        )

    # This is for point clouds, not for image resampling. Note that this
    # requires invertibility of the back point transform, which may not be
    # be available for some transforms (e.g. folded elastic).
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
        filter_offgrid: bool | SpatialDim | List[SpatialDim] | None = None,  # Filter off-grid points, or those that are off-grid along a certain axis.
        # grid: SamplingGrid | None = None,   # Required for filtering off-grid points and some transforms, e.g. Rotate.
        return_filtered: bool = False,
        size: Size | None = None,           # Required for filtering off-grid points.
        **kwargs,
        ) -> Points | List[Points | Indices | List[Indices]]:
        assert_points_shapes(points, get_private_attr(self, '__dim'))
        points = arg_to_list(points, (np.ndarray, torch.Tensor))
        device = get_group_device(points, device=get_private_attr(self, '__device'))
        return_types = [type(p) for p in points]
        points = [to_tensor(p, device=device, dtype=torch.float32) for p in points]
        size = to_tensor(size, device=device, dtype=torch.int32)
        affine = to_tensor(affine, device=device, dtype=torch.float32)
        filter_offgrid = filter_offgrid if filter_offgrid is not None else get_private_attr(self, '__filter_offgrid')

        points_ts = []
        indiceses = []
        for p in points:
            # Get homogeneous matrix.
            matrix_a = self.get_affine_transform(device, (size, affine))

            # Perform forward transform.
            points_h = torch.hstack([p, torch.ones((p.shape[0], 1), device=device, dtype=torch.float32)])  # Move to homogeneous coords.
            points_t_h = torch.linalg.multi_dot([matrix_a, points_h.T]).T
            points_t = points_t_h[:, :-1]

            # Forward transformed points could end up off-screen and should be filtered.
            # However, we need to know which points are returned for loss calc for example.
            if filter_offgrid is not False:     # "is not" required here because filter_offgrid=0 is valid.
                assert size is not None, "Size must be provided for filtering off-grid points."
                assert affine is not None, "Affine must be provided for filtering off-grid points."
                fov_mm = fov((size, affine))
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

class RandomAffine(RandomSpatialTransform):
    @alias_kwargs(
        ('r', 'rotation'),
        ('rc', 'rotation_centre'),
        ('rco', 'rotation_centre_offset'),
        ('s', 'scaling'),
        ('sc', 'scaling_centre'),
        ('sco', 'scaling_centre_offset'),
        ('t', 'translation'),
        ('tp', 'translation_p'),
    )
    def __init__(
        self,
        rotation: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = DEFAULT_ROTATION_RANGE,
        rotation_centre: Point | Literal['image-centre'] = 'image-centre',
        rotation_centre_offset: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 0.0,
        scaling: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = DEFAULT_SCALING_RANGE,
        scaling_centre: Point | Literal['image-centre'] = 'image-centre',
        scaling_centre_offset: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 0.0,
        translation: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = None,
        translation_p: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = None,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        translation = arg_default((translation, translation_p), translation, None)
        translation_p = arg_default((translation, translation_p), translation_p, DEFAULT_TRANSLATION_P_RANGE)
        set_private_attr(self, '__rotation', rotation)
        set_private_attr(self, '__rotation_centre', to_tensor(rotation_centre) if not rotation_centre == 'image-centre' else rotation_centre)
        set_private_attr(self, '__rotation_centre_offset', rotation_centre_offset)
        set_private_attr(self, '__scaling', scaling)
        set_private_attr(self, '__scaling_centre', to_tensor(scaling_centre) if not scaling_centre == 'image-centre' else scaling_centre)
        set_private_attr(self, '__scaling_centre_offset', scaling_centre_offset)
        set_private_attr(self, '__translation', translation)
        set_private_attr(self, '__translation_p', translation_p)
        self.__expand_range_args()

    def __expand_range_args(self) -> None:
        dim = get_private_attr(self, '__dim')
        rotation_centre_offset_range = rotation_range = scaling_centre_offset_range = scaling_range = translation_range = translation_p_range = None
        if get_private_attr(self, '__rotation') is not None:
            rotation_range = expand_range_arg(get_private_attr(self, '__rotation'), dim=dim, negate_lower=True)
            assert_range(rotation_range, dim, 'rotation')
            rotation_range = to_tensor(rotation_range).reshape(dim, 2).T
        if get_private_attr(self, '__rotation_centre_offset') is not None:
            rotation_centre_offset_range = expand_range_arg(get_private_attr(self, '__rotation_centre_offset'), dim=dim, negate_lower=True)
            assert_range(rotation_centre_offset_range, dim, 'rotation_centre_offset')
            rotation_centre_offset_range = to_tensor(rotation_centre_offset_range).reshape(dim, 2).T
        if get_private_attr(self, '__scaling') is not None:
            scaling_range = expand_range_arg(get_private_attr(self, '__scaling'), dim=dim, negate_lower=False)
            assert_range(scaling_range, dim, 'scaling')
            scaling_range = to_tensor(scaling_range).reshape(dim, 2).T
        if get_private_attr(self, '__scaling_centre_offset') is not None:
            scaling_centre_offset_range = expand_range_arg(get_private_attr(self, '__scaling_centre_offset'), dim=dim, negate_lower=True)
            assert_range(scaling_centre_offset_range, dim, 'scaling_centre_offset')
            scaling_centre_offset_range = to_tensor(scaling_centre_offset_range).reshape(dim, 2).T
        if get_private_attr(self, '__translation') is not None:
            translation_range = expand_range_arg(get_private_attr(self, '__translation'), dim=dim, negate_lower=True)
            assert_range(translation_range, dim, 'translation')
            translation_range = to_tensor(translation_range).reshape(dim, 2).T
        if get_private_attr(self, '__translation_p') is not None:
            translation_p_range = expand_range_arg(get_private_attr(self, '__translation_p'), dim=dim, negate_lower=True)
            assert_range(translation_p_range, dim, 'translation_p')
            translation_p_range = to_tensor(translation_p_range).reshape(dim, 2).T

        set_private_attr(self, '__rotation_range', rotation_range)
        set_private_attr(self, '__rotation_centre_offset_range', rotation_centre_offset_range)
        set_private_attr(self, '__scaling_range', scaling_range)
        set_private_attr(self, '__scaling_centre_offset_range', scaling_centre_offset_range)
        set_private_attr(self, '__translation_range', translation_range)
        set_private_attr(self, '__translation_p_range', translation_p_range)

    def freeze(self) -> Affine | Identity:
        should_apply = get_private_attr(self, '__rng').random(1) < get_private_attr(self, '__p')
        if not should_apply:
            return Identity(dim=get_private_attr(self, '__dim'))

        # Draw rotation params.
        rot_draw = rot_centre_offset_draw = scale_draw = scale_centre_offset_draw = trans_draw = trans_p_draw = None
        rotation_range = get_private_attr(self, '__rotation_range')
        if rotation_range is not None:
            draw = to_tensor(get_private_attr(self, '__rng').random(get_private_attr(self, '__dim')))
            rot_draw = to_tuple(draw * (rotation_range[1] - rotation_range[0]) + rotation_range[0]) if rotation_range is not None else None
        rotation_centre_offset_range = get_private_attr(self, '__rotation_centre_offset_range')
        if rotation_centre_offset_range is not None:
            draw = to_tensor(get_private_attr(self, '__rng').random(get_private_attr(self, '__dim')))
            rot_centre_offset_draw = to_tuple(draw * (rotation_centre_offset_range[1] - rotation_centre_offset_range[0]) + rotation_centre_offset_range[0])

        # Draw scaling params.
        scaling_range = get_private_attr(self, '__scaling_range')
        if scaling_range is not None:
            draw = to_tensor(get_private_attr(self, '__rng').random(get_private_attr(self, '__dim')))
            scale_draw = to_tuple(draw * (scaling_range[1] - scaling_range[0]) + scaling_range[0]) if scaling_range is not None else None
        scaling_centre_offset_range = get_private_attr(self, '__scaling_centre_offset_range')
        if scaling_centre_offset_range is not None:
            draw = to_tensor(get_private_attr(self, '__rng').random(get_private_attr(self, '__dim')))
            scale_centre_offset_draw = to_tuple(draw * (scaling_centre_offset_range[1] - scaling_centre_offset_range[0]) + scaling_centre_offset_range[0])

        # Draw translation params.
        translation_range = get_private_attr(self, '__translation_range')
        if translation_range is not None:
            draw = to_tensor(get_private_attr(self, '__rng').random(get_private_attr(self, '__dim')))
            trans_draw = to_tuple(draw * (translation_range[1] - translation_range[0]) + translation_range[0])
        translation_p_range = get_private_attr(self, '__translation_p_range')
        if translation_p_range is not None:
            draw = to_tensor(get_private_attr(self, '__rng').random(get_private_attr(self, '__dim')))
            trans_p_draw = to_tuple(draw * (translation_p_range[1] - translation_p_range[0]) + translation_p_range[0])

        params = dict(
            rotation=rot_draw,
            rotation_centre=get_private_attr(self, '__rotation_centre'),
            rotation_centre_offset=rot_centre_offset_draw,
            scaling=scale_draw,
            scaling_centre=get_private_attr(self, '__scaling_centre'),
            scaling_centre_offset=scale_centre_offset_draw,
            translation=trans_draw,
            translation_p=trans_p_draw,
        )
        return super().freeze(Affine, params)

    def get_affine_back_transform(
        self,
        device: torch.device,
        **kwargs,
        ) -> torch.Tensor:
        return self.freeze().get_affine_back_transform(device, **kwargs)

    def get_affine_transform(
        self,
        device: torch.device,
        **kwargs,
        ) -> AffineMatrixTensor:
        return self.freeze().get_affine_transform(device, **kwargs)

    @property
    def params(self) -> TransformParams:
        rotation_range = get_private_attr(self, '__rotation_range')
        rotation_centre_offset_range = get_private_attr(self, '__rotation_centre_offset_range')
        scaling_range = get_private_attr(self, '__scaling_range')
        scaling_centre_offset_range = get_private_attr(self, '__scaling_centre_offset_range')
        translation_range = get_private_attr(self, '__translation_range')
        translation_p_range = get_private_attr(self, '__translation_p_range')
        return super().params(
            rotation=to_tuple(rotation_range.T.flatten()) if rotation_range is not None else None,
            rotation_centre=get_private_attr(self, '__rotation_centre'),
            rotation_centre_offset=to_tuple(rotation_centre_offset_range.T.flatten()),
            scaling=to_tuple(scaling_range.T.flatten()) if scaling_range is not None else None,
            scaling_centre=get_private_attr(self, '__scaling_centre'),
            scaling_centre_offset=to_tuple(scaling_centre_offset_range.T.flatten()),
            translation=to_tuple(translation_range.T.flatten()) if translation_range is not None else None,
            translation_p=to_tuple(translation_p_range.T.flatten()) if translation_p_range is not None else None,
        )

    def set_dim(
        self,
        dim: SpatialDim,
        ) -> None:
        super().set_dim(dim)
        self.__expand_range_args()

    def __str__(self) -> str:
        return self.to_str()

    def super_freeze(
        self,
        class_name: str,
        params: dict,
        ) -> str:
        return super().freeze(class_name, params)

    # There's no super().super() in Python?
    def super_params(
        self,
        **params,
        ) -> str:
        return super().params(**params)
        
    def super_str(
        self,
        class_name: str,
        subtransform: bool = False,
        **params,
        ) -> str:
        return super().__str__(class_name, subtransform=subtransform, **params)
        
    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().__str__(
            self.__class__.__name__,
            rotation=to_tuple(get_private_attr(self, '__rotation_range').T.flatten(), dp=3) if get_private_attr(self, '__rotation_range') is not None else None,
            rotation_centre=to_tuple(get_private_attr(self, '__rotation_centre'), dp=3) if get_private_attr(self, '__rotation_centre') != 'image-centre' else wrap_quotes('image-centre'),
            rotation_centre_offset=to_tuple(get_private_attr(self, '__rotation_centre_offset_range').T.flatten(), dp=3) if get_private_attr(self, '__rotation_centre_offset_range') is not None else None,
            scaling=to_tuple(get_private_attr(self, '__scaling_range').T.flatten(), dp=3) if get_private_attr(self, '__scaling_range') is not None else None,
            scaling_centre=to_tuple(get_private_attr(self, '__scaling_centre'), dp=3) if get_private_attr(self, '__scaling_centre') != 'image-centre' else wrap_quotes('image-centre'),
            scaling_centre_offset=to_tuple(get_private_attr(self, '__scaling_centre_offset_range').T.flatten(), dp=3) if get_private_attr(self, '__scaling_centre_offset_range') is not None else None,
            subtransform=subtransform,
            translation=to_tuple(get_private_attr(self, '__translation_range').T.flatten(), dp=3) if get_private_attr(self, '__translation_range') is not None else None,
            translation_p=to_tuple(get_private_attr(self, '__translation_p_range').T.flatten(), dp=3) if get_private_attr(self, '__translation_p_range') is not None else None,
        )
