import numpy as np
import torch
from typing import List, Literal

from ...typing import AffineMatrix, ImagesInput, ImageOutputs, Number, PointsInput, PointsOutputs, SamplingGridTensor, SpatialAxis, SpatialDim
from ...utils.args import alias_kwargs, arg_to_list
from ...utils.assertions import assert_image_shapes, assert_image_sizes, assert_points_shapes
from ...utils.conversion import to_return_format, to_tensor, to_tuple
from ...utils.geometry import spatial_size
from ...utils.grid import grid_points, grid_sample
from ...utils.python import get_group_device, get_private_attr
from ..transform import RandomTransform, Transform

# These transforms change the position of the sampling grid.
class GridTransform(Transform):
    def __init__(
        self,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)

    def transform_grid(
        self,
        *args,
        **kwargs,
        ) -> SamplingGridTensor:
        raise ValueError("Subclasses of 'GridTransform' must implement 'transform_grid' method.")

    # Just removes voxels outside the transformed FOV.
    @alias_kwargs(
        ('a', 'affine'),
        ('f', 'fill'),
        ('i', 'interpolation'),
        ('rg', 'return_grid'),
    )
    def transform_images(
        self,
        images: ImagesInput,
        affine: AffineMatrix | None = None,
        fill: Number | Literal['border', 'max', 'min', 'reflection', 'zeros'] | None = None,
        interpolation: Literal['bicubic', 'bilinear', 'nearest'] | None = None,
        return_grid: bool = False,
        ) -> ImageOutputs:
        self.infer_dim(images=images)
        assert_image_shapes(images, get_private_attr(self, '__dim'))
        assert_image_sizes(images, get_private_attr(self, '__dim'))
        images = arg_to_list(images, (np.ndarray, torch.Tensor))
        device = get_group_device(images, device=get_private_attr(self, '__device'))
        return_types = [type(i) for i in images]
        images = [to_tensor(i, device=device) for i in images]
        size = to_tensor(spatial_size(images[0], get_private_attr(self, '__dim')), device=device, dtype=torch.int32)
        affine = to_tensor(affine, device=device, dtype=torch.float32)
        fill = fill if fill is not None else get_private_attr(self, '__fill')
        interpolation = interpolation if interpolation is not None else get_private_attr(self, '__interpolation')

        # Check image n_dims, and spatial sizes.
        for i, img in enumerate(images):
            n_dims = len(img.shape)
            possible_dims = list(range(get_private_attr(self, '__dim'), get_private_attr(self, '__dim') + 3))   # E.g. for 3D, possible dims are 3-5 (3D spatial, optional batch/channel).
            assert n_dims in possible_dims, f"Expected {get_private_attr(self, '__dim')}-{get_private_attr(self, '__dim') + 2}D image ({get_private_attr(self, '__dim')}D spatial, optional batch/channel), got {n_dims}D for image {i}. Set 'dim' param if {get_private_attr(self, '__dim')}D spatial is not correct."
            assert spatial_size(img, get_private_attr(self, '__dim')) == spatial_size(images[0], get_private_attr(self, '__dim')), f"All images must have the same spatial size. Expected {tuple(spatial_size(images[0], get_private_attr(self, '__dim')))}, got {tuple(spatial_size(img, get_private_attr(self, '__dim')))} for image {i}."

        # Get new FOV (shared across all images).
        grid_t = self.transform_grid((size, affine))

        # Get resample points.
        points = grid_points(grid_t)

        # Reshape to image size.
        size_t, _ = grid_t
        points = points.reshape(*to_tuple(size_t), get_private_attr(self, '__dim'))
        
        # Crop images.
        image_ts = []
        for image in images:
            # Perform resample.
            image_t = grid_sample(image, points, affine=affine, dim=get_private_attr(self, '__dim'), fill=fill, interpolation=interpolation)
            image_ts.append(image_t)

        # Convert to return format.
        other_data = []
        if return_grid:
            other_data.append(grid_t)
        return to_return_format(image_ts, other_data=other_data, return_types=return_types)

    @alias_kwargs(
        ('fo', 'filter_offgrid'),
        ('rf', 'return_filtered'),
        ('s', 'size'),
    )
    def transform_points(
        self,
        points: PointsInput,
        filter_offgrid: bool | SpatialAxis | List[SpatialAxis] | None = None,
        return_filtered: bool = False,
        **kwargs,  # Allows them to pass kwargs that work for other transforms, e.g. 'affine'.
        ) -> PointsOutputs:
        self.infer_dim(points=points)
        assert_points_shapes(points, self.__dim)
        points = arg_to_list(points, (np.ndarray, torch.Tensor))
        device = get_group_device(points, device=get_private_attr(self, '__device'))
        return_types = [type(p) for p in points]
        points = [to_tensor(p, device=device) for p in points]
        filter_offgrid = filter_offgrid if filter_offgrid is not None else get_private_attr(self, '__filter_offgrid')
        other_data = []
        if filter_offgrid and return_filtered:
            indiceses = [to_tensor([], device=device, dtype=torch.int32) for _ in points]
            indiceses = to_return_format(indiceses, return_types=return_types)
            other_data.append(indiceses)
        return to_return_format(points, other_data=other_data, return_types=return_types)

class RandomGridTransform(RandomTransform, GridTransform):
    def __init__(
        self,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)

    def transform_grid(
        self,
        *args,
        **kwargs,
        ) -> SamplingGridTensor:
        return self.freeze().transform_grid(*args, **kwargs)
