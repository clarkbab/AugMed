import numpy as np
import torch
from typing import List, Literal, Tuple

from ...typing import AffineMatrix, BatchChannelImage, BatchImage, BatchLabelImage, ChannelImage, Image, LabelImage, Number, PointsTensor, SamplingGrid
from ...utils.args import alias_kwargs, arg_to_list
from ...utils.assertions import assert_image_shapes, assert_image_sizes
from ...utils.conversion import to_return_format, to_tensor, to_tuple
from ...utils.geometry import spatial_size
from ...utils.grid import grid_points, grid_sample
from ...utils.python import get_group_device, get_private_attr
from ..transform import RandomTransform, Transform

# These transforms move objects around in the world.
class SpatialTransform(Transform):
    def __init__(
        self,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)

    def back_transform_points(
        self,
        *args,
        **kwargs,
        ) -> PointsTensor:
        raise ValueError("Subclasses of 'SpatialTransform' must implement 'back_transform_points' method.")

    @alias_kwargs(
        ('a', 'affine'),
        ('f', 'fill'),
        ('i', 'interpolation'),
        ('rg', 'return_grid'),
        ('rs', 'return_single'),
    )
    def transform_images(
        self,
        images: Image | LabelImage | BatchImage | BatchLabelImage | ChannelImage | BatchChannelImage | List[Image | LabelImage | BatchImage | BatchLabelImage | ChannelImage | BatchChannelImage],
        affine: AffineMatrix | None = None,
        fill: Number | Literal['border', 'max', 'min', 'reflection', 'zeros'] | None = None,
        interpolation: Literal['bicubic', 'bilinear', 'nearest'] | None = None,
        return_grid: bool = False,
        return_single: bool = True,
        ) -> Image | LabelImage | BatchImage | BatchLabelImage | ChannelImage | BatchChannelImage | Tuple[Image | LabelImage | BatchImage | BatchLabelImage | ChannelImage | BatchChannelImage | AffineMatrix | SamplingGrid]:
        assert_image_shapes(images, self.__dim)
        assert_image_sizes(images, self.__dim)
        images, images_was_single = arg_to_list(images, (np.ndarray, torch.Tensor), return_matched=True)
        device = get_group_device(images, device=get_private_attr(self, '__device'))
        return_types = [type(i) for i in images]
        images = [to_tensor(i, device=device) for i in images]
        size = to_tensor(spatial_size(images[0], get_private_attr(self, '__dim')), device=device, dtype=torch.int32)
        affine = to_tensor(affine, device=device, dtype=torch.float32)
        fill = fill if fill is not None else get_private_attr(self, '__fill')
        interpolation = interpolation if interpolation is not None else get_private_attr(self, '__interpolation')

        # Check image n_dims, and spatial sizes.
        # TODO: Extract this to a utility for all 'transform_images' methods.
        for i, img in enumerate(images):
            n_dims = len(img.shape)
            possible_dims = list(range(get_private_attr(self, '__dim'), get_private_attr(self, '__dim') + 3))   # E.g. for 3D, possible dims are 3-5 (3D spatial, optional batch/channel).
            assert n_dims in possible_dims, f"Expected {get_private_attr(self, '__dim')}-{get_private_attr(self, '__dim') + 2}D image ({get_private_attr(self, '__dim')}D spatial, optional batch/channel), got {n_dims}D for image {i}. Set 'dim' param if {get_private_attr(self, '__dim')}D spatial is not correct."
            assert spatial_size(img, get_private_attr(self, '__dim')) == spatial_size(images[0], get_private_attr(self, '__dim')), f"All images must have the same spatial size. Expected {tuple(spatial_size(images[0], get_private_attr(self, '__dim')))}, got {tuple(spatial_size(img, get_private_attr(self, '__dim')))} for image {i}."

        # Get back transformed image points (shared across all images).
        grid_t = (size, affine)
        points = grid_points(grid_t)

        # Perform back transform of resampling points.
        points_t = self.back_transform_points(points, grid=grid_t)

        # Reshape to image size.
        points_t = points_t.reshape(*to_tuple(size), get_private_attr(self, '__dim'))

        # Resample images.
        image_ts = []
        for i in images:
            # Perform resample.
            image_t = grid_sample(i, points_t, affine=affine, dim=get_private_attr(self, '__dim'), fill=fill, interpolation=interpolation)
            image_ts.append(image_t)

        # Convert to return format.
        other_data = []
        if return_grid:
            other_data.append(grid_t)
        return to_return_format(image_ts, other_data=other_data, return_single=return_single and images_was_single, return_types=return_types)

class RandomSpatialTransform(RandomTransform, SpatialTransform):
    def __init__(
        self,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)

    def back_transform_points(
        self,
        *args,
        **kwargs,
        ) -> PointsTensor:
        return self.freeze().back_transform_points(*args, **kwargs)
 