import numpy as np
import torch
from typing import List, Literal, Tuple

from ...typing import AffineMatrix, BatchChannelImage, BatchImage, BatchLabelImage, ChannelImage, Image, Indices, LabelImage, Number, Points, SamplingGridTensor, SpatialDim
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
        ) -> Image | LabelImage | BatchImage | BatchLabelImage | ChannelImage | BatchChannelImage | Tuple[Image | LabelImage | BatchImage | BatchLabelImage | ChannelImage | BatchChannelImage | AffineMatrix | SamplingGridTensor]:
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
        return to_return_format(image_ts, other_data=other_data, return_single=return_single and images_was_single, return_types=return_types)

    @alias_kwargs(
        ('fo', 'filter_offgrid'),
        ('rf', 'return_filtered'),
        ('rs', 'return_single'),
    )
    def transform_points(
        self,
        points: Points | List[Points],
        filter_offgrid: bool | SpatialDim | List[SpatialDim] | None = None,
        return_filtered: bool = False,
        return_single: bool = True,
        **kwargs,  # Allows them to pass kwargs that work for other transforms, e.g. 'affine'.
        ) -> Points | List[Points | Indices | List[Indices]]:
        assert_points_shapes(points, self.__dim)
        points, points_was_single = arg_to_list(points, (np.ndarray, torch.Tensor), return_matched=True)
        device = get_group_device(points, device=get_private_attr(self, '__device'))
        return_types = [type(p) for p in points]
        points = [to_tensor(p, device=device) for p in points]
        filter_offgrid = filter_offgrid if filter_offgrid is not None else get_private_attr(self, '__filter_offgrid')
        other_data = []
        if filter_offgrid and return_filtered:
            indiceses = [to_tensor([], device=device, dtype=torch.int32) for _ in points]
            indiceses = to_return_format(indiceses, return_single=False, return_types=return_types)
            other_data.append(indiceses)
        return to_return_format(points, other_data=other_data, return_single=return_single and points_was_single, return_types=return_types)

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
