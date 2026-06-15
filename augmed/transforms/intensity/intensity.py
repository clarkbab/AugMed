import numpy as np
import torch

from ...typing import AffineMatrix, ImagesInput, ImageOutputs, ImageTensor, PointsInput, PointsOutputs
from ...utils.args import alias_kwargs, arg_to_list
from ...utils.assertions import assert_image_shapes, assert_image_sizes, assert_points_shapes
from ...utils.conversion import to_return_format, to_tensor
from ...utils.geometry import spatial_size
from ...utils.python import get_group_device, get_private_attr
from ..transform import RandomTransform, Transform

# These transforms change pixel/voxel intensities.
class IntensityTransform(Transform):
    def __init__(
        self,
        **kwargs) -> None:
        super().__init__(**kwargs)

    @alias_kwargs(
        ('a', 'affine'),
        ('rg', 'return_grid'),
    )
    def transform_images(
        self,
        images: ImagesInput,
        affine: AffineMatrix | None = None,
        return_grid: bool = False,
        **kwargs,   # Allows them to pass kwargs that work for other transforms, e.g. 'fill'.
        ) -> ImageOutputs:
        self.infer_dim(images=images)
        assert_image_shapes(images, get_private_attr(self, '__dim'))
        assert_image_sizes(images, get_private_attr(self, '__dim'))
        images = arg_to_list(images, (np.ndarray, torch.Tensor))
        device = get_group_device(images, device=get_private_attr(self, '__device'))
        return_types = [type(i) for i in images]
        images = [to_tensor(i, device=device) for i in images]

        # Check image n_dims, and spatial sizes.
        # TODO: Extract this to a utility for all 'transform_images' methods.
        for i, img in enumerate(images):
            n_dims = len(img.shape)
            possible_dims = list(range(get_private_attr(self, '__dim'), get_private_attr(self, '__dim') + 3))   # E.g. for 3D, possible dims are 3-5 (3D spatial, optional batch/channel).
            assert n_dims in possible_dims, f"Expected {get_private_attr(self, '__dim')}-{get_private_attr(self, '__dim') + 2}D image ({get_private_attr(self, '__dim')}D spatial, optional batch/channel), got {n_dims}D for image {i}. Set 'dim' param if {get_private_attr(self, '__dim')}D spatial is not correct."
            assert spatial_size(img, get_private_attr(self, '__dim')) == spatial_size(images[0], get_private_attr(self, '__dim')), f"All images must have the same spatial size. Expected {tuple(spatial_size(images[0], get_private_attr(self, '__dim')))}, got {tuple(spatial_size(img, get_private_attr(self, '__dim')))} for image {i}."

        # Transform images.
        image_ts = []
        for i in images:
            image_t = self.transform_intensity(i)
            image_ts.append(image_t)

        # Convert to return format.
        other_data = []
        if return_grid:
            grid_t = (spatial_size(image_t, get_private_attr(self, '__dim')), affine)
            other_data.append(grid_t)
        return to_return_format(image_ts, other_data=other_data, return_types=return_types)

    def transform_intensity(
        self,
        *args,
        **kwargs,
        ) -> ImageTensor:
        raise ValueError("Subclasses of 'IntensityTransform' must implement 'transform_intensity' method.")

    @alias_kwargs(
        ('a', 'affine'),
        ('fo', 'filter_offgrid'),
        ('rf', 'return_filtered'),
        ('s', 'size'),
    )
    def transform_points(
        self,
        points: PointsInput,
        filter_offgrid: bool | None = None,
        return_filtered: bool = False,
        **kwargs,
        ) -> PointsOutputs:
        self.infer_dim(points)
        assert_points_shapes(points, get_private_attr(self, '__dim'))
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

class RandomIntensityTransform(RandomTransform, IntensityTransform):
    def __init__(
        self,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)

    def transform_intensity(
        self,
        image: ImageTensor,
        **kwargs,
        ) -> ImageTensor:
        return self.freeze().transform_intensity(image, **kwargs)
    