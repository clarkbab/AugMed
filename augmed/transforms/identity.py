import numpy as np
import torch
from typing import List

from ..typing import AffineMatrix, BatchChannelImage, BatchImage, BatchLabelImage, ChannelImage, Image, Indices, LabelImage, Points, SamplingGrid, SpatialDim, TransformParams
from ..utils.args import alias_kwargs, arg_to_list
from ..utils.assertions import assert_image_shapes, assert_image_sizes, assert_points_shapes
from ..utils.conversion import to_return_format, to_tensor
from ..utils.geometry import spatial_size
from ..utils.python import get_group_device
from .transform import Transform

class Identity(Transform):
    def __init__(
        self,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)

    @property
    def params(self) -> TransformParams:
        return super().params()

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().__str__(self.__class__.__name__, subtransform=subtransform)

    @alias_kwargs(
        ('a', 'affine'),
        ('rg', 'return_grid'),
    )
    def transform_images(
        self,
        images: Image | LabelImage | BatchImage | BatchLabelImage | ChannelImage | BatchChannelImage | List[Image | LabelImage | BatchImage | BatchLabelImage | ChannelImage | BatchChannelImage],
        affine: AffineMatrix | None = None,
        return_grid: bool = False,
        **kwargs,
        ) -> Image | LabelImage | BatchImage | BatchLabelImage | ChannelImage | BatchChannelImage | List[Image | LabelImage | BatchImage | BatchLabelImage | ChannelImage | BatchChannelImage | AffineMatrix | SamplingGrid]:
        assert_image_shapes(images, self.__dim)
        assert_image_sizes(images, self.__dim)
        images = arg_to_list(images, (np.ndarray, torch.Tensor))
        return_types = [type(i) for i in images]
        device = get_group_device(images, device=self.__device)
        images = [to_tensor(i, device=device) for i in images]
        size = to_tensor(spatial_size(images[0], self.__dim), device=device, dtype=torch.int32)
        affine = to_tensor(affine, device=device, dtype=torch.float32)

        # Check image n_dims, and spatial sizes.
        # TODO: Extract this check for all 'transform_images' methods to use.
        for i, img in enumerate(images):
            n_dims = len(img.shape)
            possible_dims = list(range(self.__dim, self.__dim + 3))   # E.g. for 3D, possible dims are 3-5 (3D spatial, optional batch/channel).
            assert n_dims in possible_dims, f"Expected {self.__dim}-{self.__dim + 2}D image ({self.__dim}D spatial, optional batch/channel), got {n_dims}D for image {i}. Set 'dim' param if {self.__dim}D spatial is not correct."
            assert spatial_size(img, self.__dim) == spatial_size(images[0], self.__dim), f"All images must have the same spatial size. Expected {tuple(spatial_size(images[0], self.__dim))}, got {tuple(spatial_size(img, self.__dim))} for image {i}."

        # Add grid to appease the API.
        other_data = []
        if return_grid:
            grid_t = (spatial_size(images[0], self.__dim), affine)
            other_data.append(grid_t)
        return to_return_format(images, other_data=other_data, return_types=return_types)

    # When a transform has a '_device' all input data will be moved to (and returned on) that device
    @alias_kwargs(
        ('fo', 'filter_offgrid'),
        ('rf', 'return_filtered'),
    )
    def transform_points(
        self,
        points: Points | List[Points],
        filter_offgrid: bool | SpatialDim | List[SpatialDim] | None = None,
        return_filtered: bool = False,
        **kwargs,
        ) -> Points | List[Points | Indices | List[Indices]]:
        assert_points_shapes(points, self.__dim)
        points = arg_to_list(points, (np.ndarray, torch.Tensor))
        device = get_group_device(points, device=self.__device)
        return_types = [type(p) for p in points]
        points = [to_tensor(p, device=device, dtype=torch.float32) for p in points]
        filter_offgrid = filter_offgrid if filter_offgrid is not None else self.__filter_offgrid
        other_data = []
        if filter_offgrid and return_filtered:
            indiceses = [to_tensor([], device=device, dtype=torch.int32) for _ in points]
            indiceses = to_return_format(indiceses, return_types=return_types)
            other_data.append(indiceses)
        return to_return_format(points, other_data=other_data, return_types=return_types)
