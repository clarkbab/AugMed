import numpy as np
import torch
from typing import List

from ...typing import Affine, Image, PointsTensor
from ...utils.args import alias_kwargs, arg_to_list
from ...utils.conversion import to_return_format, to_tensor, to_tuple
from ...utils.grid import grid_points, grid_sample
from ...utils.misc import get_group_device
from ..transform import RandomTransform, Transform

# These transforms move objects around in the world.
class SpatialTransform(Transform):
    def __init__(
        self,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)

    def backward_transform_points(
        self,
        *args,
        **kwargs,
        ) -> PointsTensor:
        raise ValueError("Subclasses of 'SpatialTransform' must implement 'backward_transform_points' method.")

    @alias_kwargs([
        ('a', 'affine'),
    ])
    def transform_images(
        self,
        image: Image | List[Image],
        affine: Affine | None = None,
        return_affine: bool = False,
        ) -> Image | List[Image | Affine]:
        images, image_was_single = arg_to_list(image, (np.ndarray, torch.Tensor), return_expanded=True)
        device = get_group_device(images, device=self._device)
        return_types = [type(i) for i in images]
        images = [to_tensor(i, device=device) for i in images]
        size = to_tensor(images[0].shape[-self._dim:], device=device, dtype=torch.int32)
        affine = to_tensor(affine, device=device, dtype=torch.float32)

        # Check image n_dims, and spatial sizes.
        for i, img in enumerate(images):
            n_dims = len(img.shape)
            possible_dims = list(range(self._dim, self._dim + 3))   # E.g. for 3D, possible dims are 3-5 (3D spatial, optional batch/channel).
            assert n_dims in possible_dims, f"Expected {self._dim}-{self._dim + 2}D image ({self._dim}D spatial, optional batch/channel), got {n_dims}D for image {i}."
            assert img.shape[-self._dim:] == images[0].shape[-self._dim:], f"All images must have the same spatial size. Expected {tuple(images[0].shape[-self._dim:])}, got {tuple(img.shape[-self._dim:])} for image {i}."

        # Get back transformed image points (shared across all images).
        points = grid_points(size, origin=(0,) * self._dim, spacing=(1,) * self._dim)

        # Perform back transform of resampling points.
        points_t = self.backward_transform_points(points, grid=(size, affine))

        # Reshape to image size.
        points_t = points_t.reshape(*to_tuple(size), self._dim)

        # Resample images.
        image_ts = []
        for i, rt in zip(images, return_types):
            # Perform resample.
            image_t = grid_sample(i, affine_t, points_t)
            image_ts.append(image_t)

        # Convert to return format.
        other_data = []
        if return_affine:
            other_data.append(affine_out)
        results = to_return_format(image_ts, other_data=other_data, return_single=image_was_single, return_types=return_types)

        return results

class RandomSpatialTransform(RandomTransform, SpatialTransform):
    def __init__(
        self,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)

    def backward_transform_points(
        self,
        *args,
        **kwargs,
        ) -> PointsTensor:
        return self.freeze().backward_transform_points(*args, **kwargs)
 