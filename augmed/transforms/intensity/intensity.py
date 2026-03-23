import numpy as np
import torch
from typing import List

from ...typing import AffineMatrix, Image, ImageTensor, Indices, Points
from ...utils.args import alias_kwargs, arg_to_list
from ...utils.conversion import to_return_format, to_tensor
from ...utils.misc import get_group_device
from ..transform import RandomTransform, Transform

# These transforms change pixel/voxel intensities.
class IntensityTransform(Transform):
    def __init__(
        self,
        **kwargs) -> None:
        super().__init__(**kwargs)

    @alias_kwargs([
        ('a', 'affine'),
    ])
    def transform_images(
        self,
        image: Image | List[Image],
        affine: AffineMatrix | None = None,
        return_affine: bool = False,
        ) -> Image | List[Image | AffineMatrix]:
        images, image_was_single = arg_to_list(image, (np.ndarray, torch.Tensor), return_expanded=True)
        device = get_group_device(images, device=self._device)
        return_types = [type(i) for i in images]
        images = [to_tensor(i, device=device) for i in images]

        # Check image n_dims, and spatial sizes.
        for i, img in enumerate(images):
            n_dims = len(img.shape)
            possible_dims = list(range(self._dim, self._dim + 3))   # E.g. for 3D, possible dims are 3-5 (3D spatial, optional batch/channel).
            assert n_dims in possible_dims, f"Expected {self._dim}-{self._dim + 2}D image ({self._dim}D spatial, optional batch/channel), got {n_dims}D for image {i}."
            assert img.shape[-self._dim:] == images[0].shape[-self._dim:], f"All images must have the same spatial size. Expected {tuple(images[0].shape[-self._dim:])}, got {tuple(img.shape[-self._dim:])} for image {i}."

        # Transform images.
        image_ts = []
        for image, rt in zip(images, return_types):
            image_t = self.transform_intensity(image)
            image_ts.append(image_t)

        # Convert to return format.
        other_data = []
        if return_affine:
            other_data.append(affine_out)
        results = to_return_format(image_ts, other_data=other_data, return_single=image_was_single, return_types=return_types)

        return results

    def transform_intensity(
        self,
        *args,
        **kwargs,
        ) -> ImageTensor:
        raise ValueError("Subclasses of 'IntensityTransform' must implement 'transform_intensity' method.")

    def transform_points(
        self,
        points: Points | List[Points],
        filter_offgrid: bool = True,
        return_filtered: bool = False,
        **kwargs,
        ) -> Points | List[Points | Indices | List[Indices]]:
        # Add indices to support the API.
        pointses, points_was_single = arg_to_list(points, (np.ndarray, torch.Tensor), return_expanded=True)
        device = get_group_device(pointses, device=self._device)
        return_types = [type(p) for p in pointses]
        pointses = [to_tensor(p, device=device) for p in pointses]
        other_data = []
        if filter_offgrid and return_filtered:
            indiceses = [to_tensor([], device=device, dtype=torch.int32) for _ in pointses]
            indiceses = to_return_format(indiceses, return_single=False, return_types=return_types)
            other_data.append(indiceses)
        results = to_return_format(pointses, other_data=other_data, return_single=points_was_single, return_types=return_types)
        return results

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
    