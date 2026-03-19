from typing import *

from ...typing import *
from ...utils.args import alias_kwargs, arg_to_list
from ...utils.conversion import to_return_format, to_tensor
from ...utils.misc import get_group_device
from ..transform import RandomTransform, Transform

# These transforms change the position of the sampling grid.
class GridTransform(Transform):
    def __init__(
        self,
        **kwargs) -> None:
        super().__init__(**kwargs)

    def transform_grid(
        self,
        *args,
        **kwargs,
        ) -> SamplingGridTensor:
        raise ValueError("Subclasses of 'GridTransform' must implement 'transform_grid' method.")

    # Just removes voxels outside the transformed FOV.
    @alias_kwargs([
        ('a', 'affine'),
        ('rf', 'return_affine'),
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

        # Get new FOV (shared across all images).
        grid_t = self.transform_grid((size, affine))

        # Get resample points.
        points = grid_points(*grid_t)

        # Reshape to image size.
        size_t, _, _ = grid_t
        points = points.reshape(*to_tuple(size_t), self._dim)

        # The output affine comes from the transformed grid.
        _, affine_out = grid_t
        
        # Crop images.
        image_ts = []
        for image, rt in zip(images, return_types):
            # Perform resample.
            image_t = grid_sample(image, a, points)
            image_ts.append(image_t)

        # Convert to return format.
        other_data = []
        if return_affine:
            other_data.append(affine_out)
        results = to_return_format(image_ts, other_data=other_data, return_single=image_was_single, return_types=return_types)

        return results

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
