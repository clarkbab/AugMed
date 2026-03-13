from typing import *

from ...typing import *
from ...utils.args import alias_kwargs, arg_to_list
from ...utils.conversion import to_numpy, to_tensor, to_tuple
from ...utils.grid import grid_points, grid_sample
from ...utils.matrix import create_affine
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
        return_types = ['numpy' if isinstance(i, np.ndarray) else 'torch' for i in images]
        images = [to_tensor(i, self._device) for i in images]
        dims = [len(i.shape) for i in images]
        if self._dim == 2:
            for i, d in enumerate(dims):
                assert d in [2, 3, 4], f"Expected 2-4D image (2D spatial, optional batch/channel), got {d}D for image {i}."
        elif self._dim == 3:
            for i, d in enumerate(dims):
                assert d in [3, 4, 5], f"Expected 3-5D image (3D spatial, optional batch/channel), got {d}D for image {i}."
        size = to_tensor(images[0].shape[-self._dim:], device=images[0].device, dtype=torch.int32)
        for i, img in enumerate(images[1:], 1):
            assert img.shape[-self._dim:] == images[0].shape[-self._dim:], f"All images must have the same spatial size. Expected {tuple(images[0].shape[-self._dim:])}, got {tuple(img.shape[-self._dim:])} for image {i}."
        affine_t = to_tensor(affine, device=self._device, dtype=torch.float32) if affine is not None else create_affine(spacing=(1,) * self._dim, origin=(0,) * self._dim, device=self._device, return_type='torch')

        # Get back transformed image points (shared across all images).
        points = grid_points(images[0].shape, origin=(0,) * self._dim, spacing=(1,) * self._dim)
        points = to_tensor(points, device=self._device)

        # Perform back transform of resampling points.
        okwargs = dict(
            size=size,
            affine=affine_t,
        )
        points_t = self.backward_transform_points(points, **okwargs)

        # Reshape to image size.
        points_t = points_t.reshape(*to_tuple(size), self._dim)

        # Resample images.
        image_ts = []
        for i, rt in zip(images, return_types):
            # Perform resample.
            image_t = grid_sample(i, affine_t, points_t.to(i.device))

            # Convert to return types.
            if rt == 'numpy': 
                image_t = to_numpy(image_t)
                if return_affine:
                    affine_out = to_numpy(affine_t)
            else:
                if return_affine:
                    affine_out = affine_t
            image_ts.append(image_t)

        results = image_ts[0] if image_was_single else image_ts
        if return_affine:
            if isinstance(results, list):
                results.append(affine_out)
            else:
                results = [results, affine_out]

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
 