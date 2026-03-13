from typing import *

from ...typing import *
from ...utils import *
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
        return_types = ['numpy' if isinstance(i, np.ndarray) else 'torch' for i in images]
        images = [to_tensor(i, device=self._device) for i in images]
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
        a = to_tensor(affine, device=self._device, dtype=torch.float32) if affine is not None else create_affine(spacing=(1,) * self._dim, origin=(0,) * self._dim)

        # Get new FOV (shared across all images).
        grid_t = self.transform_grid(size, affine=a)

        # Get resample points.
        points = grid_points(*grid_t)
        points_t = to_tensor(points, device=self._device)

        # Reshape to image size.
        size_t, _, _ = grid_t
        points_t = points_t.reshape(*to_tuple(size_t), self._dim)

        # The output affine comes from the transformed grid.
        _, affine_out = grid_t
        
        # Crop images.
        image_ts = []
        for image, rt in zip(images, return_types):
            # Perform resample.
            image_t = grid_sample(image, a, points_t.to(image.device))

            # Convert to return types.
            if rt == 'numpy': 
                image_t = to_numpy(image_t)
            image_ts.append(image_t)

        results = image_ts[0] if image_was_single else image_ts
        if return_affine:
            out = to_numpy(affine_out) if return_types[0] == 'numpy' else affine_out
            if isinstance(results, list):
                results.append(out)
            else:
                results = [results, out]

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
