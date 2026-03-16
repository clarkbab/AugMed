from typing import *

from ...typing import *
from ...utils.args import alias_kwargs, arg_to_list
from ...utils.conversion import to_numpy, to_tensor
from ...utils.matrix import create_affine
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
        affine_t = to_tensor(affine, device=self._device, dtype=torch.float32) if affine is not None else create_affine(spacing=(1,) * self._dim, origin=(0,) * self._dim, device=self._device, return_type='torch')

        # Transform images.
        image_ts = []
        for image, rt in zip(images, return_types):
            image_t = self.transform_intensity(image)

            # Convert to return types.
            if rt == 'numpy': 
                image_t = to_numpy(image_t)
            image_ts.append(image_t)

        results = image_ts[0] if image_was_single else image_ts
        if return_affine:
            affine_out = to_numpy(affine_t) if return_types[0] == 'numpy' else affine_t
            if isinstance(results, list):
                results.append(affine_out)
            else:
                results = [results, affine_out]

        return results

    def transform_intensity(
        self,
        *args,
        **kwargs,
        ) -> ImageTensor:
        raise ValueError("Subclasses of 'IntensityTransform' must implement 'transform_intensity' method.")

    def transform_points(
        self,
        points: Points,
        return_filtered: bool = False,
        **kwargs,
        ) -> Points | List[Points | np.ndarray | torch.Tensor]:
        points, return_type = to_tensor(points, device=self._device, dtype=torch.float32, return_type=True)
        if return_filtered:
            indices = np.array([]) if return_type is np.ndarray else to_tensor([], device=points.device)

        # Convert return types.
        if return_type is np.ndarray:
            points_t = to_numpy(points_t)
            if filter_offgrid and return_filtered:
                indices = to_numpy(indices)

        # Format returned values.
        results = points_t
        if filter_offgrid and return_filtered:
            results = [points_t, indices]

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
    