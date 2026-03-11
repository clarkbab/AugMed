from typing import *

from ...typing import *
from ...utils.args import alias_kwargs, arg_to_list
from ...utils.conversion import to_array, to_tensor
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
    def transform_image(
        self,
        image: Image | List[Image],
        affine: Affine | List[Affine] | None = None,
        return_grid: bool = False,
        ) -> Image | List[Image | List[SamplingGrid]]:
        images, image_was_single = arg_to_list(image, (np.ndarray, torch.Tensor), return_expanded=True)
        return_types = ['numpy' if isinstance(i, np.ndarray) else 'torch' for i in images]
        affines = arg_to_list(affine, (np.ndarray, torch.Tensor, None), broadcast=len(images))
        images = [to_tensor(i, device=self._device) for i in images]
        dims = [len(i.shape) for i in images]
        if self._dim == 2:
            for i, d in enumerate(dims):
                assert d in [2, 3, 4], f"Expected 2-4D image (2D spatial, optional batch/channel), got {d}D for image {i}."
        elif self._dim == 3:
            for i, d in enumerate(dims):
                assert d in [3, 4, 5], f"Expected 3-5D image (3D spatial, optional batch/channel), got {d}D for image {i}."
        sizes = [to_tensor(i.shape[-self._dim:], device=i.device, dtype=torch.int32) for i in images]
        affines = [to_tensor(a, device=i.device, dtype=torch.float32) if a is not None else create_affine(spacing=(1,) * self._dim, origin=(0,) * self._dim, device=i.device, return_type='torch') for a, i in zip(affines, images)]

        # Transform images.
        image_ts = []
        grid_ts = []
        for image, s, a, rt in zip(images, sizes, affines, return_types):
            image_t = self.transform_intensity(image)

            # Convert to return types.
            grid_t = (s, a)     # Grid isn't modified by IntensityTransforms.
            if rt == 'numpy': 
                image_t = to_array(image_t)
                if return_grid:
                    grid_t = tuple(to_array(g) for g in grid_t)
            image_ts.append(image_t)
            if return_grid:
                grid_ts.append(grid_t)

        results = image_ts[0] if image_was_single else image_ts
        if return_grid:
            if isinstance(results, list):
                results.append(grid_ts)
            else:
                results = [results, grid_ts]

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
        ) -> Points:
        if isinstance(points, np.ndarray):
            return_type = 'numpy'
        else:
            return_type = 'torch'
        if return_filtered:
            indices = np.array([]) if return_type == 'numpy' else to_tensor([], device=points.device)
            return points, indices
        else:
            return points

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
    