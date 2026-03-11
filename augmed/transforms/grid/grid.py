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
        size: SizeTensor,
        affine: AffineTensor | None = None,
        **kwargs,
        ) -> GridParamsTensor:
        raise ValueError("Subclasses of 'GridTransform' must implement 'transform_grid' method.")

    # Just removes voxels outside the transformed FOV.
    @alias_kwargs([
        ('a', 'affine'),
        ('rf', 'return_grid'),
    ])
    def transform_image(
        self,
        image: Image | List[Image],
        affine: Affine | List[Affine] = None,
        return_grid: bool = False,
        ) -> Image | List[Image | List[GridParams]]:
        images, image_was_single = arg_to_list(image, (np.ndarray, torch.Tensor), return_expanded=True)
        return_types = ['numpy' if isinstance(i, np.ndarray) else 'torch' for i in images]
        affines = arg_to_list(affine, (np.ndarray, torch.Tensor, None), broadcast=len(images))
        images = [to_tensor(i) for i in images]
        devices = [i.device for i in images]
        dims = [len(i.shape) for i in images]
        if self._dim == 2:
            for i, d in enumerate(dims):
                assert d in [2, 3, 4], f"Expected 2-4D image (2D spatial, optional batch/channel), got {d}D for image {i}."
        elif self._dim == 3:
            for i, d in enumerate(dims):
                assert d in [3, 4, 5], f"Expected 3-5D image (3D spatial, optional batch/channel), got {d}D for image {i}."
        sizes = [to_tensor(i.shape[-self._dim:], device=i.device, dtype=torch.int32) for i in images]
        affines = [to_tensor(a, device=d, dtype=torch.float32) if a is not None else create_affine(spacing=(1,) * self._dim, origin=(0,) * self._dim) for a, i in zip(affines, devices)]

        # Crop images.
        image_ts = []
        grid_ts = []
        for image, dim, s, a, rt in zip(images, dims, sizes, affines, return_types):
            # Get new FOV.
            grid_t = self.transform_grid(s, affine=a)

            # Get resample points.
            points = grid_points(*grid_t)
            points_t = to_tensor(points, device=image.device)

            # Reshape to image size.
            size_t, _, _ = grid_t
            points_t = points_t.reshape(*to_tuple(size_t), self._dim)

            # Perform resample.
            print('grid resample')
            print(image.dtype)
            image_t = grid_sample(image, a, points_t)
            print(image_t.dtype)

            # Convert to return types.
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

class RandomGridTransform(RandomTransform, GridTransform):
    def __init__(
        self,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)

    def transform_grid(
        self,
        size: SizeTensor,
        affine: AffineTensor | None = None,
        **kwargs,
        ) -> GridParamsTensor:
        return self.freeze().transform_grid(size, affine=affine, **kwargs)
