from typing import *

from ...typing import *
from ...utils.args import alias_kwargs, arg_to_list
from ...utils.conversion import to_array, to_tensor, to_tuple
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
    def transform_image(
        self,
        image: Image | List[Image],
        affine: Affine | List[Affine] | None = None,
        return_grid: bool = False,  # Return a grid or list of grids as the final element.
        ) -> Image | List[Image | List[SamplingGrid]]:
        images, image_was_single = arg_to_list(image, (np.ndarray, torch.Tensor), return_expanded=True)
        return_types = ['numpy' if isinstance(i, np.ndarray) else 'torch' for i in images]
        affines = arg_to_list(affine, (np.ndarray, torch.Tensor, None), broadcast=len(images))
        images = [to_tensor(i, self._device) for i in images]
        dims = [len(i.shape) for i in images]
        if self._dim == 2:
            for i, d in enumerate(dims):
                assert d in [2, 3, 4], f"Expected 2-4D image (2D spatial, optional batch/channel), got {d}D for image {i}."
        elif self._dim == 3:
            for i, d in enumerate(dims):
                assert d in [3, 4, 5], f"Expected 3-5D image (3D spatial, optional batch/channel), got {d}D for image {i}."
        sizes = [to_tensor(i.shape[-self._dim:], device=i.device, dtype=torch.int32) for i in images]
        affines = [to_tensor(a, device=i.device, dtype=torch.float32) if a is not None else create_affine(spacing=(1,) * self._dim, origin=(0,) * self._dim, device=i.device, return_type='torch') for a, i in zip(affines, images)]

        # Group images by grid params (size, affine).
        groups = [0]    # Grid params groups.
        image_groups = { 0: 0 }     # Map from image number to grid param group.
        for i, (s, a) in enumerate(zip(sizes[1:], affines[1:])):
            for g in groups:
                g_s, g_a = sizes[g], affines[g]
                if torch.all(s == g_s) and torch.all(a == g_a):
                    image_groups[i + 1] = g
                else:
                    groups.append(i + 1)
                    image_groups[i + 1] = i + 1

        # Get back transformed image points for all groups.
        group_points_ts = []
        for g in groups:
            image, size, affine = images[g], sizes[g], affines[g]
            points = grid_points(image.shape, origin=(0,) * self._dim, spacing=(1,) * self._dim)
            points = to_tensor(points, device=image.device)

            # Perform back transform of resampling points.
            # Currently we pass all args to each transform and they can consume if they need.
            okwargs = dict(
                size=size,
                affine=affine,
            )
            points_t = self.backward_transform_points(points, **okwargs)
            group_points_ts.append(points_t)

        # Resample images.
        image_ts = []
        grid_ts = []
        for g, i, d, s, a, rt in zip(groups, images, dims, sizes, affines, return_types):
            # Get resample points.
            points_t = group_points_ts[g].to(i.device)

            # Reshape to image size.
            points_t = points_t.reshape(*to_tuple(s), self._dim)

            # Perform resample.
            image_t = grid_sample(i, sp, o, points_t)

            # Convert to return types.
            grid_t = (s, a)     # Grids are not modified by SpatialTransforms.
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
 