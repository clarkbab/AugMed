from typing import *

from ..typing import *
from ..utils.args import arg_to_list
from ..utils.conversion import to_tensor
from .transform import Transform

class Identity(Transform):
    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__, {})

    def transform_images(
        self,
        image: Image | List[Image],
        affine: Affine | None = None,
        return_grid: bool = False,
        ) -> Image | List[Image | List[SamplingGrid]]:
        images, image_was_single = arg_to_list(image, (np.ndarray, torch.Tensor), return_expanded=True)
        size = images[0].shape[-self._dim:]
        for i, img in enumerate(images[1:], 1):
            assert img.shape[-self._dim:] == size, f"All images must have the same spatial size. Expected {size}, got {img.shape[-self._dim:]} for image {i}."

        image_ts = images
        results = image_ts[0] if image_was_single else image_ts
        if return_grid:
            grid_ts = [(size, affine)]
            if isinstance(results, list):
                results.append(grid_ts)
            else:
                results = [results, grid_ts]

        return results

    def transform_points(
        self,
        points: Points,
        return_filtered: bool = False,
        **kwargs,
        ) -> Points:
        if return_filtered:
            # Create filtered indices to match API.
            indices = to_tensor([], device=points.device, dtype=torch.int32) if isinstance(points, torch.Tensor) else np.array([])
            return points, indices 
        return points
