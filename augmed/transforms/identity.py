from typing import *

from ..typing import *
from ..utils.args import arg_to_list
from .transform import Transform

class Identity(Transform):
    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__, {})

    def transform_image(
        self,
        image: Image | List[Image],
        affine: Affine | List[Affine] | None = None,
        return_grid: bool = False,
        ) -> Image | List[Image | List[GridParams]]:
        images, images_was_single = arg_to_list(image, (np.ndarray, torch.Tensor), return_expanded=True)
        sizes = [i.shape[-self._dim:] for i in images]
        affines = arg_to_list(affine, (np.ndarray, torch.Tensor, None), broadcast=len(images))

        image_ts = images
        results = image_ts[0] if images_was_single else image_ts
        if return_grid:
            grid_ts = list(zip(sizes, affines))
            if isinstance(results, list):
                results.append(grid_ts)
            else:
                results = [results, grid_ts]

        return results

    def transform_points(
        self,
        points: Points,
        size: Size | None = None,
        affine: Affine | None = None,
        return_filtered: bool = False,
        **kwargs) -> Points:
        if return_filtered:
            # Create filtered indices to match API.
            indices = to_tensor([], device=points.device, dtype=torch.int32) if isinstance(points, torch.Tensor) else np.array([])
            return points, indices 
        return points
