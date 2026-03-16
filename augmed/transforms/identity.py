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
        return_affine: bool = False,
        ) -> Image | List[Image | Affine]:
        images, image_was_single = arg_to_list(image, (np.ndarray, torch.Tensor), return_expanded=True)
        size = images[0].shape[-self._dim:]
        for i, img in enumerate(images[1:], 1):
            assert img.shape[-self._dim:] == size, f"All images must have the same spatial size. Expected {size}, got {img.shape[-self._dim:]} for image {i}."

        image_ts = images
        results = image_ts[0] if image_was_single else image_ts
        if return_affine:
            if isinstance(results, list):
                results.append(affine)
            else:
                results = [results, affine]

        return results

    # When a transform has a '_device' all input data will be moved to (and returned on) that device
    def transform_points(
        self,
        points: Points,
        filter_offgrid: bool = False,
        return_filtered: bool = False,
        **kwargs,
        ) -> Points | List[Points | np.ndarray | torch.Tensor]:
        points, return_type = to_tensor(points, device=self._device, dtype=torch.float32, return_type=True)
        if filter_offgrid and return_filtered:
            # Create filtered indices to match API.
            indices = to_tensor([], device=points.device, dtype=torch.int32) if return_type is torch.Tensor else np.array([])

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
        
