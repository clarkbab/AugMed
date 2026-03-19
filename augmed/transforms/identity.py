from typing import *

from ..typing import *
from ..utils.args import arg_to_list
from ..utils.conversion import to_return_format, to_tensor
from ..utils.misc import get_group_device
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
        # Add affine to appease the API.
        other_data = []
        if return_affine:
            other_data.append(affine)
        results = to_return_format(image, other_data=other_data)

        return results

    # When a transform has a '_device' all input data will be moved to (and returned on) that device
    def transform_points(
        self,
        points: Points | List[Points],
        filter_offgrid: bool = True,
        return_filtered: bool = False,
        **kwargs,
        ) -> Points | List[Points | Indices | List[Indices]]:
        # Add indices to support the API.
        pointses, points_was_single = arg_to_list(points, (np.ndarray, torch.Tensor), return_expanded=True)
        device = get_group_device(pointses, device=self._device)
        return_types = [type(p) for p in pointses]
        pointses = [to_tensor(p, device=device, dtype=torch.float32) for p in pointses]
        other_data = []
        if filter_offgrid and return_filtered:
            indiceses = [to_tensor([], device=device, dtype=torch.int32) for _ in pointses]
            indiceses = to_return_format(indiceses, return_single=True, return_types=return_types)
            other_data.append(indiceses)
        results = to_return_format(pointses, other_data=other_data, return_types=return_types)
        return results
