import numpy as np
import numpy as np
import torch
import torch
from typing import List

from ...typing import Indices, Points, PointsTensor
from .spatial import SpatialTransform, arg_to_list, get_group_device, to_return_format, to_tensor

# This is really just a utility class for breaking affine chains in the pipeline
# for testing purposes. It doesn't actually move objects.
class BreakAffineChain(SpatialTransform):
    def __init__(
        self,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        self._params = dict(
            dim=self._dim,
            type=self.__class__.__name__,
        )

    def backward_transform_points(
        self,
        points: PointsTensor,
        **kwargs,
        ) -> PointsTensor:
        return points

    def __str__(self) -> str:
        return super().__str__(
            self.__class__.__name__,
        )

    def transform_points(
        self,
        points: Points | List[Points],
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
            indiceses = to_return_format(indiceses, return_single=False, return_types=return_types)
            other_data.append(indiceses)
        results = to_return_format(pointses, other_data=other_data, return_single=points_was_single, return_types=return_types)
        return results
