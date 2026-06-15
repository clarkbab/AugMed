import numpy as np
import numpy as np
import torch
import torch

from ...typing import PointsInput, PointsOutputs, PointsTensor
from ...utils.args import alias_kwargs, arg_to_list
from ...utils.assertions import assert_points_shapes
from ...utils.conversion import to_return_format, to_tensor
from ...utils.python import get_group_device, set_private_attr
from .spatial import SpatialTransform

# This is really just a utility class for breaking affine chains in the pipeline
# for testing purposes. It doesn't actually move objects.
class BreakAffineChain(SpatialTransform):
    def __init__(
        self,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        set_private_attr(self, '__params', dict(
            dim=self.__dim,
            type=self.__class__.__name__,
        ))

    def back_transform_points(
        self,
        points: PointsTensor,
        **kwargs,
        ) -> PointsTensor:
        return points

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().__str__(
            self.__class__.__name__,
            subtransform=subtransform,
        )

    @alias_kwargs(
        ('fo', 'filter_offgrid'),
        ('rf', 'return_filtered'),
        ('s', 'size'),
    )
    def transform_points(
        self,
        points: PointsInput,
        filter_offgrid: bool | None = None,
        return_filtered: bool = False,
        **kwargs,
        ) -> PointsOutputs:
        self.infer_dim(points)
        assert_points_shapes(points, self.__dim)
        points = arg_to_list(points, (np.ndarray, torch.Tensor))
        device = get_group_device(points, device=self.__device)
        return_types = [type(p) for p in points]
        points = [to_tensor(p, device=device, dtype=torch.float32) for p in points]
        filter_offgrid = filter_offgrid if filter_offgrid is not None else self.__filter_offgrid
        other_data = []
        if filter_offgrid and return_filtered:
            indiceses = [to_tensor([], device=device, dtype=torch.int32) for _ in points]
            indiceses = to_return_format(indiceses, return_types=return_types)
            other_data.append(indiceses)
        return to_return_format(points, other_data=other_data, return_types=return_types)
