from .spatial import SpatialTransform

from ...typing import *

# This is really just a utility class for breaking affine chains in the pipeline
# for testing purposes. It doesn't actually move objects.
class BreakAffineChain(SpatialTransform):
    def __init__(
        self,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        self._params = dict(
            type=self.__class__.__name__,
            dim=self._dim,
        )

    def backward_transform_points(
        self,
        points: PointsTensor,
        **kwargs,
        ) -> PointsTensor:
        return points

    def __str__(self) -> str:
        params = dict()
        return super().__str__(self.__class__.__name__, params)

    def transform_points(
        self,
        points: Points,
        **kwargs,
        ) -> Points:
        return points
