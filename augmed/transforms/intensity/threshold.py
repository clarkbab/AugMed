from __future__ import annotations

import torch
from typing import Tuple

from ...typing import Dist, ImageTensor, Number, TransformParams
from ...utils.args import alias_kwargs
from ...utils.assertions import assert_range
from ...utils.conversion import to_tensor, to_tuple
from ...utils.maths import round
from ..identity import Identity
from .intensity import IntensityTransform, RandomIntensityTransform

# Min:
# m=-200 -> m=(-200, -200, ...)
# m=(-200, 0) -> m=(-200, 0, ...)
# Max:
# m=1500 -> m=(1500, 1500, ...)
# m=(1500, 1800) -> m=(1500, 1800, ...)
class RandomThreshold(RandomIntensityTransform):
    @alias_kwargs(
        ('mn', 'min'),
        ('mx', 'max'),
    )
    def __init__(
        self,
        min: Number | Tuple[Number, ...] | None = None,
        max: Number | Tuple[Number, ...] | None = None,
        # TODO: Add literals to express reasonable defaults.
        # modality: Optional[Literal['ct', 'mr', 'pet']] = None,
        # that sets the min/max range to something like (-1000, 2000) for CT.
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        self.__min_range = to_tensor(min, broadcast=2) if min is not None else None
        if self.__min_range is not None:
            assert_range(self.__min_range, 1, 'min')
        self.__max_range = to_tensor(max, broadcast=2) if max is not None else None
        if self.__max_range is not None:
            assert_range(self.__max_range, 1, 'max')

    def freeze(
        self,
        dist: Dist | None = None,
        dist_std: float | None = None,
        ) -> Threshold | Identity:
        should_apply = self.__rng.random(1) < self.__p
        if not should_apply:
            return Identity(dim=self.__dim)

        min_draw = self.draw_from_range(self.__min_range, dist=dist, dist_std=dist_std).item() if self.__min_range is not None else None
        max_draw = self.draw_from_range(self.__max_range, dist=dist, dist_std=dist_std).item() if self.__max_range is not None else None
        params = dict(
            max=max_draw,
            min=min_draw,
        )
        return super().freeze(Threshold, params)

    @property
    def params(self) -> TransformParams:
        return super().params(
            max=to_tuple(self.__max_range) if self.__max_range is not None else None,
            min=to_tuple(self.__min_range) if self.__min_range is not None else None,
        )

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().__str__(
            self.__class__.__name__,
            max=to_tuple(self.__max_range, dp=3) if self.__max_range is not None else None,
            min=to_tuple(self.__min_range, dp=3) if self.__min_range is not None else None,
            subtransform=subtransform,
        )

class Threshold(IntensityTransform):
    @alias_kwargs(
        ('mn', 'min'),
        ('mx', 'max'),
    )
    def __init__(
        self,
        min: Number | None = None,
        max: Number | None = None,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        self.__min = min
        self.__max = max

    @property
    def params(self) -> TransformParams:
        return super().params(max=self.__max, min=self.__min)

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().__str__(
            self.__class__.__name__,
            max=round(self.__max, dp=3) if self.__max is not None else None,
            min=round(self.__min, dp=3) if self.__min is not None else None,
            subtransform=subtransform,
        )

    def transform_intensity(
        self,
        image: ImageTensor,
        ) -> ImageTensor:
        if image.dtype == torch.bool:
            return image    # Boolean tensors are unchanged by intensity transforms. 
        image_t = image.clone()
        if self.__min is not None:
            image_t[image_t < self.__min] = self.__min
        if self.__max is not None:
            image_t[image_t > self.__max] = self.__max
        return image_t
