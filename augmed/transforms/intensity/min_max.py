from __future__ import annotations

import torch
from typing import Tuple

from ...typing import Dist, ImageTensor, Number, TransformParams
from ...utils.args import alias_kwargs, expand_range_arg
from ...utils.assertions import assert_range
from ...utils.conversion import to_tensor, to_tuple
from ...utils.maths import round
from ..identity import Identity
from .intensity import IntensityTransform, RandomIntensityTransform

class MinMax(IntensityTransform):
    @alias_kwargs(
        ('mn', 'min'),
        ('mx', 'max'),
    )
    def __init__(
        self,
        min: Number = 0,
        max: Number = 1,
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
            max=round(self.__max, dp=3),
            min=round(self.__min, dp=3),
            subtransform=subtransform,
        )

    def transform_intensity(
        self,
        image: ImageTensor,
        ) -> ImageTensor:
        if image.dtype == torch.bool:
            return image    # Boolean tensors are unchanged by intensity transforms. 
        image_t = (self.__max - self.__min) * (image - image.min()) / (image.max() - image.min()) + self.__min
        return image_t

class RandomMinMax(RandomIntensityTransform):
    @alias_kwargs(
        ('mn', 'min'),
        ('mx', 'max'),
    )
    def __init__(
        self,
        min: Number | Tuple[Number, ...] = 0,
        max: Number | Tuple[Number, ...] = 1,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        self.__min = min
        self.__max = max
        self.__expand_range_args()

    def __expand_range_args(self) -> None:
        dim = 1    # Dim=1 for all intensity transforms.
        min_range = expand_range_arg(self.__min, dim=dim)
        assert_range(min_range, dim, 'min')
        self.__min_range = to_tensor(min_range)
        max_range = expand_range_arg(self.__max, dim=dim)
        assert_range(max_range, dim, 'max')
        self.__max_range = to_tensor(max_range)

    def freeze(
        self,
        dist: Dist | None = None,
        dist_std: float | None = None,
        ) -> MinMax | Identity:
        should_apply = self.__rng.random(1) < self.__p
        if not should_apply:
            return Identity(dim=self.__dim)

        min_draw = self.draw_from_range(self.__min_range, dist=dist, dist_std=dist_std).item()
        max_draw = self.draw_from_range(self.__max_range, dist=dist, dist_std=dist_std).item()
        params = dict(
            max=max_draw,
            min=min_draw,
        )
        return super().freeze(MinMax, params)

    @property
    def params(self) -> TransformParams:
        return super().params(max=to_tuple(self.__max_range), min=to_tuple(self.__min_range))

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().__str__(
            self.__class__.__name__,
            max=to_tuple(self.__max_range, dp=3),
            min=to_tuple(self.__min_range, dp=3),
            subtransform=subtransform,
        )
