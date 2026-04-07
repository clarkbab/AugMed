from __future__ import annotations

import torch
from typing import Tuple

from ...typing import ImageTensor, Number
from ...utils.args import expand_range_arg
from ...utils.conversion import to_tensor, to_tuple
from ..identity import Identity
from .intensity import IntensityTransform, RandomIntensityTransform

class MinMax(IntensityTransform):
    def __init__(
        self,
        min: Number = 0,
        max: Number = 1,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        self.__min = min
        self.__max = max
        super().set_params(
            self.__class__.__name__,
            max=self.__max,
            min=self.__min,
        )

    def __str__(self) -> str:
        return super().__str__(
            self.__class__.__name__,
            max=round(self.__max, 3),
            min=round(self.__min, 3),
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
    def __init__(
        self,
        min: Number | Tuple[Number, ...] = (-0.2, 0.2),
        max: Number | Tuple[Number, ...] = (0.8, 1.2),
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        min_range = expand_range_arg(min, dim=1)
        assert len(min_range) == 2, f"Expected 'min' of length 2, got {len(min_range)}."
        self.__min_range = to_tensor(min_range, dtype=torch.float32)
        max_range = expand_range_arg(max, dim=1)
        assert len(max_range) == 2, f"Expected 'max' of length 2, got {len(max_range)}."
        self.__max_range = to_tensor(max_range, dtype=torch.float32)
        super().set_params(
            self.__class__.__name__,
            max=self.__max_range,
            min=self.__min_range,
        )

    def freeze(self) -> MinMax:
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random(2), dtype=torch.float32)
        min_draw = draw[0] * (self.__min_range[1] - self.__min_range[0]) + self.__min_range[0]
        max_draw = draw[0] * (self.__max_range[1] - self.__max_range[0]) + self.__max_range[0]
        params = dict(
            max=max_draw.item(),
            min=min_draw.item(),
        )
        return super().freeze(MinMax, params)

    def __str__(self) -> str:
        return super().__str__(
            self.__class__.__name__,
            max=to_tuple(self.__max_range, decimals=3),
            min=to_tuple(self.__min_range, decimals=3),
        )
