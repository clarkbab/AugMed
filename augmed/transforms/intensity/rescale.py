from typing import *

from ...typing import *
from ..identity import Identity
from .intensity import IntensityTransform, RandomIntensityTransform

# Min:
# m=0.2 -> m=(0.2, 0.2, ...)
# m=(-0.2, 0.2) -> m=(-0.2, 0.2, ...)
# Max:
# m=1.5 -> m=
class RandomRescale(RandomIntensityTransform):
    def __init__(
        self,
        min: Number | Tuple[Number, ...] = 0,
        max: Number | Tuple[Number, ...] = 1,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        self.__min_range = to_tensor(min, broadcast=2)
        assert len(self.__min_range) == 2, f"Expected 'min' of length 2, got {len(self.__min_range)}."
        self.__max_range = to_tensor(max, broadcast=2)
        assert len(self.__max_range) == 2, f"Expected 'max' of length 2, got {len(self.__max_range)}."
        self._params = dict(
            type=self.__class__.__name__,
            dim=self._dim,
            max=self.__max_range,
            min=self.__min_range,
            p=self._p,
        )

    def freeze(self) -> 'Norm':
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random(2))
        min_draw = draw[0] * (self.__min_range[1] - self.__min_range[0]) + self.__min_range[0]
        max_draw = draw[0] * (self.__max_range[1] - self.__max_range[0]) + self.__max_range[0]
        params = dict(
            min=min_draw,
            max=max_draw,
        )
        return super().freeze(Rescale, params)

    def __str__(self) -> str:
        params = dict(
            min=to_tuple(self.__min_range, decimals=3),
            max=to_tuple(self.__max_range, decimals=3),
        )
        return super().__str__(self.__class__.__name__, params)

class Rescale(IntensityTransform):
    def __init__(
        self,
        min: Number = 0,
        max: Number = 1,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        self.__min = min
        self.__max = max
        self._params = dict(
            type=self.__class__.__name__,
            dim=self._dim,
            max=self.__max,
            min=self.__min,
        )

    def __str__(self) -> str:
        params = dict(
            min=round(self.__min, 3),
            max=round(self.__max, 3),
        )
        return super().__str__(self.__class__.__name__, params)

    def transform_intensity(
        self,
        image: ImageTensor,
        ) -> ImageTensor:
        if image.dtype == torch.bool:
            return image    # Boolean tensors are unchanged by intensity transforms. 
        print('rescaling')
        print(image.shape)
        print(self.__min, self.__max)
        print(image.min(), image.max())
        image_t = (self.__max - self.__min) * (image - image.min()) / (image.max() - image.min()) + self.__min
        print(image_t.min(), image_t.max())
        print(image_t.shape)
        return image_t
