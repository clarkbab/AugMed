from typing import *

from ...typing import *
from ..identity import Identity
from .intensity import IntensityTransform, RandomIntensityTransform

# Min:
# m=-200 -> m=(-200, -200, ...)
# m=(-200, 0) -> m=(-200, 0, ...)
# Max:
# m=1500 -> m=(1500, 1500, ...)
# m=(1500, 1800) -> m=(1500, 1800, ...)
class RandomThreshold(RandomIntensityTransform):
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
            assert len(self.__min_range) == 2, f"Expected 'min' of length 2, got {len(self.__min_range)}."
        self.__max_range = to_tensor(max, broadcast=2) if max is not None else None
        if self.__max_range is not None:
            assert len(self.__max_range) == 2, f"Expected 'max' of length 2, got {len(self.__max_range)}."
        self._params = dict(
            dim=self._dim,
            max=self.__max_range,
            min=self.__min_range,
            p=self._p,
            type=self.__class__.__name__,
        )

    def freeze(self) -> 'Norm':
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random(2))
        min_draw = draw[0] * (self.__min_range[1] - self.__min_range[0]) + self.__min_range[0] if self.__min_range is not None else None
        max_draw = draw[0] * (self.__max_range[1] - self.__max_range[0]) + self.__max_range[0] if self.__max_range is not None else None
        params = dict(
            max=max_draw,
            min=min_draw,
        )
        return super().freeze(Threshold, params)

    def __str__(self) -> str:
        params = dict(
            max=to_tuple(self.__max_range, decimals=3) if self.__max_range is not None else None,
            min=to_tuple(self.__min_range, decimals=3) if self.__min_range is not None else None,
        )
        return super().__str__(self.__class__.__name__, params)

class Threshold(IntensityTransform):
    def __init__(
        self,
        min: Number | None = None,
        max: Number | None = None,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        self.__min = min
        self.__max = max
        self._params = dict(
            dim=self._dim,
            max=self.__max,
            min=self.__min,
            type=self.__class__.__name__,
        )

    def __str__(self) -> str:
        params = dict(
            max=round(self.__max, 3) if self.__max is not None else None,
            min=round(self.__min, 3) if self.__min is not None else None,
        )
        return super().__str__(self.__class__.__name__, params)

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
