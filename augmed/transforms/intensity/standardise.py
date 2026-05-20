import numpy as np
import torch
from typing import Tuple

from ...typing import ImageTensor, Number, SpatialDim, TransformParams
from ...utils.args import alias_kwargs, expand_range_arg
from ...utils.conversion import to_tensor, to_tuple
from ...utils.maths import round
from ..identity import Identity
from .intensity import IntensityTransform, RandomIntensityTransform

class Standardise(IntensityTransform):
    @alias_kwargs(
        ('m', 'mean'),
        ('s', 'std'),
    )
    def __init__(
        self,
        mean: Number = 0.0,
        std: Number = 1.0,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        self.__mean = mean
        self.__std = std

    @property
    def params(self) -> TransformParams:
        return super().params(
            mean=self.__mean,
            std=self.__std,
        )

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().__str__(
            self.__class__.__name__,
            mean=round(self.__mean, dp=3),
            std=round(self.__std, dp=3),
            subtransform=subtransform,
        )

    def transform_intensity(
        self,
        image: ImageTensor,
        ) -> ImageTensor:
        if image.dtype == torch.bool:
            return image    # Boolean tensors are unchanged by intensity transforms.

        # Standardise the image.
        image_mean = image.mean()
        image_std = image.std()
        if image_std == 0:
            image = image - image_mean + self.__mean
        else:
            image = self.__std * (image - image_mean) / image_std + self.__mean

        return image

# What is the purpose of this class?
# Standardise is used to normalise for network input for example, we don't really
# want to randomise this.
class RandomStandardise(RandomIntensityTransform):
    @alias_kwargs(
        ('m', 'mean'),
        ('s', 'std'),
    )
    def __init__(
        self,
        mean: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = (0.0, 0.0),
        std: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = (1.0, 1.0),
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        self.__mean = mean
        self.__std = std
        self.__expand_range_args()

    def __expand_range_args(self) -> None:
        dim = 1    # Dim=1 for all intensity transforms.
        mean_range = expand_range_arg(self.__mean, dim=dim, negate_lower=True)
        assert len(mean_range) == 2 * dim, f"Expected 'mean' of length {2 * dim}, got {len(mean_range)}."
        self.__mean_range = to_tensor(mean_range)
        std_range = expand_range_arg(self.__std, dim=dim)
        assert len(std_range) == 2 * dim, f"Expected 'std' of length {2 * dim}, got {len(std_range)}."
        self.__std_range = to_tensor(std_range)

    def freeze(self) -> Standardise | Identity:
        should_apply = self.__rng.random(1) < self.__p
        if not should_apply:
            return Identity(dim=self.__dim)

        dim = 1    # Dim=1 for all intensity transforms.
        draw = to_tensor(self.__rng.random(dim))
        mean_draw = ((self.__mean_range[1] - self.__mean_range[0]) * draw + self.__mean_range[0]).item()
        draw = to_tensor(self.__rng.random(dim))
        std_draw = ((self.__std_range[1] - self.__std_range[0]) * draw + self.__std_range[0]).item()

        params = dict(
            mean=mean_draw,
            std=std_draw,
        )
        return super().freeze(Standardise, params)
        # Dim=1 for all intensity transforms.
        # self.__expand_range_args()

    @property
    def params(self) -> TransformParams:
        return super().params(
            mean=to_tuple(self.__mean_range),
            std=to_tuple(self.__std_range),
        )

    def set_dim(
        self,
        dim: SpatialDim,
        ) -> None:
        super().set_dim(dim)

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().__str__(
            self.__class__.__name__,
            mean=to_tuple(self.__mean_range.T.flatten(), dp=3) if self.__mean_range is not None else None,
            std=to_tuple(self.__std_range.T.flatten(), dp=3) if self.__std_range is not None else None,
            subtransform=subtransform,
        )
