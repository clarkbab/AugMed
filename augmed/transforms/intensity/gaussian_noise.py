import numpy as np
import torch
from typing import Tuple

from ...typing import Dist, ImageTensor, Number, SpatialDim, TransformParams
from ...utils.args import alias_kwargs, arg_default, expand_range_arg
from ...utils.assertions import assert_range
from ...utils.conversion import to_tensor, to_tuple
from ...utils.maths import round
from ..identity import Identity
from .intensity import IntensityTransform, RandomIntensityTransform

DEFAULT_MEAN = 0.0
DEFAULT_STD_P = 0.05
DEFAULT_STD_P_RANGE = (0, DEFAULT_STD_P)

# For noise, should this always be mean=0? We could do level boosts with another
# transform that changes the mean level.
# It would be nice to have it here for convenience. So mean=1 will boost the image
# by intensity 1. mean_p=0.1 should boost the image by 10%, while mean_p=-0.1 should
# reduce the image by 10%.
class GaussianNoise(IntensityTransform):
    @alias_kwargs(
        ('m', 'mean'),
        ('mp', 'mean_p'),
        ('s', 'std'),
        ('sp', 'std_p'),
    )
    def __init__(
        self,
        mean: Number | None = None,
        mean_p: Number | None = None,
        std: Number | None = None,
        std_p: Number | None = None,    # Std as proportion of image intensity range.
        **kwargs,
        ) -> None:
        mean_res = arg_default((mean, mean_p), mean, DEFAULT_MEAN)
        mean_p_res = arg_default((mean, mean_p), mean_p, None)
        std_res = arg_default((std, std_p), std, None)
        std_p_res = arg_default((std, std_p), std_p, DEFAULT_STD_P)
        assert (mean_res is not None) ^ (mean_p_res is not None), "Specify either 'mean' or 'mean_p', not both."
        assert (std_res is not None) ^ (std_p_res is not None), "Specify either 'std' or 'std_p', not both."
        super().__init__(**kwargs)
        self.__mean = mean_res
        self.__mean_p = mean_p_res
        self.__std = std_res
        self.__std_p = std_p_res

    @property
    def params(self) -> TransformParams:
        return super().params(
            mean=self.__mean,
            mean_p=self.__mean_p,
            std=self.__std,
            std_p=self.__std_p,
        )

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().__str__(
            self.__class__.__name__,
            mean=round(self.__mean, dp=3) if self.__mean is not None else None,
            mean_p=round(self.__mean_p, dp=3) if self.__mean_p is not None else None,
            std=round(self.__std, dp=3) if self.__std is not None else None,
            std_p=round(self.__std_p, dp=3) if self.__std_p is not None else None,
            subtransform=subtransform,
        )

    def transform_intensity(
        self,
        image: ImageTensor,
        ) -> ImageTensor:
        if image.dtype == torch.bool:
            return image    # Boolean tensors are unchanged by intensity transforms. 

        # Resolve mean/std based on image intensity range.
        if self.__mean is not None:
            mean = self.__mean
        else:
            mean = self.__mean_p * (torch.max(image) - torch.min(image))
        if self.__std is not None:
            std = self.__std
        else:
            std = self.__std_p * (torch.max(image) - torch.min(image))

        # Add noise to the image.
        noise = torch.randn(image.shape, device=image.device, dtype=image.dtype) * std + mean
        return image + noise

class RandomGaussianNoise(RandomIntensityTransform):
    @alias_kwargs(
        ('m', 'mean'),
        ('s', 'std'),
        ('sp', 'std_p'),
    )
    def __init__(
        self,
        mean: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = None,
        mean_p: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = None,
        std: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = None,
        std_p: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = None,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        mean_res = arg_default((mean, mean_p), mean, DEFAULT_MEAN)
        mean_p_res = arg_default((mean, mean_p), mean_p, None)
        std_res = arg_default((std, std_p), std, None)
        std_p_res = arg_default((std, std_p), std_p, DEFAULT_STD_P_RANGE)
        self.__mean = mean_res
        self.__mean_p = mean_p_res
        self.__std = std_res
        self.__std_p = std_p_res
        self.__expand_range_args()

    def __expand_range_args(self) -> None:
        mean_range = mean_p_range = std_range = std_p_range = None
        dim = 1    # Dim=1 for all intensity transforms.
        if self.__mean is not None:
            mean_range = expand_range_arg(self.__mean, dim=dim, negate_lower=True)
            assert_range(mean_range, dim, 'mean')
            mean_range = to_tensor(mean_range)
        if self.__mean_p is not None:
            mean_p_range = expand_range_arg(self.__mean_p, dim=dim, negate_lower=True)
            assert_range(mean_p_range, dim, 'mean_p')
            mean_p_range = to_tensor(mean_p_range)
        if self.__std is not None:
            std_range = expand_range_arg(self.__std, dim=dim, negate_lower=True)
            assert_range(std_range, dim, 'std')
            std_range = to_tensor(std_range)
        if self.__std_p is not None:
            std_p_range = expand_range_arg(self.__std_p, dim=dim, negate_lower=True)
            assert_range(std_p_range, dim, 'std_p')
            std_p_range = to_tensor(std_p_range)
        self.__mean_range = mean_range
        self.__mean_p_range = mean_p_range
        self.__std_range = std_range
        self.__std_p_range = std_p_range

    def freeze(
        self,
        dist: Dist | None = None,
        dist_std: float | None = None,
        ) -> GaussianNoise | Identity:
        should_apply = self.__rng.random(1) < self.__p
        if not should_apply:
            return Identity(dim=self.__dim)

        # Draw param values.
        dim = 1   # Dim=1 for all intensity transforms.
        mean_draw = mean_p_draw = std_draw = std_p_draw = None
        if self.__mean_range is not None:
            mean_draw = self.draw_from_range(self.__mean_range, dist=dist, dist_std=dist_std).item()
        if self.__mean_p_range is not None:
            mean_p_draw = self.draw_from_range(self.__mean_p_range, dist=dist, dist_std=dist_std).item()
        if self.__std_range is not None:
            std_draw = self.draw_from_range(self.__std_range, dist=dist, dist_std=dist_std).item()
        if self.__std_p_range is not None:
            std_p_draw = self.draw_from_range(self.__std_p_range, dist=dist, dist_std=dist_std).item()

        params = dict(
            mean=mean_draw,
            mean_p=mean_p_draw,
            std=std_draw,
            std_p=std_p_draw,
        )
        return super().freeze(GaussianNoise, params)
        # Dim=1 for all intensity transforms.
        # Could we perhaps support dim for some intensity transforms in the future?
        # Image gaussian smoothing, where the std can be different for each spatial dim.
        # self.__expand_range_args()

    @property
    def params(self) -> TransformParams:
        return super().params(
            mean=to_tuple(self.__mean_range.T.flatten()) if self.__mean_range is not None else None,
            mean_p=to_tuple(self.__mean_p_range.T.flatten()) if self.__mean_p_range is not None else None,
            std=to_tuple(self.__std_range.T.flatten()) if self.__std_range is not None else None,
            std_p=to_tuple(self.__std_p_range.T.flatten()) if self.__std_p_range is not None else None,
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
            mean_p=to_tuple(self.__mean_p_range.T.flatten(), dp=3) if self.__mean_p_range is not None else None,
            std=to_tuple(self.__std_range.T.flatten(), dp=3) if self.__std_range is not None else None,
            std_p=to_tuple(self.__std_p_range.T.flatten(), dp=3) if self.__std_p_range is not None else None,
            subtransform=subtransform,
        )
