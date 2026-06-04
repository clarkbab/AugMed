from __future__ import annotations

import numpy as np
from skimage.exposure import equalize_hist
import torch

from ...typing import ImageTensor, TransformParams
from ...utils.conversion import to_numpy, to_tensor
from .intensity import IntensityTransform

class HistEq(IntensityTransform):
    def __init__(
        self,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)

    @property
    def params(self) -> TransformParams:
        return super().params()

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

    def transform_intensity(
        self,
        image: ImageTensor,
        ) -> ImageTensor:
        if image.dtype == torch.bool or image.dtype == np.bool_:
            return image    # Boolean tensors are unchanged by intensity transforms.

        # TODO: Preserve gradients through histogram equalisation.
        image = to_numpy(image)
        image_t = equalize_hist(image)
        image_t = to_tensor(image_t, device=image.device, dtype=torch.float32)
        return image_t
