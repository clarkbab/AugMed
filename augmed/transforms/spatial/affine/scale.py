import torch
from typing import *

from ....typing import *
from ....utils.conversion import to_tensor, to_tuple
from ...identity import Identity
from .affine import Affine, RandomAffine

class Scale(Affine):
    def __init__(
        self,
        scaling: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor,
        scaling_centre: Point | Literal['image-centre'] = 'image-centre',
        **kwargs,
        ) -> None:
        super().__init__(
            rotation=None,
            scaling=scaling,
            scaling_centre=scaling_centre,
            translation=None,
            **kwargs,
        )
        self._params = dict(
            backward_matrix=self._backward_scaling_matrix,
            dim=self._dim,
            matrix=self._scaling_matrix,
            scaling=self._scaling,
            scaling_centre=self._scaling_centre,
            type=self.__class__.__name__,
        )

    def __str__(self) -> str:
        params = dict(
            scaling=to_tuple(self._scaling, decimals=3),
            scaling_centre=to_tuple(self._scaling_centre, decimals=3) if self._scaling_centre != 'image-centre' else "\"image-centre\"",
        )
        return super().super_str(self.__class__.__name__, params)

class RandomScale(RandomAffine):
    def __init__(
        self, 
        scaling: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = (0.8, 1.2),
        scaling_centre: Point | Literal['image-centre'] = 'image-centre',
        **kwargs,
        ) -> None:
        super().__init__(
            rotation=None,
            rotation_centre=None,
            scaling=scaling,
            scaling_centre=scaling_centre,
            translation=None,
            **kwargs,
        )
        self._params = dict(
            dim=self._dim,
            p=self._p,
            scaling=self._scaling_range,
            scaling_centre=self._scaling_centre,
            type=self.__class__.__name__,
        )

    def freeze(self) -> 'Scale':
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random(self._dim))
        scale_draw = draw * (self._scaling_range[:, 1] - self._scaling_range[:, 0]) + self._scaling_range[:, 0]
        params = dict(
            scaling=scale_draw,
            scaling_centre=self._scaling_centre,
        )
        return super().super_freeze(Scale, params)

    def __str__(self) -> str:
        params = dict(
            scaling=to_tuple(self._scaling_range.flatten(), decimals=3) if self._scaling_range is not None else None,
            scaling_centre=to_tuple(self._scaling_centre, decimals=3) if self._scaling_centre != 'image-centre' else "\"image-centre\"",
        )
        return super().super_str(self.__class__.__name__, params)
