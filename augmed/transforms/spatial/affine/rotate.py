import numpy as np
import torch
import torch
from typing import Literal, Tuple

from ....typing import Number, Point
from ....utils.conversion import to_tensor, to_tuple
from ...identity import Identity
from .affine import Affine, RandomAffine

class Rotate(Affine):
    def __init__(
        self,
        rotation: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor,
        rotation_centre: Point | Literal['image-centre'] = 'image-centre',
        **kwargs,
        ) -> None:
        super().__init__(
            rotation=rotation,
            rotation_centre=rotation_centre,
            scaling=None,
            translation=None,
            **kwargs,
        )
        self._params = dict(
            backward_matrix=self._backward_rotation_matrix,
            dim=self._dim,
            matrix=self._rotation_matrix,
            rotation=self._rotation,
            rotation_centre=self._rotation_centre,
            rotation_rad=self._rotation_rad,
            type=self.__class__.__name__,
        )

    def __str__(self) -> str:
        params = dict(
            rotation=to_tuple(self._rotation, decimals=3),
            rotation_centre=to_tuple(self._rotation_centre, decimals=3) if self._rotation_centre != 'image-centre' else "\"image-centre\"",
        )
        return super().super_str(self.__class__.__name__, params)

class RandomRotate(RandomAffine):
    def __init__(
        self, 
        rotation: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = 15.0,
        rotation_centre: Point | Literal['image-centre'] = 'image-centre',
        **kwargs,
        ) -> None:
        super().__init__(
            rotation=rotation,
            rotation_centre=rotation_centre,
            scaling=None,
            translation=None,
            **kwargs,
        )
        self._params = dict(
            dim=self._dim,
            p=self._p,
            rotation=self._rotation_range,
            rotation_centre=self._rotation_centre,
            type=self.__class__.__name__,
        )

    def freeze(self) -> 'Rotate':
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random(self._dim))
        rot_draw = draw * (self._rotation_range[:, 1] - self._rotation_range[:, 0]) + self._rotation_range[:, 0]
        params = dict(
            rotation=rot_draw,
            rotation_centre=self._rotation_centre,
        )
        return super().super_freeze(Rotate, params)

    def __str__(self) -> str:
        params = dict(
            rotation=to_tuple(self._rotation_range.flatten(), decimals=3) if self._rotation_range is not None else None,
            rotation_centre=to_tuple(self._rotation_centre, decimals=3) if self._rotation_centre != 'image-centre' else "\"image-centre\"",
        )
        return super().super_str(self.__class__.__name__, params)
