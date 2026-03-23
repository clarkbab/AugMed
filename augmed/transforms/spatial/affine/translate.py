import numpy as np
import torch
import torch
from typing import Tuple

from ....typing import Affine, Number
from ....utils.conversion import to_tensor, to_tuple
from ...identity import Identity
from .affine import RandomAffine

class Translate(Affine):
    def __init__(
        self,
        translation: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor,
        **kwargs,
        ) -> None:
        super().__init__(
            rotation=None,
            scaling=None,
            translation=translation,
            **kwargs,
        )
        self._params = dict(
            backward_matrix=self._backward_translation_matrix,
            dim=self._dim,
            matrix=self._translation_matrix,
            translation=self._translation,
            type=self.__class__.__name__,
        )

    def __str__(self) -> str:
        params = dict(
            translation=to_tuple(self._translation, decimals=3),
        )
        return super().super_str(self.__class__.__name__, params)

class RandomTranslate(RandomAffine):
    def __init__(
        self, 
        translation: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = 50.0,
        **kwargs,
        ) -> None:
        super().__init__(
            rotation=None,
            scaling=None,
            translation=translation,
            **kwargs,
        )
        self._params = dict(
            dim=self._dim,
            p=self._p,
            translation=self._translation_range,
            type=self.__class__.__name__,
        )

    def freeze(self) -> 'Translate':
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random(self._dim))
        trans_draw = draw * (self._translation_range[:, 1] - self._translation_range[:, 0]) + self._translation_range[:, 0]
        params = dict(
            translation=trans_draw,
        )
        return super().super_freeze(Translate, params)

    def __str__(self) -> str:
        params = dict(
            translation=to_tuple(self._translation_range.flatten(), decimals=3) if self._translation_range is not None else None,
        )
        return super().super_str(self.__class__.__name__, params)
