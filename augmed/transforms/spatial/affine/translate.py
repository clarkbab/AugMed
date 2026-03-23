import numpy as np
import torch
import torch
from typing import Tuple

from ....typing import Number
from ....utils.args import expand_range_arg
from ....utils.conversion import to_tensor, to_tuple
from ...identity import Identity
from .affine import Affine, RandomAffine

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
        super().set_params(
            self.__class__.__name__,
            backward_matrix=self._backward_translation_matrix,
            matrix=self._translation_matrix,
            translation=self._translation,
        )

    def __str__(self) -> str:
        return super().super_str(
            self.__class__.__name__,
            translation=to_tuple(self._translation, decimals=3),
        )

class RandomTranslate(RandomAffine):
    def __init__(
        self, 
        translation: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = 20.0,
        **kwargs,
        ) -> None:
        super().__init__(
            rotation=None,
            scaling=None,
            translation=translation,
            **kwargs,
        )
        super().set_params(
            self.__class__.__name__,
            translation=self._translation,
        )

    def freeze(self) -> 'Translate':
        # Expand the range args.
        # We do this now because 'set_dim' could be called after RandomTranslate.__init__.
        translation_range = expand_range_arg(self._translation, dim=self._dim, negate_lower=True)
        assert len(translation_range) == 2 * self._dim, f"Expected 'translation' of length {2 * self._dim}, got {len(translation_range)}."
        translation_range = to_tensor(translation_range).reshape(self._dim, 2)

        # Draw the translation parameters.
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random(self._dim))
        trans_draw = draw * (translation_range[:, 1] - translation_range[:, 0]) + translation_range[:, 0]
        params = dict(
            translation=trans_draw,
        )
        return super().super_freeze(Translate, params)

    def __str__(self) -> str:
        return super().super_str(
            self.__class__.__name__,
            translation=self._translation,
        )
