from __future__ import annotations

import numpy as np
import torch
import torch
from typing import Tuple

from ....typing import Dist, Number, TransformParams
from ....utils.args import alias_kwargs
from ....utils.conversion import to_tensor, to_tuple
from ....utils.python import get_private_attr
from ...identity import Identity
from .affine import Affine, RandomAffine

class Translate(Affine):
    @alias_kwargs(
        ('t', 'translation'),
    )
    def __init__(
        self,
        translation: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = None,
        translation_p: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = None,
        **kwargs,
        ) -> None:
        super().__init__(
            rotation=None,
            scaling=None,
            translation=translation,
            translation_p=translation_p,
            **kwargs,
        )

    @property
    def params(self) -> TransformParams:
        return super().super_params(
            backward_translation_matrix=self.__backward_translation_matrix,
            translation=to_tuple(self.__translation) if self.__translation is not None else None,
            translation_matrix=self.__translation_matrix,
            translation_p=to_tuple(self.__translation_p) if self.__translation_p is not None else None,
        )

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().super_str(
            self.__class__.__name__,
            subtransform=subtransform,
            translation=to_tuple(self.__translation, dp=3) if self.__translation is not None else None,
            translation_p=to_tuple(self.__translation_p, dp=3) if self.__translation_p is not None else None,
        )

class RandomTranslate(RandomAffine):
    @alias_kwargs(
        ('t', 'translation'),
    ) 
    def __init__(
        self, 
        translation: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = None,
        translation_p: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = None,
        **kwargs,
        ) -> None:
        super().__init__(
            rotation=None,
            scaling=None,
            translation=translation,
            translation_p=translation_p,
            **kwargs,
        )

    def freeze(
        self,
        dist: Dist | None = None,
        dist_std: float | None = None,
        ) -> Translate | Identity:
        should_apply = self.__rng.random(1) < self.__p
        if not should_apply:
            return Identity(dim=self.__dim)

        # Draw the translation parameters.
        trans_draw = trans_p_draw = None
        if self.__translation_range is not None:
            trans_draw = self.draw_from_range(self.__translation_range, dist=dist, dist_std=dist_std)
        if self.__translation_p_range is not None:
            trans_p_draw = self.draw_from_range(self.__translation_p_range, dist=dist, dist_std=dist_std)

        params = dict(
            translation=trans_draw,
            translation_p=trans_p_draw,
        )
        return super().super_freeze(Translate, params)

    @property
    def params(self) -> TransformParams:
        translation_range = get_private_attr(self, '__translation_range')
        return super().super_params(
            translation=to_tuple(translation_range.T.flatten()) if translation_range is not None else None,
            translation_p=to_tuple(get_private_attr(self, '__translation_p_range').T.flatten()) if get_private_attr(self, '__translation_p_range') is not None else None,
        )

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().super_str(
            self.__class__.__name__,
            subtransform=subtransform,
            translation=to_tuple(get_private_attr(self, '__translation_range').T.flatten(), dp=3) if get_private_attr(self, '__translation_range') is not None else None,
            translation_p=to_tuple(get_private_attr(self, '__translation_p_range').T.flatten(), dp=3) if get_private_attr(self, '__translation_p_range') is not None else None,
        )

