from __future__ import annotations

import numpy as np
import torch
import torch
from typing import Literal, Tuple

from ....typing import Number, Point
from ....utils.args import expand_range_arg
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
        super().set_params(
            self.__class__.__name__,
            backward_matrix=self._backward_scaling_matrix,
            matrix=self._scaling_matrix,
            scaling=self._scaling,
            scaling_centre=self._scaling_centre,
        )

    def __str__(self) -> str:
        return super().super_str(
            self.__class__.__name__,
            scaling=to_tuple(self._scaling, decimals=3),
            scaling_centre=to_tuple(self._scaling_centre, decimals=3) if self._scaling_centre != 'image-centre' else "\"image-centre\"",
        )

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
        super().set_params(
            self.__class__.__name__,
            scaling=self._scaling,
            scaling_centre=self._scaling_centre,
        )

    def freeze(self) -> Scale:
        # Expand the range args.
        # We do this now because 'set_dim' could be called after RandomScale.__init__.
        scaling_range = expand_range_arg(self._scaling, dim=self._dim, negate_lower=False)
        assert len(scaling_range) == 2 * self._dim, f"Expected 'scaling' of length {2 * self._dim}, got {len(scaling_range)}."
        scaling_range = to_tensor(scaling_range).reshape(self._dim, 2)

        # Draw the scaling parameters.
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random(self._dim))
        scale_draw = draw * (scaling_range[:, 1] - scaling_range[:, 0]) + scaling_range[:, 0]
        params = dict(
            scaling=scale_draw,
            scaling_centre=self._scaling_centre,
        )
        return super().super_freeze(Scale, params)

    def __str__(self) -> str:
        return super().super_str(
            self.__class__.__name__,
            scaling=self._scaling,
            scaling_centre=self._scaling_centre,
        )
