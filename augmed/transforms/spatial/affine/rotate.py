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
        super().set_params(
            self.__class__.__name__,
            backward_matrix=self._backward_rotation_matrix,
            matrix=self._rotation_matrix,
            rotation=self._rotation,
            rotation_centre=self._rotation_centre,
            rotation_rad=self._rotation_rad,
        )

    def __str__(self) -> str:
        return super().super_str(
            self.__class__.__name__,
            rotation=to_tuple(self._rotation, decimals=3),
            rotation_centre=to_tuple(self._rotation_centre, decimals=3) if self._rotation_centre != 'image-centre' else "\"image-centre\"",
        )

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
        super().set_params(
            self.__class__.__name__,
            rotation=self._rotation,
            rotation_centre=self._rotation_centre,
        )

    def freeze(self) -> Rotate:
        # Expand the range args.
        # We do this now because 'set_dim' could be called after RandomRotate.__init__.
        rotation_range = expand_range_arg(self._rotation, dim=self._dim, negate_lower=True)
        assert len(rotation_range) == 2 * self._dim, f"Expected 'rotation' of length {2 * self._dim}, got {len(rotation_range)}."
        rotation_range = to_tensor(rotation_range).reshape(self._dim, 2)

        # Draw the rotation parameters.
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random(self._dim))
        rot_draw = draw * (rotation_range[:, 1] - rotation_range[:, 0]) + rotation_range[:, 0]
        params = dict(
            rotation=rot_draw,
            rotation_centre=self._rotation_centre,
        )
        return super().super_freeze(Rotate, params)

    def __str__(self) -> str:
        return super().super_str(
            self.__class__.__name__,
            rotation=self._rotation,
            rotation_centre=self._rotation_centre,
        )
