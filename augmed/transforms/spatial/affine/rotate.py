from __future__ import annotations

import numpy as np
import torch
import torch
from typing import Literal, Tuple

from ....typing import Dist, Number, Point, TransformParams
from ....utils.args import alias_kwargs
from ....utils.conversion import to_tensor, to_tuple
from ....utils.python import get_private_attr, wrap_quotes
from ...identity import Identity
from .affine import Affine, DEFAULT_ROTATION_RANGE, RandomAffine

class Rotate(Affine):
    @alias_kwargs(
        ('r', 'rotation'),
        ('c', 'centre'),
        ('co', 'centre_offset'),
    )
    def __init__(
        self,
        rotation: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 0.0,
        centre: Point | Literal['image-centre'] = 'image-centre',
        centre_offset: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 0.0,
        **kwargs,
        ) -> None:
        super().__init__(
            rotation=rotation,
            rotation_centre=centre,
            rotation_centre_offset=centre_offset,
            scaling=None,
            translation=None,
            **kwargs,
        )

    @property
    def params(self) -> TransformParams:
        return super().super_params(
            backward_rotation_matrix=get_private_attr(self, '__backward_rotation_matrix'),
            rotation=to_tuple(get_private_attr(self, '__rotation')) if get_private_attr(self, '__rotation') is not None else None,
            rotation_centre=get_private_attr(self, '__rotation_centre'),
            rotation_centre_offset=to_tuple(get_private_attr(self, '__rotation_centre_offset')),
            rotation_matrix=get_private_attr(self, '__rotation_matrix'),
            rotation_rad=to_tuple(get_private_attr(self, '__rotation_rad')) if get_private_attr(self, '__rotation_rad') is not None else None,
        )

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().super_str(
            self.__class__.__name__,
            centre=to_tuple(self.__centre, dp=3) if self.__centre != 'image-centre' else wrap_quotes('image-centre'),
            centre_offset=to_tuple(self.__centre_offset, dp=3),
            rotation=to_tuple(self.__rotation, dp=3),
            subtransform=subtransform,
        )

class RandomRotate(RandomAffine):
    @alias_kwargs(
        ('c', 'centre'),
        ('co', 'centre_offset'),
        ('r', 'rotation'),
    )
    def __init__(
        self, 
        rotation: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = DEFAULT_ROTATION_RANGE,
        centre: Point | Literal['image-centre'] = 'image-centre',
        centre_offset: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 0.0,
        **kwargs,
        ) -> None:
        super().__init__(
            rotation=rotation,
            rotation_centre=centre,
            rotation_centre_offset=centre_offset,
            scaling=None,
            translation=None,
            **kwargs,
        )

    def freeze(
        self,
        dist: Dist | None = None,
        dist_std: float | None = None,
        ) -> Rotate | Identity:
        rotation_range = get_private_attr(self, '__rotation_range')
        should_apply = self.__rng.random(1) < self.__p
        if not should_apply:
            return Identity(dim=self.__dim)

        rot_draw = self.draw_from_range(rotation_range, dist=dist, dist_std=dist_std)
        params = dict(
            centre=self.__centre,
            centre_offset=self.__centre_offset,
            rotation=rot_draw,
        )
        return super().super_freeze(Rotate, params)

    @property
    def params(self) -> TransformParams:
        rotation_range = get_private_attr(self, '__rotation_range')
        rotation_centre_offset_range = get_private_attr(self, '__rotation_centre_offset_range')
        return super().super_params(
            rotation=to_tuple(rotation_range.T.flatten()) if rotation_range is not None else None,
            rotation_centre=get_private_attr(self, '__rotation_centre'),
            rotation_centre_offset=to_tuple(rotation_centre_offset_range.T.flatten()),
        )

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().super_str(
            self.__class__.__name__,
            centre=to_tuple(self.__rotation_centre, dp=3) if self.__rotation_centre != 'image-centre' else wrap_quotes('image-centre'),
            centre_offset=to_tuple(self.__rotation_centre_offset, dp=3),
            rotation=to_tuple(self.__rotation, dp=3),
            subtransform=subtransform,
        )
