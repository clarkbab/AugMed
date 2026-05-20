from __future__ import annotations

import numpy as np
import torch
import torch
from typing import Literal, Tuple

from ....typing import Number, Point, TransformParams
from ....utils.args import alias_kwargs
from ....utils.conversion import to_tensor, to_tuple
from ....utils.python import get_private_attr, wrap_quotes
from ...identity import Identity
from .affine import Affine, DEFAULT_SCALING_RANGE, RandomAffine

class Scale(Affine):
    @alias_kwargs(
        ('c', 'centre'),
        ('co', 'centre_offset'),
        ('s', 'scaling'),
    )
    def __init__(
        self,
        scaling: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 1.0,
        centre: Point | Literal['image-centre'] = 'image-centre',
        centre_offset: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 0.0,
        **kwargs,
        ) -> None:
        super().__init__(
            rotation=None,
            scaling=scaling,
            scaling_centre=centre,
            scaling_centre_offset=centre_offset,
            translation=None,
            **kwargs,
        )

    @property
    def params(self) -> TransformParams:
        return super().super_params(
            backward_scaling_matrix=get_private_attr(self, '__backward_scaling_matrix'),
            scaling=to_tuple(get_private_attr(self, '__scaling')) if get_private_attr(self, '__scaling') is not None else None,
            scaling_centre=get_private_attr(self, '__scaling_centre'),
            scaling_centre_offset=to_tuple(get_private_attr(self, '__scaling_centre_offset')),
            scaling_matrix=get_private_attr(self, '__scaling_matrix'),
        )

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().super_str(
            self.__class__.__name__,
            scaling=to_tuple(self.__scaling, dp=3),
            centre=to_tuple(self.__scaling_centre, dp=3) if self.__scaling_centre != 'image-centre' else wrap_quotes('image-centre'),
            centre_offset=to_tuple(self.__scaling_centre_offset, dp=3),
            subtransform=subtransform,
        )

class RandomScale(RandomAffine):
    @alias_kwargs(
        ('s', 'scaling'),
        ('c', 'centre'),
        ('co', 'centre_offset'),
    )
    def __init__(
        self, 
        scaling: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor | None = DEFAULT_SCALING_RANGE,
        centre: Point | Literal['image-centre'] = 'image-centre',
        centre_offset: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 0.0,
        **kwargs,
        ) -> None:
        super().__init__(
            rotation=None,
            rotation_centre=None,
            scaling=scaling,
            scaling_centre=centre,
            scaling_centre_offset=centre_offset,
            translation=None,
            **kwargs,
        )

    def freeze(self) -> Scale | Identity:
        scaling_range = get_private_attr(self, '__scaling_range')
        should_apply = self.__rng.random(1) < self.__p
        if not should_apply:
            return Identity(dim=self.__dim)

        draw = to_tensor(self.__rng.random(self.__dim))
        scale_draw = draw * (scaling_range[1] - scaling_range[0]) + scaling_range[0]
        params = dict(
            centre=self.__scaling_centre,
            centre_offset=self.__scaling_centre_offset,
            scaling=scale_draw,
        )
        return super().super_freeze(Scale, params)

    @property
    def params(self) -> TransformParams:
        scaling_range = get_private_attr(self, '__scaling_range')
        scaling_centre_offset_range = get_private_attr(self, '__scaling_centre_offset_range')
        return super().super_params(
            scaling=to_tuple(scaling_range.T.flatten()) if scaling_range is not None else None,
            scaling_centre=get_private_attr(self, '__scaling_centre'),
            scaling_centre_offset=to_tuple(scaling_centre_offset_range.T.flatten()),
        )

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().super_str(
            self.__class__.__name__,
            scaling=to_tuple(self.__scaling, dp=3),
            centre=to_tuple(self.__scaling_centre, dp=3) if self.__scaling_centre != 'image-centre' else wrap_quotes('image-centre'),
            centre_offset=to_tuple(self.__scaling_centre_offset, dp=3),
            subtransform=subtransform,
        )
