from __future__ import annotations

import numpy as np
import torch
from typing import Literal, Tuple

from ....config import get_dim
from ....typing import Number, Point, SpatialDim, TransformParams
from ....utils.args import alias_kwargs, arg_default, arg_to_list
from ....utils.conversion import to_tensor, to_tuple
from ...identity import Identity
from .affine import Affine, RandomAffine
        
class Flip(Affine):
    @alias_kwargs(
        ('f', 'flip'),
        ('c', 'centre'),
        ('co', 'centre_offset'),
    )
    def __init__(
        self,
        flip: bool | Tuple[bool] | np.ndarray | torch.Tensor,
        centre: Point | Literal['image-centre'] = 'image-centre',
        centre_offset: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 0.0,
        dim: SpatialDim | None = None,
        **kwargs,
        ) -> None:
        # SpatialDim is defined in superclass, but we need to know "scaling" first for parent class.
        # Let parent handle the extension? We can't do this, it'll be confusing talking about
        # "scaling" instead of "flip" to the user.
        dim = arg_default(dim, dim, get_dim())
        self.__dim = dim
        self.__flip = to_tensor(flip, broadcast=dim, dtype=torch.bool)
        assert len(self.__flip) == self.__dim, f"Expected 'flip' of length {self.__dim} for dim={self.__dim}, got {len(self.__flip)}."
        scaling = [-1 if f else 1 for f in self.__flip]
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
        return super().params(flip=to_tuple(self.__flip))

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().super_str(
            self.__class__.__name__,
            centre=to_tuple(self.__centre, dp=3) if self.__centre != 'image-centre' else "\"image-centre\"",
            centre_offset=to_tuple(self.__centre_offset, dp=3),
            flip=to_tuple(self.__flip),
            subtransform=subtransform,
        )

# This might not be a random affine, which expects a scaling range.
class RandomFlip(RandomAffine):
    @alias_kwargs(
        ('f', 'flip'),
        ('c', 'centre'),
        ('co', 'centre_offset'),
    )
    def __init__(
        # TODO: Look into using Affine (superclass) __init__ here to save 'flip'
        # as scaling values. Might be a bit tricky, as this is typically a range.
        # Is a flip even a true affine? Maybe only at the deterministic level.
        self,
        flip: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 0.5,
        centre: Point | Literal['image-centre'] = 'image-centre',
        centre_offset: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 0.0,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        self.__flip = flip
        self.__centre = centre
        self.__centre_offset = centre_offset
        self.__expand_args()

    def __expand_args(self) -> None:
        self.__flip = to_tensor(arg_to_list(self.__flip, (int, float), broadcast=self.__dim))

    def freeze(self) -> Flip | Identity:
        should_apply = self.__rng.random() < self.__p
        if not should_apply:
            return Identity(dim=self.__dim)

        draw = to_tensor(self.__rng.random(self.__dim))
        flip_draw = draw < self.__flip
        params = dict(
            centre=self.__centre,
            centre_offset=self.__centre_offset,
            flip=flip_draw,
        )
        return super().freeze(Flip, params)

    @property
    def params(self) -> TransformParams:
        return super().params(flip=to_tuple(self.__flip))

    def set_dim(
        self,
        dim: SpatialDim,
        ) -> None:
        super().set_dim(dim)
        self.__expand_args()

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().__str__(
            self.__class__.__name__,
            centre=to_tuple(self.__centre, dp=3) if self.__centre != 'image-centre' else "\"image-centre\"",
            centre_offset=to_tuple(self.__centre_offset, dp=3),
            flip=to_tuple(self.__flip),
            subtransform=subtransform,
        )
