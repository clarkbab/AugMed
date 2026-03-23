import numpy as np
import torch
from typing import Literal, Tuple

from ....typing import Number, Point, SpatialDim
from ....utils.args import alias_kwargs, arg_to_list
from ....utils.conversion import to_tensor, to_tuple
from ...identity import Identity
from .affine import Affine, RandomAffine
        
class Flip(Affine):
    @alias_kwargs([
        ('f', 'flips'),
        ('fc', 'flip_centre'),
    ])
    def __init__(
        self,
        flips: bool | Tuple[bool] | np.ndarray | torch.Tensor,
        flip_centre: Point | Literal['image-centre'] = 'image-centre',
        dim: SpatialDim = 3,
        **kwargs,
        ) -> None:
        # SpatialDim is defined in superclass, but we need to know "scaling" first for parent class.
        # Let parent handle the extension? We can't do this, it'll be confusing talking about
        # "scaling" instead of "flips" to the user.
        self._dim = dim
        self.__flips = to_tensor(flips, broadcast=dim, dtype=torch.bool)
        assert len(self.__flips) == self._dim, f"Expected 'flips' of length {self._dim} for dim={self._dim}, got {len(self.__flips)}."
        scaling = [-1 if f else 1 for f in self.__flips]
        super().__init__(
            rotation=None,
            scaling=scaling,
            translation=None,
            **kwargs,
        )
        super().set_params(
            self.__class__.__name__,
            backward_matrix=self._backward_scaling_matrix,
            flips=self.__flips,
            matrix=self._scaling_matrix,
        )

    def __str__(self) -> str:
        return super().super_str(
            self.__class__.__name__,
            flips=to_tuple(self.__flips),
        )

# This might not be a random affine, which expects a scaling range.
class RandomFlip(RandomAffine):
    def __init__(
        self,
        p_flip: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 0.5,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        self.__p_flip = p_flip
        super().set_params(
            self.__class__.__name__,
            p_flip=self.__p_flip,
        )

    def freeze(self) -> 'Flip':
        # Expand the args.
        # We do this now because 'set_dim' could be called after RandomFlip.__init__.
        p_flip = arg_to_list(self.__p_flip, (int, float), broadcast=self._dim)
        assert len(p_flip) == self._dim, f"Expected 'p_flip' of length {self._dim} for dim={self._dim}, got {len(p_flip)}."
        p_flip = to_tensor(p_flip)

        # Draw the flip parameters.
        should_apply = self._rng.random() < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random(self._dim))
        flip_draw = draw < p_flip
        t_frozen = Flip(flips=flip_draw)
        params = dict(
            flips=flip_draw,
        )
        return super().freeze(Flip, params)

    def __str__(self) -> str:
        return super().__str__(
            self.__class__.__name__,
            p_flip=self.__p_flip,
        )
