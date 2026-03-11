from typing import *

from ....typing import *
from ....utils.args import alias_kwargs, arg_to_list
from ....utils.conversion import to_tensor, to_tuple
from ....utils.geometry import fov_centre
from ....utils.matrix import create_scaling
from ...identity import Identity
from ..spatial import RandomSpatialTransform
from .affine import Affine
        
class Flip(Affine):
    @alias_kwargs([
        ('f', 'flips'),
        ('fc', 'flip_centre'),
    ])
    def __init__(
        self,
        flips: bool | Tuple[bool] | np.ndarray | torch.Tensor,
        flip_centre: Point | Literal['image-centre'] = 'image-centre',
        dim: Dim = 3,
        **kwargs) -> None:
        # Dim is defined in superclass, but we need to know "scaling" first for parent class.
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
        self._params = dict(
            type=self.__class__.__name__,
            backward_matrix=self._backward_scaling_matrix,
            dim=self._dim,
            flips=self.__flips,
            matrix=self._scaling_matrix,
        )

    def __str__(self) -> str:
        params = dict(
            flips=to_tuple(self.__flips),
        )
        return super().super_str(self.__class__.__name__, params)

# This might not be a random affine, which expects a scaling range.
class RandomFlip(RandomSpatialTransform):
    def __init__(
        self,
        p_flip: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 0.5,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        p_flip = arg_to_list(p_flip, (int, float), broadcast=self._dim)
        assert len(p_flip) == self._dim, f"Expected 'p_flip' of length {self._dim} for dim={self._dim}, got {len(p_flip)}."
        self.__p_flip = to_tensor(p_flip)
        self._params = dict(
            type=self.__class__.__name__,
            dim=self._dim,
            p=self._p,
            p_flip=self.__p_flip,
        )

    def freeze(self) -> 'Flip':
        should_apply = self._rng.random() < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random(self._dim))
        flip_draw = draw < self.__p_flip
        t_frozen = Flip(flips=flip_draw)
        params = dict(
            flips=flip_draw,
        )
        return super().freeze(Flip, params)

    def __str__(self) -> str:
        params = dict(
            p_flip=to_tuple(self.__p_flip, decimals=3),
        )
        return super().__str__(self.__class__.__name__, params)
