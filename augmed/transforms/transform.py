import numpy as np
import torch
from typing import Any, Dict, List, Literal

from ..typing import AffineMatrix, Image, Indices, Number, Points, SamplingGrid, Size, SpatialDim, TransformParams
from ..utils.args import alias_kwargs, arg_to_list
from ..utils.conversion import to_return_format

# What is a Transform?
# Transform defines the API that any (deterministic) Transform
# and RandomTransform must follow.
# What about pipeline? Yeah, I guess so. We treat it just like a transform.
class Transform:
    def __init__(
        self,
        debug: bool = False,
        device: torch.device | Literal['cpu', 'cuda'] | None = None,
        dim: SpatialDim = 3,
        verbose: bool = False,
        ) -> None:
        assert dim in [2, 3], "Only 2D and 3D flips are supported."
        self._debug = debug
        self._device = torch.device(device) if isinstance(device, str) else device
        self._dim = dim
        self._verbose = verbose
    
    def __call__(
        self,
        *args,
        **kwargs,
        ) -> Image | Points | List[Image | Points | List[SamplingGrid] | TransformParams]:
        return self.transform(*args, **kwargs)

    @property
    def dim(self) -> SpatialDim:
        return self._dim

    @property
    def params(self) -> TransformParams:
        if not hasattr(self, '_params'):
            raise ValueError("Subclasses of 'Transform' must have '_params' attribute.")
        return self._params

    def __repr__(self) -> str:
        return str(self)

    # Can be called by Pipeline to set sub-transforms debug mode.
    def set_debug(
        self,
        debug: bool,
        ) -> None:
        self._debug = debug

    # Can be called by Pipeline to set sub-transforms devices.
    def set_device(
        self,
        device: torch.device | Literal['cpu', 'cuda'] | None,
        ) -> None:
        self._device = torch.device(device) if isinstance(device, str) else device

    # Can be called by Pipeline to set sub-transforms dims.
    def set_dim(
        self,
        dim: SpatialDim,
        ) -> None:
        assert dim in [2, 3], "Only 2D and 3D transforms are supported."
        self._dim = dim

    # Can be called by all child classes.
    def set_params(
        self,
        type: str,
        **params: TransformParams,
        ) -> None:
        # Put transform type first.
        self._params = dict(
            device=self._device,
            **params,
            dim=self._dim,
            type=type,
        )

    def __str__(
        self,
        class_name: str,
        **params: Any,
        ) -> str:
        params['device'] = self._device
        params['dim'] = self._dim
        return f"{class_name}({', '.join([f'{k}={v}' for k, v in params.items()])})"

    # Originally this was defined as a mixin to avoid having RandomTransforms override the method.
    # However, as a mixin, each new transform class needs to subclass the mixin also, which creates
    # more boilerplate for new transforms.
    @alias_kwargs([
        ('a', 'affine'),
        ('fo', 'filter_offgrid'),
        ('ra', 'return_affine'),
        ('rp', 'return_params'),
        ('s', 'size'),
    ])
    # Can pass a single array/tensor or a list of arrays/tensors.
    # Points arrays/tensors are inferred by their Nx2/3 shape. It's unlikely that images of this size will
    # be passed, but it would break.
    # Labels are inferred by the data type of the passed array/tensor (bool) and will be returned
    # in boolean type.
    # Will return a single transformed array/tensor or list of arrays/tensors.
    # All images/points must have a single size/affine - but size is inferred when images are passed. Points
    # require SamplingGrid for filtering off-grid points after transforming.
    def transform(
        self,
        *data: Image | Points | List[Image | Points],
        affine: AffineMatrix | None = None,
        filter_offgrid: bool = True,
        return_affine: bool = False,
        return_filtered: bool = False,
        size: Size | None = None,
        ) -> Image | Points | List[Image | Points | AffineMatrix | TransformParams]:
        data, data_was_single = arg_to_list(data, (np.ndarray, torch.Tensor), return_expanded=True)
        return_types = [type(d) for d in data]

        # Split points and images.
        image_indices = []
        points_indices = []
        data_types = []
        for i, d in enumerate(data):
            if d.shape[-1] == 2 or d.shape[-1] == 3:
                points_indices.append(i)
                data_types.append('points')
            else:
                image_indices.append(i)
                data_types.append('image')

        # Why do we need image size?
        # 1. Points should be filtered if they end up off-grid.
        # 2. Some transforms need the grid size to determine "image-centre", e.g. rotation/scaling.
        # 3. Grid transforms require the size for the input SamplingGrid.
        if size is None:
            if len(image_indices) == 0:
                # TODO: Perhaps we should check the transform to see if it needs size.
                # For example, a Pipeline that only contains intensity transforms doesn't need size.
                raise ValueError("Size must be provided when filtering off-grid points without images.")
            size = data[image_indices[0]].shape[-self._dim:]

        # Transform images.
        images = [data[i] for i in image_indices]
        if len(images) > 0:
            results = self.transform_images(images, affine=affine, return_affine=return_affine, return_single=False)
            if return_affine:
                *image_ts, affine_t = results
            else:
                image_ts = results
        else:
            image_ts = []

        # Transform points.
        pointses = [data[i] for i in points_indices]
        if len(pointses) > 0:
            results = self.transform_points(pointses, affine=affine, filter_offgrid=filter_offgrid, return_filtered=return_filtered, return_single=False, size=size)
            if filter_offgrid and return_filtered:
                *points_ts, indices = results
            else:
                points_ts = results
        else:
            points_ts = []

        # Flatten image and points results.
        data_ts = []
        image_i, points_i = 0, 0
        for i, t in enumerate(data_types):
            if t == 'image':
                data_ts.append(image_ts[image_i])
                image_i += 1
            else:
                data_ts.append(points_ts[points_i])
                points_i += 1

        # Convert to return format.
        other_data = []
        if return_affine:
            other_data.append(affine_t)
        if filter_offgrid and return_filtered and len(pointses) > 0:
            # Indices could be a tensor or list of tensors for multiple points arrays.
            points_return_types = [return_types[i] for i in points_indices]
            indices = to_return_format(indices, return_single=True, return_types=points_return_types)
            other_data.append(indices)
        results = to_return_format(data_ts, other_data=other_data, return_single=data_was_single, return_types=return_types)

        return results

    def transform_images(
        self,
        *args,
        **kwargs,
        ) -> Image | List[Image | List[SamplingGrid]]:
        raise ValueError("Subclasses of 'Transform' must implement 'transform_images' method.")

    def transform_points(
        self,
        *args,
        **kwargs,
        ) -> Points | List[Points | Indices]:
        raise ValueError("Subclasses of 'Transform' must implement 'transform_points' method.")

# RandomTransforms should have all the behaviour of a normal transform.
class RandomTransform(Transform):
    def __init__(
        self,
        p: Number = 1.0,    # What proportion of the time is the transform applied? Un-applied transforms resolve to 'Identity' when frozen.
        seed: int | None = None,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        self._p = p
        self.set_seed(seed)

    def freeze(
        self,
        klass: 'Object',
        params: Dict[str, Any],
        ) -> None:
        # Copy general params from random -> frozen transform. I always forget these.
        params['debug'] = self._debug
        params['device'] = self._device
        params['dim'] = self._dim
        return klass(**params)

    def set_params(
        self,
        *args,
        **params: TransformParams,
        ) -> None:
        super().set_params(*args, p=self._p, **params)

    def set_seed(
        self,
        seed: int | None = None,
        ) -> None:
        self._rng = np.random.default_rng(seed=seed)

    def __str__(
        self,
        class_name: str,
        **params: Any,
        ) -> str:
        return super().__str__(class_name, p=self._p, **params)

    def transform(
        self,
        *args,
        return_params: bool = False,
        **kwargs,
        ) -> Image | Points | List[Image | Points | List[SamplingGrid] | TransformParams]:
        # Delegate to frozen transform.
        t_frozen = self.freeze()
        results = t_frozen.transform(*args, **kwargs)
        results_is_single = isinstance(results, (np.ndarray, torch.Tensor))

        # Convert to return format.
        other_data = []
        if return_params:
            other_data.append(t_frozen.params)
        results = to_return_format(results, other_data=other_data, return_single=results_is_single)
        
        return results

    def transform_images(
        self,
        *args,
        return_params: bool = False,
        **kwargs,
        ) -> Image | List[Image | List[SamplingGrid] | TransformParams]:
        # Delegate to frozen transform.
        t_frozen = self.freeze()
        results = t_frozen.transform_images(*args, **kwargs)
        results_is_single = isinstance(results, (np.ndarray, torch.Tensor))

        # Add optional "params".
        other_data = []
        if return_params:
            other_data.append(t_frozen.params)
        results = to_return_format(results, other_data=other_data, return_single=results_is_single)
        
        return results

    def transform_points(
        self,
        *args,
        return_params: bool = False,
        **kwargs,
        ) -> Points | List[Points | Indices | TransformParams]:
        # Delegate to frozen transform.
        t_frozen = self.freeze()
        results = t_frozen.transform_points(*args, **kwargs)
        results_is_single = isinstance(results, (np.ndarray, torch.Tensor))

        # Convert to return format.
        other_data = []
        if return_params:
            other_data.append(t_frozen.params)
        results = to_return_format(results, other_data=other_data, return_single=results_is_single)
        return results
    