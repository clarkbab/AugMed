import numpy as np
import torch
from typing import Any, Dict, List, Literal

from ..config import get_dim
from ..typing import AffineMatrix, BatchChannelImage, BatchImage, BatchLabelImage, ChannelImage, Image, Indices, LabelImage, Number, Points, SamplingGrid, Size, SpatialDim, TransformParams
from ..utils.args import alias_kwargs, arg_default, arg_to_list
from ..utils.assertions import assert_dim
from ..utils.conversion import to_return_format
from ..utils.geometry import spatial_size
from ..utils.python import get_private_attr, set_private_attr, wrap_quotes

# Superclass of all (random and deterministic) transforms.
class Transform:
    def __init__(
        self,
        debug: bool = False,
        device: torch.device | Literal['cpu', 'cuda'] | None = None,
        dim: SpatialDim | None = None,
        fill: Number | Literal['border', 'max', 'min', 'reflection', 'zeros'] = 'min',
        filter_offgrid: bool | SpatialDim | List[SpatialDim] = True,
        interpolation: Literal['bicubic', 'bilinear', 'nearest'] = 'bilinear',
        verbose: bool = False,
        ) -> None:
        dim = arg_default(dim, dim, get_dim())
        assert_dim(dim)
        set_private_attr(self, '__debug', debug)
        set_private_attr(self, '__device', torch.device(device) if isinstance(device, str) else device)
        set_private_attr(self, '__dim', dim)
        set_private_attr(self, '__fill', fill)
        set_private_attr(self, '__filter_offgrid', filter_offgrid)
        set_private_attr(self, '__interpolation', interpolation)
        set_private_attr(self, '__verbose', verbose)
    
    def __call__(
        self,
        *args,
        **kwargs,
        ) -> Image | Points | List[Image | Points | List[SamplingGrid] | TransformParams]:
        return self.transform(*args, **kwargs)

    @property
    def dim(self) -> SpatialDim:
        return get_private_attr(self, '__dim')

    def params(
        self,
        **kwargs,
        ) -> TransformParams:
        #linter:nosort
        return dict(
            type=self.__class__.__name__,
            device=get_private_attr(self, '__device'),
            dim=get_private_attr(self, '__dim'),
            fill=get_private_attr(self, '__fill'),
            filter_offgrid=get_private_attr(self, '__filter_offgrid'),
            interpolation=get_private_attr(self, '__interpolation'),
            **kwargs,
        )

    def __repr__(self) -> str:
        return str(self)

    # Can be called by Pipeline to set sub-transforms debug mode.
    def set_debug(
        self,
        debug: bool,
        ) -> None:
        set_private_attr(self, '__debug', debug)

    # Can be called by Pipeline to set sub-transforms devices.
    def set_device(
        self,
        device: torch.device | Literal['cpu', 'cuda'] | None,
        ) -> None:
        set_private_attr(self, '__device', torch.device(device) if isinstance(device, str) else device)

    # Can be called by Pipeline to set sub-transforms dims.
    def set_dim(
        self,
        dim: SpatialDim,
        ) -> None:
        assert_dim(dim)
        set_private_attr(self, '__dim', dim)

    def __str__(
        self,
        class_name: str,
        skip_format: str | List[str] | None = None,
        subtransform: bool = False,
        **params: Any,
        ) -> str:
        # Pipeline subtransforms shouldn't show these values as they are ignored,
        # only the Pipeline-level values are relevant.
        if not subtransform:
            params['device'] = wrap_quotes(str(get_private_attr(self, '__device'))) if isinstance(get_private_attr(self, '__device'), torch.device) else get_private_attr(self, '__device')
            params['dim'] = get_private_attr(self, '__dim')
            params['fill'] = wrap_quotes(get_private_attr(self, '__fill')) if isinstance(get_private_attr(self, '__fill'), str) else get_private_attr(self, '__fill')
            params['filter_offgrid'] = get_private_attr(self, '__filter_offgrid')
            params['interpolation'] = wrap_quotes(get_private_attr(self, '__interpolation'))
        skip_format = arg_to_list(skip_format, str)
        def format(k: str, v: Any) -> str:
            if skip_format is not None and k in skip_format:
                return f"{k}={v}"
            return f"{k}={v}"
        return f"{class_name}({', '.join([format(k, v) for k, v in params.items()])})"

    # Originally this was defined as a mixin to avoid having RandomTransforms override the method.
    # However, as a mixin, each new transform class needs to subclass the mixin also, which creates
    # more boilerplate for new transforms.
    @alias_kwargs(
        ('a', 'affine'),
        ('fo', 'filter_offgrid'),
        ('rg', 'return_grid'),
        ('rp', 'return_params'),
        ('s', 'size'),
    )
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
        *data: Image | LabelImage | BatchImage | BatchLabelImage | ChannelImage | BatchChannelImage | Points | List[Image | LabelImage | Points],
        affine: AffineMatrix | None = None,
        fill: Number | Literal['border', 'max', 'min', 'reflection', 'zeros'] | None = None,
        filter_offgrid: bool | SpatialDim | List[SpatialDim] | None = None,
        interpolation: Literal['bicubic', 'bilinear', 'nearest'] | None = None,
        return_grid: bool = False,
        return_filtered: bool = False,
        size: Size | None = None,
        ) -> Image | LabelImage | BatchImage | BatchLabelImage | ChannelImage | BatchChannelImage | Points | List[Image | LabelImage | Points | AffineMatrix | SamplingGrid | TransformParams]:
        data = arg_to_list(data, (np.ndarray, torch.Tensor))
        return_types = [type(d) for d in data]
        filter_offgrid = filter_offgrid if filter_offgrid is not None else get_private_attr(self, '__filter_offgrid')

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
            size = spatial_size(data[image_indices[0]], get_private_attr(self, '__dim'))

        # Transform images.
        images = [data[i] for i in image_indices]
        if len(images) > 0:
            results = self.transform_images(images, affine=affine, fill=fill, interpolation=interpolation, return_grid=return_grid)
            if isinstance(results, (np.ndarray, torch.Tensor)):
                results = [results]
            if return_grid:
                *image_ts, grid_t = results
            else:
                image_ts = results
        else:
            image_ts = []

        # Transform points.
        points = [data[i] for i in points_indices]
        if len(points) > 0:
            results = self.transform_points(points, affine=affine, filter_offgrid=filter_offgrid, return_filtered=return_filtered, size=size)
            if isinstance(results, (np.ndarray, torch.Tensor)):
                results = [results]
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
        if return_grid:
            other_data.append(grid_t)
        if filter_offgrid and return_filtered and len(points) > 0:
            # Indices could be a tensor or list of tensors for multiple points arrays.
            points_return_types = [return_types[i] for i in points_indices]
            indices = to_return_format(indices, return_types=points_return_types)
            other_data.append(indices)
        return to_return_format(data_ts, other_data=other_data, return_types=return_types)

    def transform_images(
        self,
        *args,
        **kwargs,
        ) -> Image | LabelImage | BatchImage | BatchLabelImage | ChannelImage | BatchChannelImage | List[Image | LabelImage | BatchImage | BatchLabelImage | ChannelImage | BatchChannelImage | AffineMatrix | SamplingGrid]:
        raise ValueError("Subclasses of 'Transform' must implement 'transform_images' method.")

    def transform_points(
        self,
        *args,
        **kwargs,
        ) -> Points | List[Points | Indices]:
        raise ValueError("Subclasses of 'Transform' must implement 'transform_points' method.")

class RandomTransform(Transform):
    def __init__(
        self,
        p: Number = 1.0,    # What proportion of the time is the transform applied? Un-applied transforms resolve to 'Identity' when frozen.
        seed: int | None = None,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        set_private_attr(self, '__p', p)
        self.set_seed(seed)

    def freeze(
        self,
        klass: 'Object',
        params: Dict[str, Any],
        ) -> Transform:
        # Copy general params from random -> frozen transform. I always forget these.
        params['debug'] = get_private_attr(self, '__debug')
        params['device'] = get_private_attr(self, '__device')
        params['dim'] = get_private_attr(self, '__dim')
        return klass(**params)

    def params(
        self,
        **kwargs,
        ) -> TransformParams:
        return super().params(p=get_private_attr(self, '__p'), **kwargs)

    # Can be called by Pipeline to set sub-transforms random seeds.
    def set_seed(
        self,
        seed: int | None = None,
        ) -> None:
        set_private_attr(self, '__seed', seed)
        set_private_attr(self, '__rng', np.random.default_rng(seed=seed))

    def __str__(
        self,
        class_name: str,
        subtransform: bool = False,
        **params: Any,
        ) -> str:
        return super().__str__(
            class_name,
            **params,
            p=get_private_attr(self, '__p'),
            seed=get_private_attr(self, '__seed'),
            subtransform=subtransform,
        )

    def transform(
        self,
        *args,
        return_params: bool = False,
        **kwargs,
        ) -> Image | LabelImage | Points | List[Image | LabelImage | Points | List[SamplingGrid] | TransformParams]:
        # Delegate to frozen transform.
        t_frozen = self.freeze()
        results = t_frozen.transform(*args, **kwargs)

        # Convert to return format.
        other_data = []
        if return_params:
            other_data.append(t_frozen.params)
        return to_return_format(results, other_data=other_data)

    @alias_kwargs(
        ('rp', 'return_params'),
    )
    def transform_images(
        self,
        *args,
        return_params: bool = False,
        **kwargs,
        ) -> Image | LabelImage | BatchImage | BatchLabelImage | ChannelImage | BatchChannelImage | List[Image | LabelImage | BatchImage | BatchLabelImage | ChannelImage | BatchChannelImage | SamplingGrid | TransformParams]:
        # Delegate to frozen transform.
        t_frozen = self.freeze()
        results = t_frozen.transform_images(*args, **kwargs)

        # Add optional "params".
        other_data = []
        if return_params:
            other_data.append(t_frozen.params)
        return to_return_format(results, other_data=other_data)

    @alias_kwargs(
        ('rp', 'return_params'),
    )
    def transform_points(
        self,
        *args,
        return_params: bool = False,
        **kwargs,
        ) -> Points | List[Points | Indices | TransformParams]:
        # Delegate to frozen transform.
        t_frozen = self.freeze()
        results = t_frozen.transform_points(*args, **kwargs)

        # Convert to return format.
        other_data = []
        if return_params:
            other_data.append(t_frozen.params)
        return to_return_format(results, other_data=other_data)
    