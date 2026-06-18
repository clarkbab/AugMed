from jaxtyping import Float32
import numpy as np
import torch
from typing import Any, Dict, List, Literal, Tuple

from ..defaults import get_dim, get_dist, get_dist_std, get_fill, get_filter_offgrid, get_interpolation, get_p, get_seed, get_verbose
from ..typing import AffineMatrix, BatchChannelImage, BatchImage, BatchLabelImage, ChannelImage, Dist, Image, ImagesInput, ImageOutputs, LabelImage, Number, Points, PointsInput, PointsOutputs, SamplingGrid, Size, SpatialAxis, SpatialDim, TransformParams
from ..utils.args import alias_kwargs, arg_default, arg_to_list
from ..utils.assertions import assert_dim, assert_dist, assert_dist_std, assert_fill, assert_filter_offgrid, assert_interpolation, assert_p, assert_seed
from ..utils.conversion import to_return_format, to_tensor
from ..utils.geometry import spatial_size
from ..utils.logging import logger
from ..utils.python import get_private_attr, set_private_attr, wrap_quotes

# Superclass of all (random and deterministic) transforms.
class Transform:
    def __init__(
        self,
        debug: bool = False,
        device: torch.device | Literal['cpu', 'cuda'] | None = None,
        # These params fall back to config defaults (env var or conf.set_<param>) when not passed explicitly.
        # They can also be overridden per-call during transform_images/points.
        dim: SpatialDim | None = None,
        fill: Number | Literal['border', 'max', 'min', 'reflection', 'zeros'] | None = None,
        filter_offgrid: bool | SpatialAxis | List[SpatialAxis] | None = None,
        interpolation: Literal['bicubic', 'bilinear', 'nearest'] | None = None,
        verbose: bool | None = None,
        ) -> None:
        dim = arg_default(dim, dim, get_dim())
        assert_dim(dim, caller='Transform.__init__')
        fill = arg_default(fill, fill, get_fill())
        assert_fill(fill, caller='Transform.__init__')
        filter_offgrid = arg_default(filter_offgrid, filter_offgrid, get_filter_offgrid())
        assert_filter_offgrid(filter_offgrid, caller='Transform.__init__')
        interpolation = arg_default(interpolation, interpolation, get_interpolation())
        assert_interpolation(interpolation, caller='Transform.__init__')
        verbose = arg_default(verbose, verbose, get_verbose())
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

    # Given input images and/or points, we can infer the transform dimensionality.
    # This saves the user from having to change it manually for 2D data.
    # Inference is not always possible. For example, it points=None and image data
    # has batch and/or channel dim.
    #
    # The problem now is that we infer the dim within the frozen transform, however,
    # ranges are expanded in the random transform and frozen values are drawn of length
    # based on the random transform dim. For example, RandomCrop(dim=3, centre=0)
    # will freeze to Crop(dim=3, centre=(0, 0, 0)) which won't make any sense when
    # Crop.transform_images' infer_dim method sets the dim=2.
    # Can we fix this by inferring dim at the random transform level?
    # 1. This should be fine if we're calling .freeze implicitly within transform/
    # transform_images/points, as we can look at the inputs to infer the random transform
    # dim before freezing.
    # 2. When manually calling freeze however, the dim=3 of the random transform will
    # we used to set the frozen transform parameters. There's not much we can do here,
    # and the user should just ensure that they have dim set correctly if they want
    # to call .freeze manually.  
    def infer_dim(
        self,
        images: ImagesInput | None = None,
        points: PointsInput | None = None,
        ) -> SpatialDim | None:
        # Handle empty lists.
        if isinstance(images, list) and len(images) == 0:
            images = None
        if isinstance(points, list) and len(points) == 0:
            points = None

        # Points is our strongest source of dim.
        if points is not None:
            points = arg_to_list(points, (np.ndarray, torch.Tensor))[0]
            dim = points.shape[-1]
            if dim != get_private_attr(self, '__dim'):
                logger.warn(f"Inferred dim={dim} for transform {self.name} due to points shape: {points.shape}")
                self.set_dim(dim)
        elif images is not None:
            # Find the image with the least number of dimensions.
            images = arg_to_list(images, (np.ndarray, torch.Tensor))
            image = list(sorted(images, key=lambda i: i.ndim))[0]
            if image.ndim == 2:
                dim = 2
                if dim != get_private_attr(self, '__dim'):
                    logger.warn(f"Inferred dim={dim} for transform {self.name} due to image shape: {image.shape}")
                    self.set_dim(dim)

    @property
    def name(self) -> str:
        return self.__class__.__name__

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

    # Can be called by Pipeline to set sub-transforms debug mode.
    def __repr__(self) -> str:
        return str(self)

    # Can be called by Pipeline to set sub-transforms devices.
    def set_debug(
        self,
        debug: bool,
        ) -> None:
        set_private_attr(self, '__debug', debug)

    # Can be called by Pipeline to set sub-transforms dims.
    def set_device(
        self,
        device: torch.device | Literal['cpu', 'cuda'] | None,
        ) -> None:
        set_private_attr(self, '__device', torch.device(device) if isinstance(device, str) else device)

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

    # Splits data into images/points.
    def split_data(
        self,
        # TODO: Fix the *data types, it's not really a list of lists. Maybe ImageAndPointsInput?
        # Also: ImageAndPointsOutputs for the return.
        *data: ImagesInput | PointsInput | List[ImagesInput | PointsInput],
        ) -> Tuple[List[Image] | None, List[Points] | None, List[Tuple[Literal['image', 'points'], int]]]:
        data = arg_to_list(data, (np.ndarray, torch.Tensor))

        # Split points and images.
        # This is just based on the size of the final dimension. Size=2 or 3 => points.
        images = []
        points = []
        combine_map = []
        images_i, points_i = 0, 0
        for d in data:
            # Don't check specific 'self.__dim' here as we might actually want to infer
            # the 'dim' from the passed points during transform. This is just easier for
            # users than changing to dim=2 manually (through transform or config).
            if d.shape[-1] in (2, 3):
                points.append(d)
                combine_map.append(('points', points_i))
                points_i += 1
            else:
                images.append(d)
                combine_map.append(('image', images_i))
                images_i += 1

        if len(images) == 0:
            images = None
        if len(points) == 0:
            points = None

        return images, points, combine_map

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
        # TODO: Fix the *data types, it's not really a list of lists. Maybe ImageAndPointsInput?
        # Also: ImageAndPointsOutputs for the return.
        *data: ImagesInput | PointsInput | List[ImagesInput | PointsInput],
        affine: AffineMatrix | None = None,
        fill: Number | Literal['border', 'max', 'min', 'reflection', 'zeros'] | None = None,
        filter_offgrid: bool | SpatialAxis | List[SpatialAxis] | None = None,
        interpolation: Literal['bicubic', 'bilinear', 'nearest'] | None = None,
        return_grid: bool = False,
        return_filtered: bool = False,
        size: Size | None = None,
        ) -> Image | LabelImage | BatchImage | BatchLabelImage | ChannelImage | BatchChannelImage | Points | List[Image | LabelImage | Points | AffineMatrix | SamplingGrid | TransformParams]:
        data = arg_to_list(data, (np.ndarray, torch.Tensor))
        return_types = [type(d) for d in data]
        filter_offgrid = filter_offgrid if filter_offgrid is not None else get_private_attr(self, '__filter_offgrid')

        # Infer the transform dimensionality.
        # This is useful for 2D data as it saves the user from having to 
        # set the dimensionality manually.
        images, points, combine_map = self.split_data(*data)
        self.infer_dim(images=images, points=points)

        # Infer 'size' if it wasn't passed explicitly.
        # Why do we need image size?
        # 1. Points should be filtered if they end up off-grid.
        # 2. Some transforms need the grid size to determine "image-centre", e.g. rotation/scaling.
        # 3. Grid transforms require the size for the input SamplingGrid.
        if size is None:
            if images is None:
                # TODO: Perhaps we should check the transform to see if it needs size.
                # For example, a Pipeline that only contains intensity transforms doesn't need size.
                raise ValueError("Size must be provided when filtering off-grid points without images.")
            size = spatial_size(images[0], get_private_attr(self, '__dim'))

        # Transform images.
        if images is not None:
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
        if points is not None:
            results = self.transform_points(points, affine=affine, filter_offgrid=filter_offgrid, return_filtered=return_filtered, size=size)
            if isinstance(results, (np.ndarray, torch.Tensor)):
                results = [results]
            if filter_offgrid and return_filtered:
                *points_ts, indices = results
            else:
                points_ts = results
        else:
            points_ts = []

        # Combine image and points results.
        # We need a map from data_ts index -> (images/points) -> index.
        data_ts = []
        for t, i in combine_map:
            if t == 'image':
                data_ts.append(image_ts[i])
            elif t == 'points':
                data_ts.append(points_ts[i])

        # Convert to return format.
        other_data = []
        if return_grid:
            other_data.append(grid_t)
        if points and filter_offgrid and return_filtered:
            # Indices could be a tensor or list of tensors for multiple points arrays.
            points_return_types = [return_types[i] for i in range(len(data)) if combine_map[i][0] == 'points']
            indices = to_return_format(indices, return_types=points_return_types)
            other_data.append(indices)
        return to_return_format(data_ts, other_data=other_data, return_types=return_types)

    def transform_images(
        self,
        *args,
        **kwargs,
        ) -> ImageOutputs:
        raise ValueError("Subclasses of 'Transform' must implement 'transform_images' method.")

    def transform_points(
        self,
        *args,
        **kwargs,
        ) -> PointsOutputs:
        raise ValueError("Subclasses of 'Transform' must implement 'transform_points' method.")

class RandomTransform(Transform):
    # Might not be a good idea to use 'alias_kwargs' here
    # as it might clash with transform-specific kwarg aliases?
    def __init__(
        self,
        dist: Literal['gaussian', 'uniform'] | None = None,
        dist_std: float | None = None,
        p: Number | None = None,    # What proportion of the time is the transform applied? Un-applied transforms resolve to 'Identity' when frozen.
        seed: int | None = None,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        dist = arg_default(dist, dist, get_dist())
        assert_dist(dist, caller='RandomTransform.__init__')
        self.set_dist(dist)
        dist_std = arg_default(dist_std, dist_std, get_dist_std())
        assert_dist_std(dist_std, caller='RandomTransform.__init__')
        set_private_attr(self, '__dist_std', dist_std)
        p = arg_default(p, p, get_p())
        assert_p(p, caller='RandomTransform.__init__')
        set_private_attr(self, '__p', p)
        seed = arg_default(seed, seed, get_seed())
        assert_seed(seed, caller='RandomTransform.__init__')
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
        params['fill'] = get_private_attr(self, '__fill')
        params['filter_offgrid'] = get_private_attr(self, '__filter_offgrid')
        params['interpolation'] = get_private_attr(self, '__interpolation')
        params['verbose'] = get_private_attr(self, '__verbose')
        return klass(**params)

    def params(
        self,
        **kwargs,
        ) -> TransformParams:
        return super().params(p=get_private_attr(self, '__p'), **kwargs)

    # Can be called by Pipeline to set sub-transform's random dists.
    def set_dist(
        self,
        dist: Dist, 
        ) -> None:
        assert_dist(dist, caller='set_dist')
        set_private_attr(self, '__dist', dist)

    # Can be called by Pipeline to set sub-transform's random seeds.
    def set_seed(
        self,
        seed: int | None = None,
        ) -> None:
        set_private_attr(self, '__seed', seed)
        set_private_attr(self, '__rng', np.random.default_rng(seed=seed))

    def draw_from_range(
        self,
        range: Float32[torch.Tensor, "2"] | Float32[torch.Tensor, "2 D"] | Float32[torch.Tensor, "2 2 D"],
        dist: Dist | None = None,
        dist_std: float | None = None,
        ) -> Float32[torch.Tensor, "1"] | Float32[torch.Tensor, "D"] | Float32[torch.Tensor, "2 D"]:
        dist = dist if dist is not None else get_private_attr(self, '__dist')
        dist_std = dist_std if dist_std is not None else get_private_attr(self, '__dist_std')
        if dist == 'gaussian':
            mean = (range[0] + range[1]) / 2
            std = (range[1] - range[0]) / (2 * dist_std)
            draw = to_tensor(get_private_attr(self, '__rng').normal(loc=mean, scale=std))
        elif dist == 'uniform':
            n = 1 if range.ndim == 1 else tuple(range.shape[1:])
            draw = to_tensor(get_private_attr(self, '__rng').random(n)) * (range[1] - range[0]) + range[0]
        return draw

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
        *data: ImagesInput | PointsInput,
        dist: Dist | None = None,
        dist_std: float | None = None,
        return_params: bool = False,
        **kwargs,
        ) -> Image | LabelImage | Points | List[Image | LabelImage | Points | List[SamplingGrid] | TransformParams]:
        # Infer random transform dim before freezing. Otherwise, we get an issue when random dim=3,
        # but frozen transform dim=2 is inferred during transform_images. See notes above "def infer_dim".
        images, points, _ = self.split_data(*data)
        self.infer_dim(images=images, points=points)

        # Delegate to frozen transform.
        t_frozen = self.freeze(dist=dist, dist_std=dist_std)
        results = t_frozen.transform(*data, **kwargs)

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
        dist: Dist | None = None,
        dist_std: float | None = None,
        return_params: bool = False,
        **kwargs,
        ) -> ImageOutputs:
        # Delegate to frozen transform.
        t_frozen = self.freeze(dist=dist, dist_std=dist_std)
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
        dist: Dist | None = None,
        dist_std: float | None = None,
        return_params: bool = False,
        **kwargs,
        ) -> PointsOutputs:
        # Delegate to frozen transform.
        t_frozen = self.freeze(dist=dist, dist_std=dist_std)
        results = t_frozen.transform_points(*args, **kwargs)

        # Convert to return format.
        other_data = []
        if return_params:
            other_data.append(t_frozen.params)
        return to_return_format(results, other_data=other_data)
    