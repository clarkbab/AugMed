from typing import *

from ..typing import *
from ..utils.args import alias_kwargs, arg_to_list

# What is a Transform?
# Transform defines the API that any (deterministic) Transform
# and RandomTransform must follow.
# What about pipeline? Yeah, I guess so. We treat it just like a transform.
class Transform:
    def __init__(
        self,
        device: torch.device | Literal['cpu', 'cuda'] | None = None,
        dim: Dim = 3,
        verbose: bool = False,
        ) -> None:
        assert dim in [2, 3], "Only 2D and 3D flips are supported."
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
    def dim(self) -> Dim:
        return self._dim

    @property
    def params(self) -> TransformParams:
        if not hasattr(self, '_params'):
            raise ValueError("Subclasses of 'Transform' must have '_params' attribute.")
        return self._params

    def __repr__(self) -> str:
        return str(self)

    # Can be called by Pipeline to set sub-transforms devices.
    def set_device(
        self,
        device: torch.device | Literal['cpu', 'cuda'] | None,
        ) -> None:
        self._device = torch.device(device) if isinstance(device, str) else device

    # Can be called by Pipeline to set sub-transforms dims.
    def set_dim(
        self,
        dim: Dim,
        ) -> None:
        assert dim in [2, 3], "Only 2D and 3D transforms are supported."
        self._dim = dim

    def __str__(
        self,
        class_name: str,
        params: Dict[str, Any],
        ) -> str:
        params['device'] = self._device
        params['dim'] = self._dim
        return f"{class_name}({', '.join([f'{k}={v}' for k, v in params.items()])})"

    # Originally this was defined as a mixin to avoid having RandomTransforms override the method.
    # However, as a mixin, each new transform class needs to subclass the mixin also, which creates
    # more boilerplate for new transforms.
    @alias_kwargs([
        ('a', 'affine'),
    ])
    # Can pass a single array/tensor or a list of arrays/tensors.
    # Points arrays/tensors are inferred by their Nx2/3 shape. It's unlikely that images of this size will
    # be passed, but it would break.
    # Labels are inferred by the data type of the passed array/tensor (bool) and will be returned
    # in boolean type.
    # Will return a single transformed array/tensor or list of arrays/tensors.
    # If a single affine/size is passed, this is broadcast to all image arrays/tensors,
    # other
    def transform(
        self,
        *data: Image | Points | List[Image | Points],
        affine: Affine | List[Affine] | None = None,
        size: Size | List[Size] | np.ndarray | torch.Tensor | None = None,
        filter_offgrid: bool = True,
        return_grid: bool = False,
        return_params: bool = False,
        ) -> Image | Points | List[Image | Points | List[SamplingGrid] | TransformParams]:
        datas, data_was_single = arg_to_list(data, (np.ndarray, torch.Tensor), return_expanded=True)
        affines = arg_to_list(affine, (np.ndarray, torch.Tensor, None), broadcast=len(datas))
        sizes = arg_to_list(size, (tuple, np.ndarray, torch.Tensor, None), broadcast=len(datas), iter_types=(np.ndarray, torch.Tensor))
        assert len(affines) == len(datas), "Number of affines must match number of data arrays/tensors."
        assert len(sizes) == len(datas), "Number of sizes must match number of data arrays/tensors."

        # Infer data types.
        image_indices = []
        points_indices = []
        data_types = []
        for i, d in enumerate(datas):
            if d.shape[-1] == 2 or d.shape[-1] == 3:
                points_indices.append(i)
                data_types.append('points')
            else:
                image_indices.append(i)
                data_types.append('image')

        # Infer sizes for offscreen point filtering.
        if filter_offgrid:
            for i in points_indices:
                if sizes[i] is None:
                    # Infer size from images - must all have same shape.
                    image_sizes = [datas[j].shape[-self._dim:] for j in image_indices]
                    if len(image_sizes) > 0 and np.unique(image_sizes, axis=0).shape[0] == 1:
                        sizes[i] = tuple(image_sizes[0])

        # Transform images.
        # Wait, if this is a random transform, then 'transform_image'
        # and 'transform_points' will apply different transforms.
        images = [datas[i] for i in image_indices]
        image_data_was_single = len(images) == 1
        image_affines = [affines[i] for i in image_indices]
        if len(images) > 0:
            res_ts = self.transform_image(images, affine=image_affines, return_grid=return_grid)
            if return_grid:
                *image_ts, grid_ts = res_ts
            else:
                image_ts = res_ts
        else:
            image_ts = []

        # Transform points.
        points = [datas[i] for i in points_indices]
        points_sizes = [sizes[i] for i in points_indices]
        points_affines = [affines[i] for i in points_indices]
        points_ts = []
        for p, a, sz in zip(points, points_affines, points_sizes):
            if sz is None:
                filter_offgrid = False    # Only filter if 'size' was passed or inferred from image sizes.
            points_t = self.transform_points(p, filter_offgrid=filter_offgrid, size=sz, affine=a)
            points_ts.append(points_t)

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
        results = data_ts[0] if data_was_single else data_ts
        if return_grid:
            if isinstance(results, list):
                results.append(grid_ts)
            else:
                results = [results, grid_ts]
        if return_params:
            params_result = self._params
            if isinstance(results, list):
                results.append(params_result)
            else:
                results = [results, params_result]

        return results

    def transform_image(
        self,
        *args,
        **kwargs,
        ) -> Image | List[Image | List[SamplingGrid]]:
        raise ValueError("Subclasses of 'Transform' must implement 'transform_image' method.")

    def transform_points(
        self,
        *args,
        **kwargs,
        ) -> Points:
        raise ValueError("Subclasses of 'Transform' must implement 'transform_points' method.")

# RandomTransforms should have all the behaviour of a normal transform.
class RandomTransform(Transform):
    def __init__(
        self,
        p: Number = 1.0,    # What proportion of the time is the transform applied? Un-applied transforms resolve to 'Identity' when frozen.
        seed: Optional[int] = None,
        **kwargs) -> None:
        super().__init__(**kwargs)
        self._p = p
        self.set_seed(seed)

    def freeze(
        self,
        klass: 'Object',
        params: Dict[str, Any],
        ) -> None:
        # Copy general params from random -> frozen transform. I always forget these.
        params['dim'] = self._dim
        return klass(**params)

    def set_seed(
        self,
        seed: Optional[int]) -> None:
        self._rng = np.random.default_rng(seed=seed)

    def __str__(
        self,
        class_name: str,
        params: Dict[str, Any],
        ) -> str:
        params['p'] = self._p
        return super().__str__(class_name, params)

    def transform(
        self,
        *args,
        return_params: bool = False,
        **kwargs,
        ) -> Image | Points | List[Image | Points | List[SamplingGrid] | TransformParams]:
        t_frozen = self.freeze()
        results = t_frozen.transform(*args, **kwargs)

        # Convert to return format.
        if return_params:
            if isinstance(results, list):
                results.append(t_frozen.params)
            else:
                results = [results, t_frozen.params]
        
        return results

    def transform_image(
        self,
        *args,
        return_params: bool = False,
        **kwargs,
        ) -> Image | List[Image | List[SamplingGrid] | TransformParams]:
        t_frozen = self.freeze()
        results = t_frozen.transform_image(*args, **kwargs)

        # Convert to return format.
        if return_params:
            if isinstance(results, list):
                results.append(t_frozen.params)
            else:
                results = [results, t_frozen.params]
        
        return results

    def transform_points(
        self,
        *args,
        return_params: bool = False,
        **kwargs,
        ) -> Points | List[Points | TransformParams]:
        t_frozen = self.freeze()
        results = t_frozen.transform_points(*args, **kwargs)

        # Convert to return format.
        if return_params:
            if isinstance(results, list):
                results.append(t_frozen.params)
            else:
                results = [results, t_frozen.params]
        
        return results
    