import numpy as np
import numpy as np
import torch
import torch
from typing import List, Union

from ..typing import Affine, AffineTensor, Image, Indices, Points, PointsTensor, SamplingGrid, SamplingGridTensor, Size
from ..utils.args import alias_kwargs, arg_to_list
from ..utils.conversion import to_return_format, to_tensor, to_tuple
from ..utils.geometry import fov
from ..utils.grid import grid_points, grid_sample
from ..utils.logging import logger
from ..utils.misc import get_group_device
from .grid import GridTransform
from .identity import Identity
from .intensity import IntensityTransform
from .spatial import SpatialTransform
from .transform import RandomTransform, Transform

# This shouldn't be instantiated by the user.
# FrozenPipeline is needed so that it can inherit the 'Transform.transform' method,
# which expects to be called only on deterministic transforms (as image and points
# transorms are called separately and should apply the same transforms).
class FrozenPipeline(Transform):
    def __init__(
        self,
        transforms: List[Union[Transform]],
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        self.__transforms = transforms
        self.__warn_resamples()
        self._params = dict(
            transforms=[t.params for t in self.__transforms],
            type=self.__class__.__name__,
        )

    # Performs the back transform for a grid/spatial group applying
    # the affine optimisation if possible.
    def __backward_transform_points_for_group(
        self,
        transforms: List[Transform],     # TODO: SpatialTransforms?
        points: PointsTensor,
        grids: List[SamplingGrid],     # These are the input grids to each transform - required by some, e.g. Rotate.
        **kwargs,
        ) -> PointsTensor:
        points_t = points

        # Create chains of homogeneous matrix multiplications.
        # E.g. for flip and rotate, naively we could perform each separately by 
        # running points_t = matmul(T_2, R, T_1, points.T).T, where T_1 translates centre of rotation to origin,
        # R performs rotation, and T_2 reverses the initial translation, followed by
        # points_t = matmul(T_2, F, T_1, points_t.T).T where F flips along certain axes. Note that these are performed
        # in reverse order because it's the back transform. With this approach, we perform two large matrix
        # multiplications using 3xN points.T matrix.
        # A better approach is to pull out chains of homogeneous matrix multiplications and concatenate them
        # so that the points matrix is only used once (for each chain).
        affine_chain = []
        for i, (t, g) in enumerate(reversed(list(zip(transforms, grids)))):
            # Chain resolution conditions:
            # 1. Non-affine transform.
            # 2. Final transform.
            if isinstance(t, SpatialTransform):
                # Store any affine multiplications for later.
                if isinstance(t, Affine):
                    t_affine = t.get_affine_backward_transform(points_t.device, grid=g)
                    # Transform 't' iterates backwards through transform list.
                    # We want transforms that are later in the list to be applied first (i.e. to be
                    # later in the affine chain). So prepend transforms to the list.
                    affine_chain.insert(0, t_affine)
                else:
                    # Resolve chain.
                    if len(affine_chain) > 0:
                        points_t = self.__resolve_affine_chain(points_t, affine_chain)
                        affine_chain = []

                    # Perform current transform.
                    points_t = t.backward_transform_points(points_t, grid=g)

            # Resolve if final round.
            if i == len(transforms) - 1 and len(affine_chain) > 0:
                points_t = self.__resolve_affine_chain(points_t, affine_chain)

        return points_t

    # Gives us "pipeline[i]" access to transforms.
    def __get_transform_groups(
        self,
        ) -> List[List[Transform]]:
        current_types = None
        transform_groups = []
        transform_group = []

        for i, t in enumerate(self.__transforms):
            if isinstance(t, Identity):
                continue

            # Add transform to group.
            if current_types is not None and isinstance(t, tuple(current_types)):
                # Append transform to existing transform group of same type.
                transform_group.append(t)
            else:
                # Close out existing transform group - unless first iteration.
                if current_types is not None:
                    transform_groups.append(transform_group)
                
                # Start new transform group.
                if isinstance(t, IntensityTransform):
                    current_types = [IntensityTransform]
                else:
                    current_types = [GridTransform, SpatialTransform]
                transform_group = [t]

        # Add final group.
        if len(transform_group) > 0:
            transform_groups.append(transform_group)

        return transform_groups

    # Groups transforms by type (intensity vs. grid/spatial).
    # This is because we can "backward_transform_points" through a whole
    # group before performing a single resample.
    def __get_transform_groups_grid_params(
        self,
        grid: SamplingGridTensor,
        ) -> List[List[SamplingGridTensor]]:
        # Each group contains the input grid params to each transform in the group (required
        # for some transforms), plus the final grid params (required for resampling groups).
        current_types = None
        grid_groups = []
        grid_group = []
        grid_t = grid

        for i, t in enumerate(self.__transforms):
            if isinstance(t, Identity):
                continue

            # Add transform to group.
            if current_types is not None and isinstance(t, tuple(current_types)):
                grid_group.append(grid_t)
            else:
                # Close out existing grid group - unless first iteration.
                if current_types is not None:
                    grid_group.append(grid_t)    # Add final grid params to group.
                    grid_groups.append(grid_group)
                
                # Append transform to new group of new type.
                if isinstance(t, IntensityTransform):
                    current_types = [IntensityTransform]
                else:
                    current_types = [GridTransform, SpatialTransform]
                grid_group = [grid_t]
    
            # Update grid params.
            if isinstance(t, GridTransform):
                grid_t = t.transform_grid(grid_t)

        # Add final group - final transform could have been identity.
        if len(grid_group) > 0:
            grid_group.append(grid_t)    # Add final grid params to group.
            grid_groups.append(grid_group)

        return grid_groups

    # Returns input/output grid params for all transform groups.
    def __getitem__(
        self,
        i: int,
        ) -> Transform:
        return self.__transforms[i]

    def __resolve_affine_chain(
        self,
        points: PointsTensor,
        chain: List[AffineTensor],
        ) -> PointsTensor:
        if self._verbose:
            logger.info(f"Resolving affine chain of length {len(chain)}.")
        points_h = torch.hstack([points, torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)])  # Move to homogeneous coords.
        chain = [c.to(points.dtype) for c in chain]
        points_h_t = torch.linalg.multi_dot(chain + [points_h.T]).T
        points_t = points_h_t[:, :-1]
        return points_t
        
    def __str__(self) -> str:
        params = dict(
            transforms=self.__transforms,
        )
        return super().__str__(self.__class__.__name__, params)

    @alias_kwargs([
        ('a', 'affine'),
    ])
    def transform_images(
        self,
        image: Image | List[Image],
        affine: Affine | None = None,
        return_affine: bool = False,
        return_single: bool = True,
        ) -> Image | List[Image | Affine]:
        images, image_was_single = arg_to_list(image, (np.ndarray, torch.Tensor), return_expanded=True)
        return_types = [type(i) for i in images]
        device = get_group_device(images, device=self._device)
        images = [to_tensor(i, device=device) for i in images]
        dims = [len(i.shape) for i in images]
        size = to_tensor(images[0].shape[-self._dim:], device=device, dtype=torch.int32)
        affine = to_tensor(affine, device=device, dtype=torch.float32)

        # Check image n_dims, and spatial sizes.
        for i, img in enumerate(images):
            n_dims = len(img.shape)
            possible_dims = list(range(self._dim, self._dim + 3))   # E.g. for 3D, possible dims are 3-5 (3D spatial, optional batch/channel).
            assert n_dims in possible_dims, f"Expected {self._dim}-{self._dim + 2}D image ({self._dim}D spatial, optional batch/channel), got {n_dims}D for image {i}."
            assert img.shape[-self._dim:] == images[0].shape[-self._dim:], f"All images must have the same spatial size. Expected {tuple(images[0].shape[-self._dim:])}, got {tuple(img.shape[-self._dim:])} for image {i}."

        # Load transforms - grouped by intensity or grid/spatial types.
        transform_groups = self.__get_transform_groups()

        # Save the data required for each resampling step.
        # Resampling requires a tensor of sample locations in the moving image and
        # the grid params defining the tensor position in patient coords.
        moving_grids = []       # List[SamplingGridTensor]
        resample_points_list = []    # List[PointsTensor] 

        # Get grid params for each transform group.
        grid_groups = self.__get_transform_groups_grid_params((size, affine))

        # Calculate info needed for resampling steps: moving grid params, resampling points
        # plus final grid for returning to user. 
        for ts, gs in zip(transform_groups, grid_groups):
            if isinstance(ts[0], IntensityTransform):
                # Ensuring arrays have same length as number of transforms.
                moving_grids.append(None)
                resample_points_list.append(None)
            elif isinstance(ts[0], (GridTransform, SpatialTransform)):
                # Get final grid points.
                points_t = grid_points(*gs[-1]).to(device)

                # Back transform to their moving image locations.
                # - Each transform requires the input grid params.
                points_t = self.__backward_transform_points_for_group(ts, points_t, gs[:-1])

                # Reshape points to the fixed image size.
                points_t = points_t.reshape(*to_tuple(gs[-1][0]), self._dim)

                # Append to resampling info.
                moving_grids.append(gs[0])
                resample_points_list.append(points_t)

        assert len(moving_grids) == len(transform_groups), f"Got {len(moving_grids)}, expected {len(transform_groups)}"
        assert len(resample_points_list) == len(transform_groups), f"Got {len(resample_points_list)}, expected {len(transform_groups)}"

        # Transform images.
        image_ts = []
        for i, (image, rt) in enumerate(zip(images, return_types)):
            image_t = image

            for j, ts in enumerate(transform_groups):
                if isinstance(ts[0], IntensityTransform):
                    # Perform all intensity transforms in the transform group.
                    for t in ts:
                        image_t = t.transform_intensity(image_t)
                elif isinstance(ts[0], (GridTransform, SpatialTransform)):
                    # Perform a single resample for all grid/spatial transforms in the 
                    # transform group.
                    moving_grid = moving_grids[j]
                    moving_grid = (g.to(device) for g in moving_grid)
                    moving_size, moving_affine = moving_grid
                    # This warning is more for development.
                    if to_tuple(image_t.shape[-self._dim:]) != to_tuple(moving_size):
                        raise ValueError(f"Transform group {j} expected image to have spatial shape {to_tuple(moving_size)}, got {to_tuple(image_t.shape[-self._dim:])}.")
                    points = resample_points_list[j].to(device)

                    # Perform resample.
                    image_t = grid_sample(image_t, moving_affine, points.to(device))

            # Save resulting image.
            image_ts.append(image_t)

        # Convert to return format.
        other_data = []
        if return_affine:
            other_data.append(gs[-1][1])   # Final grid affine.
        results = to_return_format(image_ts, other_data=other_data, return_single=return_single or image_was_single, return_types=return_types)

        return results

    def transform_points(
        self,
        points: Points | List[Points],
        affine: Affine | None = None,       # Required for some transforms, e.g. Rotate, to get centre of rotation.
        filter_offgrid: bool = True,
        # grid: SamplingGrid | None = None,   # Required for filtering off-grid points and some transforms, e.g. Rotate.
        return_filtered: bool = False,
        return_single: bool = True,
        size: Size | None = None,           # Required for filtering off-grid points.
        **kwargs,
        ) -> Points | List[Points | Indices | List[Indices]]:
        pointses, points_was_single = arg_to_list(points, (np.ndarray, torch.Tensor), return_expanded=True)
        device = get_group_device(pointses, device=self._device)
        pointses = [to_tensor(p, device=device) for p in pointses]
        return_types = [type(p) for p in pointses]
        size = to_tensor(size, device=device, dtype=torch.int32)
        affine = to_tensor(affine, device=device, dtype=torch.float32)

        points_ts = []
        indiceses = []
        for p in pointses:
            # Chain 'transform_points' calls for SpatialTransforms.
            grid_t = (size, affine)
            points_t = p
            affine_chain = []   # Resolve chains of 3x3 or 4x4 affines before applying to large Nx3 or Nx4 points matrix.
            for i, t in enumerate(self.__transforms):
                if isinstance(t, GridTransform):
                    # GridTransforms don't move points/objects.
                    # Get current SamplingGrid, transform might need e.g. for centre of image for flip/crop/rotate.
                    grid_t = t.transform_grid(grid_t)
                elif isinstance(t, Identity):
                    pass
                elif isinstance(t, IntensityTransform):
                    pass
                elif isinstance(t, SpatialTransform):
                    if isinstance(t, Affine):
                        # Store affine for later.
                        t_affine = t.get_affine_transform(device, grid=grid_t)
                        # Transform 't' iterates forwards through the transform list.
                        # We want transforms that are earlier in the list to be applied first (i.e. to be
                        # later in the affine chain). So prepend transforms to the list.
                        affine_chain.insert(0, t_affine)
                    else:
                        # Resolve chain.
                        if len(affine_chain) > 0:
                            points_t = self.__resolve_affine_chain(points_t, affine_chain)
                            affine_chain = []

                        # Perform current transform.
                        points_t = t.transform_points(points_t, filter_offgrid=False, grid=grid_t)
                else:
                    raise ValueError(f"Unrecognised transform type: {type(t)}.")

            # Resolve affines if final transform.
            if len(affine_chain) > 0:
                points_t = self.__resolve_affine_chain(points_t, affine_chain)

            # Filter off-grid points.
            if filter_offgrid:
                size_t, affine_t = grid_t
                assert size_t is not None, "Size is required for filtering off-grid points."
                assert affine_t is not None, "Affine is required for filtering off-grid points."
                fov_d = fov(size_t, affine=affine_t)
                to_keep = (points_t >= fov_d[0]) & (points_t < fov_d[1])
                to_keep = to_keep.all(axis=1)
                points_t = points_t[to_keep].reshape(-1, self._dim)   # If a single point, shape could be (3, ) instead of (1, 3).
                indices = torch.where(~to_keep)[0].type(torch.int32)
                indiceses.append(indices)

            points_ts.append(points_t)

        # Convert to return format.
        other_data = []
        if filter_offgrid and return_filtered:
            indiceses = to_return_format(indiceses, return_single=True, return_types=return_types)
            other_data.append(indiceses)
        results = to_return_format(points_ts, other_data=other_data, return_single=return_single and points_was_single, return_types=return_types)
        return results

    @property
    def transforms(self) -> List[Transform]:
        return self.__transforms

    def __warn_resamples(self) -> None:
        # If there are multiple 'grid/spatial' groups, multiple resamples will be triggered.
        groups = self.__get_transform_groups()
        gs_groups = [g for g in groups if isinstance(g[0], (GridTransform, SpatialTransform))]
        n_resamples = len(gs_groups)
        if n_resamples > 1:
            logger.warning(f"Separating grid/spatial transforms with intensity transforms will trigger additional resampling steps " \
f"({n_resamples} resamples total for current pipeline). Consider moving intensity transform/s to first/last position.")

# A Pipeline is by default a 'RandomTransform'. Therefore it inherits 'transform_images/points' from
# 'RandomTransform', which freezes the pipeline before applying the transform.
# When 'Pipeline.freeze' returned a 'Pipeline', this introduced recursive calls to 'freeze'.
class Pipeline(RandomTransform):
    def __init__(
        self,
        transforms: RandomTransform | Transform | List[RandomTransform | Transform],
        freeze: bool | List[bool] = False,
        seed: int | List[int] | None = None,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        transforms = arg_to_list(transforms, (Transform))
        freezes = arg_to_list(freeze, (bool, None), broadcast=len(transforms))
        seeds = arg_to_list(seed, (int, None), broadcast=len(transforms))
        assert len(seeds) == len(transforms), "Random seeds ('seed') must have same length as 'transforms'."
        [t.set_debug(self._debug) for t in transforms]
        if self._device is not None:
            [t.set_device(self._device) for t in transforms]
        if self._dim is not None:
            assert self._dim in [2, 3], "Only 2D and 3D pipelines are supported."
            [t.set_dim(self._dim) for t in transforms]
        else:
            dim = transforms[0].dim
            for t in transforms:
                assert t.dim == dim, "All transforms must have same 'dim'."

        # Reseed the random transforms if requested - just easier doing it during pipeline creation rather than
        # for each transform.
        [t.set_seed(s) for s, t in zip(seeds, transforms) if s is not None and isinstance(t, RandomTransform)]

        # Freeze transforms if requested.
        transforms = [t.freeze() if f and isinstance(t, RandomTransform) else t for f, t in zip(freezes, transforms)]

        self.__transforms = transforms
        self._params = dict(
            transforms=[t.params for t in self.__transforms],
            type=self.__class__.__name__,
        )

    def freeze(self) -> 'FrozenPipeline':
        transforms = [t.freeze() if isinstance(t, RandomTransform) else t for t in self.__transforms]
        return FrozenPipeline(transforms, device=self._device, dim=self._dim)

    def __getitem__(
        self,
        i: int,
        ) -> Transform:
        return self.__transforms[i]
        
    def __str__(self) -> str:
        params = dict(
            transforms=self.__transforms,
        )
        return super().__str__(self.__class__.__name__, params)

    @property
    def transforms(self) -> List[Transform]:
        return self.__transforms
