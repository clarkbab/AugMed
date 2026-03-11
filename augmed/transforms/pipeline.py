import numpy as np
import torch
from typing import *

from ..typing import *
from ..utils.args import alias_kwargs, arg_to_list
from ..utils.conversion import to_array, to_tensor, to_tuple
from ..utils.geometry import fov
from ..utils.grid import grid_points, grid_sample
from ..utils.logging import logger
from ..utils.matrix import create_affine
from .grid import GridTransform
from .identity import Identity
from .intensity import IntensityTransform
from .spatial import Affine, SpatialTransform
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
            type=self.__class__.__name__,
            transforms=[t.params for t in self.__transforms],
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
            size, affine = g
            grid_kwargs = dict(
                size=size,
                affine=affine,
            )

            # Chain resolution conditions:
            # 1. Non-affine transform.
            # 2. Final transform.
            if isinstance(t, SpatialTransform):
                # Store any affine multiplications for later.
                if isinstance(t, Affine):
                    t_affine = t.get_affine_backward_transform(points_t.device, **grid_kwargs)
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
                    points_t = t.backward_transform_points(points_t, **grid_kwargs)

            # Resolve if final round.
            if i == len(transforms) - 1 and len(affine_chain) > 0:
                points_t = self.__resolve_affine_chain(points_t, affine_chain)

        return points_t

    # Gives us "pipeline[i]" access to transforms.
    def __getitem__(
        self,
        i: int,
        ) -> Transform:
        return self.__transforms[i]

    # Groups transforms by type (intensity vs. grid/spatial).
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

    # Returns input/output grid params for all transform groups.
    def __get_transform_groups_grid_params(
        self,
        size: SizeTensor,
        affine: AffineTensor | None = None,
        ) -> List[List[SamplingGridTensor]]:
        # Each group contains the input grid params to each transform in the group (required
        # for some transforms), plus the final grid params (required for resampling groups).
        current_types = None
        grid_groups = []
        grid_group = []
        size_t, affine_t = size, affine

        for i, t in enumerate(self.__transforms):
            if isinstance(t, Identity):
                continue

            # Add transform to group.
            if current_types is not None and isinstance(t, tuple(current_types)):
                grid_group.append((size_t, affine_t))
            else:
                # Close out existing grid group - unless first iteration.
                if current_types is not None:
                    grid_group.append((size_t, affine_t))    # Add final grid params to group.
                    grid_groups.append(grid_group)
                
                # Append transform to new group of new type.
                if isinstance(t, IntensityTransform):
                    current_types = [IntensityTransform]
                else:
                    current_types = [GridTransform, SpatialTransform]
                grid_group = [(size_t, affine_t)]

            # Update grid params.
            if isinstance(t, GridTransform):
                size_t, affine_t = t.transform_grid(size_t, affine=affine_t)

        # Add final group - final transform could have been identity.
        if len(grid_group) > 0:
            grid_group.append((size_t, affine_t))    # Add final grid params to group.
            grid_groups.append(grid_group)

        return grid_groups

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
        return f"{self.__class__.__name__}({self.__transforms})"

    @alias_kwargs([
        ('a', 'affine'),
    ])
    def transform_image(
        self,
        image: Image | List[Image],
        affine: Affine | List[Affine] = None,
        return_grid: bool = False,
        ) -> Image | List[Image | SamplingGrid]:
        images, image_was_single = arg_to_list(image, (np.ndarray, torch.Tensor), return_expanded=True)
        if self._verbose:
            logger.info(f"Transforming {len(images)} images.")
        return_types = ['numpy' if isinstance(i, np.ndarray) else 'torch' for i in images]
        affines = arg_to_list(affine, (np.ndarray, torch.Tensor, None), broadcast=len(images))
        images = [to_tensor(i, device=self._device) for i in images]
        dims = [len(i.shape) for i in images]
        if self._dim == 2:
            for i, d in enumerate(dims):
                assert d in [2, 3, 4], f"Expected 2-4D image (2D spatial, optional batch/channel), got {d}D for image {i}."
        elif self._dim == 3:
            for i, d in enumerate(dims):
                assert d in [3, 4, 5], f"Expected 3-5D image (3D spatial, optional batch/channel), got {d}D for image {i}."
        sizes = [to_tensor(i.shape[-self._dim:], device=i.device, dtype=torch.int32) for i in images]
        affines = [to_tensor(a, device=i.device, dtype=torch.float32) if a is not None else create_affine(spacing=(1,) * self._dim, origin=(0,) * self._dim, device=i.device, return_type='torch') for a, i in zip(affines, images)]

        # How do we handle image and transform grouping.
        # - Transform grouping allows us to chain intensity and grid/spatial transforms.
        # - Image grouping is an optimisation that allows us to calculate resampling positions
        #   for images with the same grid params only once.
        # - Now that multiple resampling steps may be applied, we need to store multiple resampling
        #   position points tensors for each image group. 
        # - For resampling, we require the tensor of back-transformed points (final grid params generate
        #   these points, then 'backward_transform_points' places them in moving image space) and the grid
        #   params (affine, size) for the resampled image (first image in group).

        # Group images by grid parameters: affine, spacing
        # We need to know which groups there are -> i.e. the first image in each grid group.
        # We need to know the mapping from image to group.
        image_groups_map = []       # Maps images (by position in 'image_groups') to groups.
        for i, (a, s) in enumerate(zip(affines, sizes)):
            # Check if this image has same grid params as an existing group.
            image_groups = np.unique(image_groups_map).tolist()
            for j in image_groups:
                g_a, g_s = affines[j], sizes[j]
                if torch.all(a == g_a) and torch.all(s == g_s):
                    image_groups_map.append(j)
                    break

            # Otherwise, add to a new group.
            if len(image_groups_map) != i + 1:
                image_groups_map.append(i)

        # Load transforms - grouped by intensity or grid/spatial types.
        transform_groups = self.__get_transform_groups()

        # Save the data required for each resampling step.
        # Resampling requires a tensor of sample locations in the moving image and
        # the grid params defining the tensor position in patient coords.
        image_groups = np.unique(image_groups_map).tolist()
        moving_grids = []       # List[List[SamplingGridTensor]]
        resample_points = []    # List[List[PointsTensor]] 
        final_grids = []        # List[SamplingGridTensor]
        for i in image_groups:
            image_group_moving_grids = []
            image_group_resample_points = []

            # Get grid params for each transform group.
            size, affine = sizes[i], affines[i]
            grid_groups = self.__get_transform_groups_grid_params(size, affine=affine)

            # Calculate info needed for resampling steps: moving grid params, resampling points
            # plus final grids for returning to user. 
            for ts, gs in zip(transform_groups, grid_groups):
                if isinstance(ts[0], IntensityTransform):
                    # Ensuring group arrays have same length as number of transforms.
                    image_group_moving_grids.append(None)
                    image_group_resample_points.append(None)
                elif isinstance(ts[0], (GridTransform, SpatialTransform)):
                    # Get final grid points.
                    points_t = grid_points(*gs[-1]).to(size.device)

                    # Back transform to their moving image locations.
                    # - Each transform requires the input grid params.
                    points_t = self.__backward_transform_points_for_group(ts, points_t, gs[:-1])

                    # Reshape points to the fixed image size.
                    points_t = points_t.reshape(*to_tuple(gs[-1][0]), self._dim)

                    # Append to group resampling info.
                    image_group_moving_grids.append(gs[0])
                    image_group_resample_points.append(points_t)

            # Append group results.
            moving_grids.append(image_group_moving_grids)
            resample_points.append(image_group_resample_points)
            final_grids.append(gs[-1])

        assert len(moving_grids) == len(image_groups)
        assert len(resample_points) == len(image_groups)
        assert len(final_grids) == len(image_groups)
        assert len(moving_grids[0]) == len(transform_groups), f"Got {len(moving_grids[0])}, expected {len(transform_groups)}"
        assert len(resample_points[0]) == len(transform_groups), f" Got {len(resample_points[0])}, expected {len(transform_groups)}"

        # Transform images.
        image_ts = []
        grid_ts = []
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
                    moving_grid = moving_grids[image_groups_map[i]][j]
                    moving_grid = (g.to(image.device) for g in moving_grid)
                    moving_size, moving_affine = moving_grid
                    # This warning is more for development.
                    if to_tuple(image_t.shape[-self._dim:]) != to_tuple(moving_size):
                        raise ValueError(f"Transform group {j} expected image to have spatial shape {to_tuple(moving_size)}, got {to_tuple(image_t.shape[-self._dim:])}.")
                    points = resample_points[image_groups_map[i]][j].to(image.device)

                    # Perform resample.
                    image_t = grid_sample(image_t, moving_affine, points)

            # Get the final grid.
            grid_t = final_grids[image_groups_map[i]]

            # Convert to return types.
            if rt == 'numpy': 
                image_t = to_array(image_t)
                if return_grid:
                    grid_t = tuple(to_array(g) for g in grid_t)
            image_ts.append(image_t)
            if return_grid:
                grid_ts.append(grid_t)

        results = image_ts[0] if image_was_single else image_ts
        if return_grid:
            if isinstance(results, list):
                results.append(grid_ts)
            else:
                results = [results, grid_ts]

        return results

    # This is for point clouds, not for image resampling. Note that this
    # requires invertibility of the back point transform, which may not be
    # be available for some transforms (e.g. folded elastic).
    def transform_points(
        self,
        points: Points,
        filter_offgrid: bool = True,
        grid: SamplingGrid | None = None,   # Required for filtering off-grid points and some transforms, e.g. Rotate.
        return_filtered: bool = False,
        **kwargs,
        ) -> Points:
        if isinstance(points, np.ndarray):
            points = to_tensor(points)
            return_type = 'numpy'
        else:
            return_type = 'torch'
        size = to_tensor(size, device=points.device, dtype=torch.int32)
        affine = to_tensor(affine, device=points.device)

        print('pipeline: transform points')
        print('points: ', points)

        # Chain 'transform_points' calls for SpatialTransforms.
        grid_t = (size, affine)
        points_t = points
        affine_chain = []   # Resolve chains of 4x4 affines before applying to large Nx4 points matrix.
        for i, t in enumerate(self.__transforms):
            print('transform: ', i)
            # Get current FOV, transform might need e.g. for centre of image for flip/crop/rotate.
            size_t, affine_t = grid_t
            print('grid params: ', size_t, affine_t)

            if isinstance(t, GridTransform):
                # GridTransforms don't move points/objects.
                grid_t = t.transform_grid(size_t, affine=affine_t)
            elif isinstance(t, Identity):
                pass
            elif isinstance(t, IntensityTransform):
                pass
            elif isinstance(t, SpatialTransform):
                # SpatialTransforms don't affect the grid.
                okwargs = dict(
                    affine=affine_t,
                    size=size_t,
                )
                if isinstance(t, Affine):
                    # Store affine for later.
                    t_affine = t.get_affine_transform(points_t.device, **okwargs)
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
                    points_t = t.transform_points(points_t, filter_offgrid=False, **okwargs)
            else:
                raise ValueError(f"Unrecognised transform type: {type(t)}.")

        # Resolve affines if final transform.
        if len(affine_chain) > 0:
            print('resolving final affine chain')
            print('chain: ', affine_chain)
            points_t = self.__resolve_affine_chain(points_t, affine_chain)

        # Filter off-grid points.
        if filter_offgrid:
            size_t, affine_t = grid_t
            fov_d = fov(size, affine_t)
            to_keep = (points_t >= fov_d[0]) & (points_t < fov_d[1])
            to_keep = to_keep.all(axis=1)
            points_t = points_t[to_keep]
            indices = torch.where(to_keep)[0]
            if return_type == 'numpy':
                points_t, indices = to_array(points_t), to_array(indices)
            if return_filtered:
                return points_t, indices
            else:
                return points_t
        else:
            if return_type == 'numpy':
                points_t = to_array(points_t)
            return points_t

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

# A Pipeline is by default a 'RandomTransform'. Therefore it inherits 'transform_image/points' from
# 'RandomTransform', which freezes the pipeline before applying the transform.
# When 'Pipeline.freeze' returned a 'Pipeline', this introduced recursive calls to 'freeze'.
class Pipeline(RandomTransform):
    def __init__(
        self,
        transforms: RandomTransform | Transform | List[RandomTransform | Transform],
        device: torch.device | Literal['cpu', 'cuda'] | None = None,
        dim: int | None = None,
        freeze: bool | List[bool] = False,
        seed: int | List[int] | None = None,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        transforms = arg_to_list(transforms, (Transform))
        freezes = arg_to_list(freeze, (bool, None), broadcast=len(transforms))
        seeds = arg_to_list(seed, (int, None), broadcast=len(transforms))
        assert len(seeds) == len(transforms), "Random seeds ('seed') must have same length as 'transforms'."
        if device is not None:
            [t.set_device(device) for t in transforms]
        if dim is not None:
            assert dim in [2, 3], "Only 2D and 3D pipelines are supported."
            [t.set_dim(dim) for t in transforms]
        else:
            dim = transforms[0].dim
            for t in transforms:
                assert t.dim == dim, "All transforms must have same 'dim'."

        # Can be a combination of random and deterministic transforms.
        # For example, we may always want to centre crop first, or extract patches last,
        # with random augmentations in the middle.
        self._dim = dim

        # Reseed the random transforms if requested - just easier doing it during pipeline creation rather than
        # for each transform.
        [t.set_seed(s) for s, t in zip(seeds, transforms) if s is not None and isinstance(t, RandomTransform)]

        # Freeze transforms if requested.
        transforms = [t.freeze() if f and isinstance(t, RandomTransform) else t for f, t in zip(freezes, transforms)]

        self.__transforms = transforms
        self._params = dict(
            type=self.__class__.__name__,
            transforms=[t.params for t in self.__transforms],
        )

    def freeze(self) -> 'FrozenPipeline':
        transforms = [t.freeze() if isinstance(t, RandomTransform) else t for t in self.__transforms]
        return FrozenPipeline(transforms)

    def __getitem__(
        self,
        i: int,
        ) -> Transform:
        return self.__transforms[i]

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.__transforms})"

    @property
    def transforms(self) -> List[Transform]:
        return self.__transforms
