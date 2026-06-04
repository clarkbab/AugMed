from __future__ import annotations

import numpy as np
import numpy as np
import torch
import torch
from typing import List, Literal, Tuple, Union

from ..typing import AffineMatrix, AffineMatrixTensor, BatchChannelImage, BatchImage, BatchLabelImage, ChannelImage, Image, Indices, LabelImage, Number, Points, PointsTensor, SamplingGrid, Size, SpatialDim, TransformParams
from ..utils.args import alias_kwargs, arg_to_list
from ..utils.assertions import assert_image_shapes, assert_image_sizes, assert_points_shapes
from ..utils.conversion import to_return_format, to_tensor, to_tuple
from ..utils.geometry import create_affine, fov, spatial_size
from ..utils.grid import grid_points, grid_sample
from ..utils.logging import logger
from ..utils.python import get_group_device, set_private_attr
from .grid.grid import GridTransform
from .identity import Identity
from .intensity.intensity import IntensityTransform
from .spatial import Affine
from .spatial.spatial import SpatialTransform
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

    def __back_transform_group_points(
        self,
        transforms: List[Transform],     # TODO: SpatialTransforms?
        points: PointsTensor,
        grids: List[SamplingGrid],     # These are the input grids to each transform - required by some, e.g. Rotate.
        **kwargs,
        ) -> PointsTensor:
        points_t = points
        assert len(grids) == len(transforms), f"Got {len(grids)} grids, expected {len(transforms)} - same length as transforms."

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
        for t, g in reversed(list(zip(transforms, grids))):
            # Chain resolution conditions:
            # 1. Non-affine transform.
            # 2. Final transform.
            if isinstance(t, SpatialTransform):
                # Store any affine multiplications for later.
                if isinstance(t, Affine):
                    t_affine = t.get_affine_back_transform(points_t.device, grid=g)
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
                    points_t = t.back_transform_points(points_t, grid=g)

        # Resolve remaining affine chain.
        if len(affine_chain) > 0:
            points_t = self.__resolve_affine_chain(points_t, affine_chain)

        return points_t

    # Performs the back transform for a grid/spatial group applying
    # the affine optimisation if possible.
    def __get_transform_groups(
        self,
        grid: SamplingGrid | None = None,
        ) -> List[List[Transform]] | Tuple[List[List[Transform]], List[List[SamplingGrid]]]:
        transform_groups = []
        # Group types just defines the types that can be added to the current group.
        # This is defined by which groups trigger a resample.
        current_group_types = None
        # If computing grids for each transform group, these consist of the
        # input grid for each transfrom (required for some transforms, e.g.
        # Rotate), plus the final grid after all transforms in the group
        # (required for resampling).
        if grid is not None:
            grid_groups = []
            grid_t = grid

        for t in self.__transforms:
            if isinstance(t, Identity):
                continue

            # Start new transform group.
            if current_group_types is None or not isinstance(t, current_group_types):
                # Close out existing transform group.
                if current_group_types is not None:
                    transform_groups.append(current_transforms)
                    if grid is not None:
                        current_grids.append(grid_t)  # Add final grid - required for resampling.
                        grid_groups.append(current_grids)

                # Set the group type - only transforms of these types will be
                # added to the group.
                if isinstance(t, IntensityTransform):
                    current_group_types = (IntensityTransform,)
                else:
                    current_group_types = (GridTransform, SpatialTransform)

                # Start the transform group.
                current_transforms = [t]
                if grid is not None:
                    current_grids = [grid_t]

            # Append transform to existing transform group of same type.
            elif isinstance(t, current_group_types):
                current_transforms.append(t)
                if grid is not None:
                    current_grids.append(grid_t)
    
            # Update grid params.
            if grid is not None and isinstance(t, GridTransform):
                grid_t = t.transform_grid(grid_t)

        # Close out final group.
        if len(current_transforms) > 0:
            transform_groups.append(current_transforms)
            if grid is not None:
                current_grids.append(grid_t)  # Add final grid - required for resampling.
                grid_groups.append(current_grids)

        if grid is not None:
            return transform_groups, grid_groups
        else:
            return transform_groups

    # Splits pipeline into groups oftransforms that can be performed with a 
    # single resample.
    # For example:
    # - Flip + Rotate + Crop + MinMax (single resample at end).
    # - Rotate + Crop + MinMax + Flip (two resamples, one before MinMax, one at end).
    # If 'grid' is passed then grid params are return for each transform in each
    # transform group. These are required for resampling at the end of the group
    # and for some transforms, e.g. Rotate - to determine the 'image-centre' for
    # rotation.
    def __getitem__(
        self,
        i: int,
        ) -> Transform:
        return self.__transforms[i]

    # Let's use use pipeline[0] syntax.
    @property
    def params(self) -> TransformParams:
        return super().params(transforms=[t.params for t in self.__transforms])

    def __resolve_affine_chain(
        self,
        points: PointsTensor,
        chain: List[AffineMatrixTensor],
        ) -> PointsTensor:
        if self.__verbose:
            logger.info(f"Resolving affine chain of length {len(chain)}.")
        points_h = torch.hstack([points, torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)])  # Move to homogeneous coords.
        chain = [c.to(points.dtype) for c in chain]
        points_h_t = torch.linalg.multi_dot(chain + [points_h.T]).T
        points_t = points_h_t[:, :-1]
        return points_t
        
    def __str__(self) -> str:
        return self.to_str()

    def to_str(self) -> str:
        transforms_str = '[' + ', '.join(t.to_str(subtransform=True) for t in self.__transforms) + ']'
        return super().__str__(
            self.__class__.__name__,
            skip_format='transforms',
            transforms=transforms_str,
        )

    @alias_kwargs(
        ('a', 'affine'),
        ('f', 'fill'),
        ('i', 'interpolation'),
        ('rg', 'return_grid'),
    )
    def transform_images(
        self,
        images: Image | LabelImage | BatchImage | BatchLabelImage | ChannelImage | BatchChannelImage | List[Image | LabelImage | BatchImage | BatchLabelImage | ChannelImage | BatchChannelImage],
        affine: AffineMatrix | None = None,
        fill: Number | Literal['border', 'max', 'min', 'reflection', 'zeros'] | None = None,
        interpolation: Literal['bicubic', 'bilinear', 'nearest'] | None = None,
        return_grid: bool = False,
        ) -> Image | LabelImage | BatchImage | BatchLabelImage | ChannelImage | BatchChannelImage | List[Image | LabelImage | BatchImage | BatchLabelImage | ChannelImage | BatchChannelImage | AffineMatrix | SamplingGrid]:
        assert_image_shapes(images, self.__dim)
        assert_image_sizes(images, self.__dim)
        images = arg_to_list(images, (np.ndarray, torch.Tensor))
        return_types = [type(i) for i in images]
        device = get_group_device(images, device=self.__device)
        images = [to_tensor(i, device=device) for i in images]
        size = to_tensor(spatial_size(images[0], self.__dim), device=device, dtype=torch.int32)
        affine = to_tensor(affine, device=device, dtype=torch.float32)
        fill = fill if fill is not None else self.__fill
        interpolation = interpolation if interpolation is not None else self.__interpolation

        # Load transforms - grouped by intensity or grid/spatial types.
        grid = (size, affine)
        transform_groups, tg_grids = self.__get_transform_groups(grid=grid)

        # For resampling, we need the input (moving) image grid (first grid)
        # and the output (fixed) image grid (final grid) for each transform group.
        # We then need to back transform the resampling points derived from the 
        # output grid.
        moving_grids = []       # List[SamplingGridTensor]
        moving_points = []      # List[PointsTensor] 
        for ts, gs in zip(transform_groups, tg_grids):
            if isinstance(ts[0], IntensityTransform):
                # No resampling required for intensity transform groups.
                # Add 'None' because len of these lists should match the
                # transform groups.
                moving_grids.append(None)
                moving_points.append(None)

            elif isinstance(ts[0], (GridTransform, SpatialTransform)):
                # Get resampling grid points.
                # The final grid defines the resampling points.
                fixed_grid = gs[-1] 
                points_t = grid_points(fixed_grid).to(device)

                # Back transform sampling grid points to their moving image locations.
                # This requires the input grids for each transform, as some transforms
                # require grid for 'image-centre' for example.
                other_grids = gs[:-1]
                points_t = self.__back_transform_group_points(ts, points_t, other_grids)

                # Reshape points to the fixed image size.
                fixed_size, _ = fixed_grid
                points_t = points_t.reshape(*to_tuple(fixed_size), self.__dim)

                # Append to resampling info.
                moving_grid = gs[0]
                moving_grids.append(moving_grid)
                moving_points.append(points_t)

        assert len(moving_grids) == len(transform_groups), f"Got {len(moving_grids)}, expected {len(transform_groups)}"
        assert len(moving_points) == len(transform_groups), f"Got {len(moving_points)}, expected {len(transform_groups)}"

        # Transform images.
        image_ts = []
        for i, image in enumerate(images):
            image_t = image

            for j, ts in enumerate(transform_groups):
                if isinstance(ts[0], IntensityTransform):
                    # Perform successive intensity transforms.
                    for t in ts:
                        image_t = t.transform_intensity(image_t)

                elif isinstance(ts[0], (GridTransform, SpatialTransform)):
                    # Perform a single resample for the whole transform group.
                    moving_grid = moving_grids[j]
                    moving_grid = (g.to(device) if g is not None else g for g in moving_grid)
                    moving_size, moving_affine = moving_grid
                    # This warning is more for development.
                    if to_tuple(spatial_size(image_t, self.__dim)) != to_tuple(moving_size):
                        raise ValueError(f"Transform group {j} expected image {i} to have spatial shape {to_tuple(moving_size)}, got {to_tuple(spatial_size(image_t, self.__dim))}.")
                    points = moving_points[j].to(device)

                    # Perform resample.
                    image_t = grid_sample(image_t, points, affine=moving_affine, dim=self.__dim, fill=fill, interpolation=interpolation)

            # Save resulting image.
            image_ts.append(image_t)

        # Convert to return format.
        other_data = []
        if return_grid:
            grid_t = gs[-1]   # Final grid.
            other_data.append(grid_t)
        return to_return_format(image_ts, other_data=other_data, return_types=return_types)

    @alias_kwargs(
        ('a', 'affine'),
        ('fo', 'filter_offgrid'),
        ('rf', 'return_filtered'),
        ('s', 'size'),
    )
    def transform_points(
        self,
        points: Points | List[Points],
        affine: AffineMatrix | None = None,       # Required for some transforms, e.g. Rotate, to get centre of rotation.
        filter_offgrid: bool | SpatialDim | List[SpatialDim] | None = None,
        return_filtered: bool = False,
        size: Size | None = None,           # Required for filtering off-grid points.
        **kwargs,
        ) -> Points | List[Points | Indices | List[Indices]]:
        assert_points_shapes(points, self.__dim)
        points = arg_to_list(points, (np.ndarray, torch.Tensor))
        device = get_group_device(points, device=self.__device)
        points = [to_tensor(p, device=device) for p in points]
        return_types = [type(p) for p in points]
        size = to_tensor(size, device=device, dtype=torch.int32)
        affine = to_tensor(affine, device=device, dtype=torch.float32)
        filter_offgrid = filter_offgrid if filter_offgrid is not None else self.__filter_offgrid

        points_ts = []
        indiceses = []
        for p in points:
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
                        t_affine = t.get_affine_transform(device, grid_t)
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
            if filter_offgrid is not False:    # "is not" required here because filter_offgrid=0 is valid.
                size_t, affine_t = grid_t
                assert size_t is not None, "Size is required for filtering off-grid points."
                if affine_t is None:
                    affine_t = create_affine(device=device, dim=self.__dim)
                fov_d = fov((size_t, affine_t))
                if filter_offgrid is True:
                    in_fov = (points_t >= fov_d[0]) & (points_t <= fov_d[1])
                else:
                    dims = arg_to_list(filter_offgrid, int)
                    in_fov = (points_t[:, dims] >= fov_d[0][dims]) & (points_t[:, dims] <= fov_d[1][dims])
                to_keep = in_fov.all(axis=1)
                points_t = points_t[to_keep].reshape(-1, self.__dim)   # If a single point, shape could be (3, ) instead of (1, 3).
                indices = torch.where(~to_keep)[0].type(torch.int32)
                indiceses.append(indices)

            points_ts.append(points_t)

        # Convert to return format.
        other_data = []
        if filter_offgrid and return_filtered:
            indiceses = to_return_format(indiceses, return_types=return_types)
            other_data.append(indiceses)
        return to_return_format(points_ts, other_data=other_data, return_types=return_types)

    @property
    def transforms(self) -> List[Transform]:
        return self.__transforms

    def __warn_resamples(self) -> None:
        # If there are multiple 'grid/spatial' groups, multiple resamples will be triggered.
        groups = self.__get_transform_groups()
        gs_groups = [g for g in groups if isinstance(g[0], (GridTransform, SpatialTransform))]
        n_resamples = len(gs_groups)
        if n_resamples > 1:
            logger.warn(f"Separating grid/spatial transforms with intensity transforms will trigger additional resampling steps " \
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
        [t.set_debug(self.__debug) for t in transforms]
        if self.__device is not None:
            [t.set_device(self.__device) for t in transforms]
        if self.__dim is not None:
            assert self.__dim in [2, 3], "Only 2D and 3D pipelines are supported."
            [t.set_dim(self.__dim) for t in transforms]
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
        set_private_attr(self, '__params', dict(
            dim=self.__dim,
            transforms=[t.params for t in self.__transforms],
            type=self.__class__.__name__,
        ))

    def freeze(self) -> FrozenPipeline:
        transforms = [t.freeze() if isinstance(t, RandomTransform) else t for t in self.__transforms]
        return FrozenPipeline(
            transforms,
            debug=self.__debug,
            device=self.__device,
            dim=self.__dim,
            fill=self.__fill,
            filter_offgrid=self.__filter_offgrid,
            interpolation=self.__interpolation,
        )

    def __getitem__(
        self,
        i: int,
        ) -> Transform:
        return self.__transforms[i]
        
    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        transforms_str = '[' + ', '.join(t.to_str(subtransform=True) for t in self.__transforms) + ']'
        return super().__str__(
            self.__class__.__name__,
            skip_format='transforms',
            subtransform=subtransform,
            transforms=transforms_str,
        )

    @property
    def transforms(self) -> List[Transform]:
        return self.__transforms
