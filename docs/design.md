## Design decisions

### API

Public methods should take np.ndarray/torch.Tensor types and perform calcs on the same device as the passed tensor (or CPU for numpy), however device can be overridden at transform instantation. Internal methods (e.g. transform_grid) should accept only torch.Tensor types.

Public methods should accept size/affine separately as only size or affine may be required. Also, grouping these might be confusing for the user. For internal methods, it's convenient to think of these as a group as they define the sampling grid. So some internal methods take a SamplingGrid (size, affine | None), instead of the separate params.

Public methods:
- transform/transform_images/transform_points
- fov/fov_centre/fov_width

Internal methods:
- back_transform_points
- transform_grid
- transform_intensity

All points are expected in world coordinates (size/affine required only for filtering), but this might be changed in future. I don't think we could infer voxel coords from absence of affine as it's only used for filtering - we'd need a param, e.g. "use_image_coords".
