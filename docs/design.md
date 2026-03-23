## Design decisions

TODO:
- Benchmark in comparison with other libraries: torchio, albumentations, monai, torchvision.
- Create test suite.
- Get some alpha/beta testing.

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

All points are expected in world coordinates, is this reasonable?

Transform_images accepts multiple images, but these images must share the sampling SamplingGrid as then we only need to compute resampling point positions once and apply to all images to create the resampled images. For transform_points, there's really no point in allowing a List[Points] as input, as there's no efficiency to be gained by batching these. But.. maybe it's better for the user to be able to pass transform_points[List[points]], like they can do with transform(List[Points])?

Should we allow users to compute List[Image] transforms on multiple devices with a single "transform_images" call? I don't think this will give much speed-up as most of the work is the "back_transform_points" for a huge points array which is shared by all images - also, which device do we assign to do this task?. The resampling step should be fairly quick by comparison.

In that case, how do we assign a device to perform the transforms? Transform.__init__(device=...) allows us to set this at a high level. But if this is not set: For "transform_points" it's easy, just use the device of the points torch.Tensor (or CPU if np.ndarray). For "transform_images", we'll just have to select the device of images[0].

We should match the input types when setting return types. I.e. if the user passed all numpy arrays, they should get numpy arrays out, even though Transform(device='cuda').

### Thoughts

What's the difference between Transform._params and the params set in Transform.__str__. Transform._params is a complete set of the values required to replicate the transform, whereas Transform.__str__ prints a human-friendly version of these params but skips large params (e.g. matrices) and rounds values for viewing.
