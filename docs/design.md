## Design decisions

TODO:
- Use in your own projects:
  - Breathing signal prediction
  - Markerless lung model
- Benchmark in comparison with other libraries: torchio, albumentations, monai, torchvision.
  - Monai is much faster than AugMed for 3D, but we haven't optimised yet.
  - Make optimisations to be more competitive for CPU applications.
- Create test suite.
- Ask colleagues to test
- Write a paper
- Publish the library

OPTIMISATIONS (from profiling):
- Grid sample is very slow. Why doesn't monai have this problem? 

OPTIMISATIONS (potential)
- Pre-interpolate the coarse b-spline grid to full resolution using separable 1D convolutions (like MONAI).
- Reduce materialised tensors (avoid creating the 8.4M, 64, 3 intermediate).
- Grid transforms are implemented as resamples, which makes sense when chaining
  together, as it means that other transforms operate on reduced numbers of sampling
  points. However, for a standalone crop (or pipeline with single crop), it would be
  much faster to implement by remove voxels instead of resampling.

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

### Decisions

- Why do we expand range args for random transforms during freeze and not during init? This is because the "dim" param could be set by a pipeline after the transform has been initialised, and expanding range args requires the dim. However, this means that we can't debug the expanded range args by printing the random transform, i.e. we don't actually ever see this expansion as the user.
- String params vs raw params. Right now we save all params to the __params key and this can be used to return the actual params (full precision) used to perform the transform. Whereas the string params are rounded for visibility. I think we can just have a "def params" method that collects all the relevant private params to recreate the transform. When calling to string we could use a sanitised version of these params. 
- Currently, filter_offgrid filters points that are offgrid along any axis. However, for our 2D breathing signal prediction augmentation, we only want to filter points that are off along the x-axis, as we'll just renormalise our points along the y-axis before training anyway. So filter_offgrid should accept bool or SpatialDim. 
- Private attributes. I'm using get/set_private_attr so we can use '__<param>' in the leaf classes. But it looks very messy in some intermediate classes (e.g. Affine). The other option is just to go with a single underscore.
- Gaussian transforms are a bit different from other transforms. For example, Rotate is a deterministic transform that doesn't take a seed, whereas RandomRotate randomises the rotation parameters and produces a Rotate transform when frozen. However, for Gaussian, the low-level transform has randomness, but the transform parameters are deterministic (mean/std). It would be good to have a RandomGaussian transform that also randomises these parameters, e.g. mean=(-0.2, 0.2), std=(0.8, 1.2). But, we have a seed param that creates reproducable parameters, so this would pass the same frozen mean/std for a given seed, but is this seed also used for drawing the random values within Gaussian? Is there anything wrong with this approach?
- Why do we need to batch points (B x N x 3) instead of just concatenating? What if we have two 1D signals (breathing amplitude/phase) and we want to separate these during processing, perhaps because we want to normalise each independently. This is easier if our API (e.g. normalisation methods) accept a batch of points.
- Currently, intensity transforms don't operate on points. Makes sense! This is because points live in the world space and should only move using spatial transforms. But what if we want to normalise our points at the end of the pipeline? E.g. our breathing signal prediction points are tied to image points but then we want to minmax normalise before passing to our prediction model. Is this something we should offer in Augmed, or can it just be applied afterwards using the minmax utility?
