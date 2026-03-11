# Augmed

## Installation

```
pip install augmed
```

## Example usage

- When are affine/size needed during transforms?
    - TODO: Show examples of where all this goes wrong!
    - Affine default to spacing=1mm and origin=0mm respectively - which is equivalent to voxel coords.
    - Image transforms do not require spacing but transforms will not be accurate if images have isotropic spacing and spacing is not passed. TODO: Show incorrect rotation because non-iso spacing wasn't passed.
    - Image transforms do not require origin except when transforming images and points to ensure points and image objects are transformed correctly. TODO: Show separate transforming of points and images where origin wasn't set on the image, and images/points were transformed differently.
    - Size is inferred for image transforms from the passed image size.
    - Size is not required for points transforms if we don't want to filter off-grid points after transformation, or if points are transformed alongside image/s - as size can be inferred as long as all images have the same size. TODO: Show error when points without image, or points with images of different sizes.

# Usage

## Minimal example

## Detailed example

TODO: Maybe turn this into a jupyter notebook.

```python
import augmed as am

### Create pipeline.

pipeline = am.Pipeline([
  # In general:
  # Passing a single param (a) to a range transform gives a range of [0, a] along each axis.
  # Passing two params (a, b) to a range transforms gives a range of [a, b] along each axis.
  # Passing six params (a, b, c, d, e, f) gives ranges of [a, b]/[c, d]/[e, f] along x/y/z axes.

  am.RandomCrop(50),                          # Crop 0-50mm from each axis. 
  # am.RandomCrop((30, 50)),                  # Crop 30-50mm from each axis.
  # am.RandomCrop((30, 50, 30, 50, 0, 0)),    # Crop 30-50mm from x/y axes.
  am.RandomFlip(0.5),                         # Flip along each axis 50% of the time.
  # am.RandomFlip((0, 0.5, 0)),               # Flip along y-axis 50% of the time.
  am.RandomRotation(15)                       # Rotate 0-15 degrees around each axis.
  # am.RandomRotation((15, 15, 0)),           # Rotate 0-15 degrees around x/y axes.

],
# dim=2,                                      # For 2D images.
# freeze=True,                                # For reproducable transforms.
)

### Apply transforms to images and points.

# Images have dimensions (N, C, X, Y, Z) (or (N, C, X, Y) for 2D) - following the PyTorch convention. N, C are
# optional dimensions, with AugMed inferring the presence/position of a dimension from the image size and the
# the transform "dim" attribute.
image = ...     # 2-5D image
label = ...     # 2-5D image

# Points have dimensions (N, 3) (or (N, 2) for 2D).
points = ...    # 2-3D array of points

### Apply the pipeline to images/points.

# Passed arrays are inferred as points if their final dimension has size=3 (or size=2 for 2D), otherwise
# they're handled as images.
# For pipelines with points, affine must be passed to locate the points relative to images. Points are
# assumed to be in the same world coordinates as the image.
image_t, label_t, points_t = p([image, label, points], affine)       # Images and points.

# For image-only pipelines, affine optional (defaults to spacing: 1mm iso. and origin: (0, 0, 0)), however not passing
# these will result in inaccurate transformations (TODO: see example below).
image_t, label_t = p([image, label])                                          # Images only.

# For points-only pipelines, size must be passed if "filter_offgrid=True" - removes points which are transformed
# outside of the image field-of-view. Otherwise, "size" is inferred from images passed to the pipeline alongside
# the points.
points_t = p(points, affine, size)                                   # Points only.

# Specify different affines for images/points with different world coordinates.
image_spacing = (1.175, 1.175, 2.0)
image_origin = (-300.0, -300.0, 250.0)
image_affine = create_affine(image_spacing, image_origin)
points_spacing = (2.0, 2.0, 2.0)
points_origin = (125.0, 125.0, 0.0)
points_affine = create_affine(points_spacing, points_origin)
image_t, points_t = p([image, points], [image_affine, points_affine])
```

## Motivation

What properties are desirable for a medical imaging data augmentation library?

- **Point transforms**. Point transforms provided for all transform classes.\
Image registration models benefit from the inclusion of anatomical landmarks during training (references?).\
However, existing augmentation libraries only provide image augmentation methods. Augmed provides point (forward) \
transforms, in addition to image (backward) transforms for all transform classes. This includes elastic deformation,\
where invertibility can be ensured by constraining the magnitude of random displacements.
- **Single resample**. Augmed performs a single resample for all pipeline transforms.\
Performing a resampling can lose offscreen information. For example, a rotation may bring voxels onscreen that were
previously moved offscreen by a translation. If resampling at each step, the offscreen information is lost.
Performing sequential image resampling increases image artifacts and reduces high-frequency components (references?).\
When performing sequential data augmentations (e.g. elastic -> affine -> crop) it would be beneficial to perform a single resampling operation.
- **GPU-accelerated**: GPU support provided for all transform classes.\
Using GPUs for data augmentation decreases training times and GPU idle times (references?), particularly\
when using complex transforms like elastic deformation (references?).
- **2/3D image support**: Self-explanatory. References?
- **Implicit API**. Rather than defining data types, use existing types for images and points and let the transform figure\
out the types based on the input shapes and data types.

## Transforms

### GridTransforms

- Grid or WindowTransform?
  - I'm concerned that we might want to use WindowTransform for windowing the data (e.g. applying
    a lung window to CT data).
- We could implement some grid transforms (e.g. crop/pad) as just adding or removing voxels. But for
  consistency with the other grid transform (resize), we just adjust the sample points and then resample.
  For crop/pad this won't actually produce and interpolation. TODO: test this.

### IntensityTransforms

- Spatial and grid transforms require resolving (resampling) before applying intensity transforms.
  - Spatial transform: Yes, these will change the distribution of intensity samples.
  - Grid transform: Yes. Some grid transforms definitely change the distribution (e.g. resize). 
  But how about crop/pad? What would the user expect from a crop/pad followed by normalisation? I think
  for a crop, you'd expect only to calculate the normalisation across intensity values in the window. For example,
  imagine images with large background regions, the user might want to crop to a region of interest and then
  normalise based on these values alone. It would make sense for the lowest value within this window to be zero
  after normalisation - not some much higher value based on setting offscreen values to zero. Therefore, a crop
  probably requires resolution before performing intensity transforms. How about pad? If the user padded the image
  with background, you'd expect this background to be set to zero after normalisation, you wouldn't expect it to be
  ignored. I think the answer is YES, crop/pad should trigger a resample so that subsequent intensity transforms
  are calculated on the expected distribution of intensities. 
- Intensity transforms in middle positions will require multiple resampling steps.
  - We need to break up our Pipeline.transform_image code to handle this. 
  - Currently grids are forward-transformed and points are back-transformed by group to get resampling point locations.
  - Then each image is resampled using these locations.
  - If we know the resampling locations, which we do from our fancy "resample_conditions" code, then we could break
    the pipeline up into "chunks" and run the whole forward grid, back points piece on each chunk - storing each
    transformed image as we go.

## Single resample

How do we achieve this property?

- Some transforms move points/objects around within an image (`SpatialTransform`).
- These transforms typically map resample points from fixed -> moving image.
- May knock resampling points off-grid and require interolation - increases image artifacts and removes high-frequency information (references?).
- When chaining transforms, more and more information is lost.
- Our `Pipeline` propagates resampling grid points from fixed -> moving image through all `SpatialTransform` objects in reverse order using
`backward_transform_points` method before performing a single resampling.
- `GridTransform` objects (e.g. crop/pad) in a `Pipeline` don't move points off-grid, but may increase or decrease the number of resampling grid points back-transformed.

## Point transforms

How do we achieve this property?

- For `SpatialTransform` objects, we define a 'transform_points' method that is the inverse of `backward_transform_points`.
- For elastic deformation, an inverse is ensured by keeping displacement magnitudes to less than half the control grid
spacing. Inverse is calculated using a GPU-accelerated iterative method.

## Performance tweaks

- Chained affine transforms are resolved by successive 4x4 matrix multiplications (homogeneous coords) before applying to large tensor of grid points (Nx4).

## Library comparison

Which MONAI transforms support a torch backend? 

| Library     | 3D image support | GPU support | Single resample | Point transforms |
| :-------    | :------: | :------: | -------: | ------: |
| monai       | &#x2713; | Partial (some transforms) | Partial ([affine only](https://docs.monai.io/en/latest/transforms.html#lazytrait)) | &#x2717;
| torchio     | &#x2713; | [&#x2717;](https://github.com/TorchIO-project/torchio/issues/388)  | [&#x2717;](https://github.com/TorchIO-project/torchio/blob/8065c45838ce92a0bbddb5f6b65319ea93b7deaa/src/torchio/transforms/augmentation/composition.py#L55) | [&#x2717;](https://github.com/TorchIO-project/torchio/issues/1274)
| torchvision | &#x2717; | &#x2713; | [&#x2717;](https://github.com/pytorch/vision/blob/ccb801b88af136454798b945175c4c87e636ac33/torchvision/transforms/v2/_container.py#L52) | [&#x2713;](https://docs.pytorch.org/vision/main/generated/torchvision.tv_tensors.KeyPoints.html#torchvision.tv_tensors.KeyPoints) ([approx. for elastic](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.ElasticTransform.html#torchvision.transforms.v2.ElasticTransform))
| albumentations |

## Transform types

- Spatial transforms: Move the positions of objects in the image.
- Intensity transforms: Change the appearance of objects in the image.
- Field-of-view transforms: Change the view window size and position (e.g. crop/pad). Sampling (draw random patches from the image), Patching (draw a batch of regularly spaced samples from the image).

## Transform API

Implicit or explicit API?
- The point of the implicit API was that extra work wasn't required to wrap arrays/tensors in library-specific data types.
- But many libraries define specific data types (e.g. ScalarImage, LabelImage) that contains (origin/spacing/)

- 3-5D arrays or tensors (N, C, X, Y, Z) are accepted with 2-3 spatial dimensions and optional batch/channel. The spatial
dimension is determined through the 'dim' parameter shared by all transforms.
- Types (image vs. points, intensity vs. label image) are determined by the passed array/tensor shapes and types.
- When are size/spacing/origin needed during transforms?
    - TODO: Show examples of where all this goes wrong!
    - Spacing/origin default to 1mm and 0mm respectively - which is equivalent to voxel coords.
    - Image transforms do not require spacing but transforms will not be accurate if images have isotropic spacing and spacing is not passed. TODO: Show incorrect rotation because non-iso spacing wasn't passed.
    - Image transforms do not require origin except when transforming images and points to ensure points and image objects are transformed correctly. TODO: Show separate transforming of points and images where origin wasn't set on the image, and images/points were transformed differently.
    - Size is inferred for image transforms from the passed image size.
    - Size is not required for points transforms if we don't want to filter off-grid points after transformation, or if points are transformed alongside image/s - as size can be inferred as long as all images have the same size. TODO: Show error when points without image, or points with images of different sizes.
- What about 'return_grid' parameter? When transforming using GridTransforms, we might need to return the grid for plotting (so that we know
where our landmarks sit relative to the image origin). However, the API is getting quite convoluted.
  - Do we need to return the grid for each transformed image? E.g: (image_t, grid_t), (label_t, _), points_t = p(image, label, points, return_grid=True)
  - Or do we just return a separate grid tuple? E.g: image_t, label_t, points_t, (grid_t, _) = p (image, label, points, return_grid=True)
  - Just return a separate grid tuple (or list of grid tuples) as the last returned value.

## Range API

For many random transforms we have to define ranges of values.

For example, how much to crop from an image?
We define this as a tuple that tells us how much to remove from each end of each axis.
E.g. crop=(80, 120, 80, 120, ...) tells us to remove between 80-120mm from each end of
the x-axis. These ranges can be asymmetric of course, if we'd like to remove more from one
end of the image -> e.g. crop=(0, 0, 80, 120) won't remove anything from the lower end of
the x-axis.
So what if we would like symmetric crops?
We might define crop=(80, 120, 80, 120, ...), but we'd actually like x (80 <= x <= 120) to
match on both ends. We should have a 'symmetric: bool' parameter that controls this.
Mostly, people will be passing crop=(80, 120) which will expand to symmetric ranges, no problem.
If someone passes an asymmetric range, e.g. crop=(0, 0, 80, 120) but requests 'symmetric=True'
we should raise an error.

## Monitoring errors
- Opt-in telemetry for raised exceptions?
- "report_error" block that will catch and send exception info?

## Patient or voxel coords.
- Is "use_world_coords" necessary? I mean do we need to pass it anyway (it could be there as a notification). This behaviour
could be inferred at transform-time (not set at instantiation-time), through the presence/absence of the "affines" parameter.
- All values are specified in patient coordinates (mm).
- Should we provide an option for voxels? E.g. what if someone wants to crop 10-20 voxels from
an image? They would need to calculate this value in mm before applying to the image, and then rounding, which for cropping and padding will favour adding voxels, might mean this value is wrong. This shouldn't be the case.
- If someone uses spacing=1mm, then crop=10mm will be the same as 10 voxels. But someone might want to
crop 10 voxels, for a spacing=2mm or other image.
- Each transform should have a "use_world_coords: bool = True" parameter that can be set to false if passed
parameters are in image coordinates. Parameters will be converted to mm internally.
- Why not 'use_world_coords', isn't that clearer? Because the images may not be of patients? It could be bacteria in a petri dish :)
- Should each transform type have the 'uic' parameter? Spatial transform - yes, grid transforms - yes, intensity transforms
  - I could imagine a case where they do, for example we're defining a blurring kernel and we need to define the spatial 
  neighbourhood of the kernel.

## Phantom parameters
- What do we do about parameters that make sense for most transforms, but not for some transforms. At the moment
we've pushed these up to the superclass, but they look weird for some transforms. E.g. why do intensity transforms
care about the 'dim' param?

## Identity transform
- At the moment this is a spatial transform, but that makes no sense.
- We created the identity transform so that random pipelines wouldn't change length when frozen, I still think this makes
sense and is clearer than just returning shorter pipelines if transforms aren't applied.
- An identity transform needs the basic API (transform_image, transform_points) and it should also be handled (ignored!)
appropriately by the pipeline. However, apart from that it doesn't need any transform subtype-specific methods (e.g.
backward_transform_points, transform_grid, transform_intensity).

## Data types
- For the most part we should use torch.float32 data types so that transforms play nicely when used together in a pipeline. E.g.
we can't have some transforms affine matrices with dtype=float64, while others have dtype=float32.
- But should we respect the data type of the passed images/points? This does align with our other API principle of allowing the
user to pass data in flexible formats. And for bool we respect the data type.
- Some people might want lower precision inputs for mem savings. 
- This would mean, instead of hard-coding transform parameters (e.g. translation matrix) at float32, we should just let
to_tensor() handle the conversion and then convert matrices to the 'image' or 'points' type where necessary.

## Grouping spatial transforms
- If two images have the same grid, we currently only back transform the grid points once. Why is this code duplicated in the SpatialTransform.transform_image
and Pipeline.transform_image? Remember that SpatialTransform.transform_image only executes for single transforms, not for pipelines. Perhaps we could extract
this logic to a shared method?

## Determining transform type
- How do we know that our random transform is an IntensityTransform? This is useful if we want to determine that intensity transforms are placed
in the middle of the pipeline (triggering multiple resamples).
  - Why do our intensity transforms inherit from IntensityTransform whilst our random intensity transforms do not? IntensityTransform provides the shared
  'transform_image/points' methods that are identical across all intensity transforms. Random intensity transforms define their own implementation of these
  methods (that freeze and then run the frozen method). 
  - Could our random transforms also inherit from IntensityTransform. This would be purely for type-checking in Pipeline :)

## API accessibility
- 'backward_transform_points', and other transform-specific methods (transform_grid, transform_intensity) need to be public methods so that pipeline
  can consume these. Pipeline.backward_transform_points should also be public for consistency - and someone may have a need for back-propagating points!
- Other API public methods are transform_image/points - do we need a backward_transform_image method?
- In general, all attributes/methods that are only consumed in-class should be private. If the method or attribute is accessed (defined!) in the parent
  class, then use a single underscore (e.g. self._dim) to show that it's private to the class and parent classes.

## Pipelines

- How does a pipeline with random transforms behave? When we call 'transform' method on such a pipeline, certain methods will be called
  on the transform depending upon it's type (e.g. backward_transform_points for SpatialTransform, transform_grid for GridTransform). For 
  random transforms, these methods are available but trigger a freeze and forwarded call to the frozen transform.
- How does grouping work with pipelines?
  - We split into groups to determine how many resampling steps need to occur - one for each grid/spatial group.
  - Intensity transforms need all other transforms to be resolved before they can be computed. This is because all other transforms
  can change the distribution of ONSCREEN intensity values, on which the intensity transform will operate. It only makes sense for
  intensity transforms to operate on onscreen values, for example, a crop followed by normalise shouldn't consider all the cropped
  background.
  - We handle grouping by splitting into groups of consecutive intensity transforms and groups of consecutive grid/spatial transforms.
    The grid/spatial groups require resampling. Intensity groups are simple voxel intensity updates. Grid/spatial transforms may
    require a resampling operation.
  - A group containing only crop/pad transforms will actually not perform any interpolation during resampling. However! Information can
  be lost during the resample (e.g. crop) that would have been brought onscreen by a rotation (for example) later in the pipeline. It's
  not just loss of high-frequency information that is important.

GridParams:
These define a grid in space. Need a size, but is affine optional? I think size is enough for image coords. So maybe
GridParams type should be (Size, Affine | None).

# Requirements:
- numpy
- pyyaml
- torch
