# Augmed

A PyTorch-based data augmentation library with GPU acceleration and native 2D/3D image and point cloud transforms.

## Installation

```
pip install augmed
```

# Usage

## Minimal example

```python
from augmed import Crop, Normalise, Pipeline, RandomAffine, RandomElastic
from augmed.utils import load_example_ct, plot_volume
import torch

# Plot example data.
ct, affine, labels, points = load_example_ct()
plot_volume(ct, affine=affine, labels=labels, points=points)

# Define pipeline.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
pipeline = Pipeline([
    Crop(),
    RandomAffine(),
    RandomElastic(),
    Normalise()
], device=device)

# Transform example CT image.
ct_t, labels_t, points_t, affines_t = pipeline(ct, labels, points, affine=affine, return_affine=True)

# Plot transformed data.
plot_volume(ct_t, affine=affines_t[0], labels=labels_t, points=points_t)
```

## Motivation

There are several desirable properties for data augmentation for medical imaging - existing libraries only support these sparsely.

| Library | 2D images | 3D images | GPU acceleration | Single resample pipeline | Point clouds |
| :------ | :-------: | :-------: | :---------: | :-------------: | :----------: |
| albumentations | &#x2713; | &#x2717; | &#x2717; | &#x2717; | &#x2717; |
| augmed | &#x2713; | &#x2713; | &#x2713; | &#x2713; | &#x2713; |
| monai | &#x2713; | &#x2713; | Partial (some transforms) | Partial - do they provide joint transforms ([affine only](https://docs.monai.io/en/latest/transforms.html#lazytrait)) | &#x2717; |
| torchio | &#x2713; | &#x2713; | Partial (joint transforms) | [&#x2717;](https://github.com/TorchIO-project/torchio/blob/8065c45838ce92a0bbddb5f6b65319ea93b7deaa/src/torchio/transforms/augmentation/composition.py#L55) | [&#x2717;](https://github.com/TorchIO-project/torchio/issues/1274) |
| torchvision | &#x2713; | &#x2717; | &#x2713; | [&#x2717;](https://github.com/pytorch/vision/blob/ccb801b88af136454798b945175c4c87e636ac33/torchvision/transforms/v2/_container.py#L52) | [&#x2713;](https://docs.pytorch.org/vision/main/generated/torchvision.tv_tensors.KeyPoints.html#torchvision.tv_tensors.KeyPoints) ([approx. for elastic transform](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.ElasticTransform.html#torchvision.transforms.v2.ElasticTransform)) |

### GPU acceleration
Why is GPU support important? Show benchmark for data augmentation times, in particular for expensive elastic deformation.

Augmed addresses this problem by using `torch.Tensor` objects internally, which natively support assignment to GPU devices. Images/points will be transformed using their current device (or CPU for `np.ndarray` inputs). Alternatively, the `device` param may be set on a `Transform` to force the device for all inputs.

Do we need to ensure that all transforms in a Pipeline are on the same device if set on init? Could be a problem when pulling out affine transforms. I'd say leave this until it actually becomes a problem.

### Point clouds
Add some references for why point cloud transforms can be useful:
- Use as supervision for medical image registration (largely unsupervised).

Currently, no other libraries support 3D point cloud transformations. 

Image resampling transforms operate from the output image grid points to their transformed locations in the input image space. Point clouds require the inverse transform, which is easily attained for affine transforms. For elastic transforms, we ensure a calculable inverse by preventing folding (constrain deformation magnitudes to less than half of the control grid spacing), and calculate the true inverse using a fast-converging iterative method. 

### Single resample pipeline
Why is a single resample important?
- Loss of spatial information (e.g. crop followed by rotate).
- Loss of high-frequency information (how do we show)?

Existing libraries provide partial support for single resampling by allowing chaining of affine transforms, which doesn't extend to elastic deformations, or by providing combination transforms (e.g. AffineElastic).

In augmed, the `Pipeline` ensures that only a single resampling step is applied. Firstly, the pipeline performs all `GridTransform` transforms to calculate the final sample grid location (size and affine) - which determines output image's sample point locations. These points are then propagated to the input image by applying the `back_transform_points` method for each `Transform` in the pipeline in reverse order. Finally, the input image is resampled once using these sample point locations.

### Other features

#### Pipeline optimisations
Benchmark these optimisations:

- Only the final `SamplingGrid` points are back transformed to the input image - i.e. all `GridTransform` objects are applied first.
- Grouping input images by `SamplingGrid` and performing `Pipeline` transforms' `backward_transform_points` once to get sample locations in the input image space. Useful when multiple images have same `SamplingGrid`, e.g. CT volume + labels.
- For chained `Affine` transforms, pull out the backward affine matrix (4x4 for 3D) for each transform and collapse these before applying to the sampling points (Nx3 for 3D, with N=6.7e7 for a 512x512x256 volume).
- Anything else?

#### Data types
Transforms accept images/points of types `torch/np.float32`, `torch/np.float16`, or `torch/np.bfloat16`. Internal calculations will be carried out using these types, but may be overridden for a `Transform` using the `dtype` param.

- Don't need to handle float64 I think.
- Store affine matrices (and other bits and pieces) as float32 and downcast as necessary based on resolved 'dtype'. Hmm, actually it would be better to instantiate at lower res if possible - based on `dtype` param and `set_dtype` - called from Pipeline.
- Should probably enforce that all transforms in a Pipeline use the same dtype (if set on Transform, not input) as we'll need to multiple affine chains, for example - and backward_transform_points. Actually, just leave this until it is a problem. Most people will be setting at the Pipeline level I'd say.

#### Frozen transforms
All `RandomTransform` types offer a `freeze` method that returns a deterministic transform for repeatability - although it is preferable to pass all data to `pipeline()` in a single call to make use of the built-in optimisations. 

## Types

### Images
An `Image` is a `torch.Tensor` or `np.ndarray` of size `(B, C, X, Y, Z)`, where B/C dimensions are optional, and Z is excluded for 2D images. For 2D images, the `dim=2` param must be set on transforms. `ImageTensor` is similar but is of type `torch.Tensor` only.

`LabelImage` is also a `torch.Tensor/np.ndarray` but of type `bool`. Transforms will apply nearest-neighbour interpolation when resampling label images.

### Points
A `Points` object is a `torch.Tensor/np.ndarray` of size `(N, X, Y, Z)` where Z is excluded for 2D points. `PointsTensor` is of type `torch.Tensor` only.

### SamplingGrid
A `SamplingGrid` of type `Tuple[Size, Affine | None]` defines the image sampling grid (a.k.a field-of-view, view window). If the `Affine` is none, then transforms will be applied using image (pixel/voxel) coords and may be incorrect for images with anisotropic spacing.

### Transform
`Transform` is the base type for all transforms, including `Pipeline`. All subclasses must implement `transform_images` and `transform_points` methods.

### RandomTransform
A `RandomTransform` is a special type of transform that behaves non-deterministically, but can yield a deterministic transform through the `freeze()` method. `RandomTransforms` are only applied a proportion `p` of the time, and will `freeze()` to the `IdentityTransform` when not applied.

### GridTransform
A `GridTransform` changes the image `SamplingGrid`, for example by cropping/padding the image. These transforms must implement the `transform_grid()`, which accepts and returns a `SamplingGrid`.

Do these transforms also need to implement `transform_points` (or maybe just crop does)? This is because some points will be filtered by crop and we'd like to know when points go offscreen. We need this for plotting, we also probably don't want to optimise a loss function based on points that aren't visible.

### IntensityTransform
An `IntensityTransform` changes the intensity of voxels/pixels in the image, e.g. normalisation. These transforms implement the `transform_intensity()` method, which accepts and returns an `ImageTensor` object.  

To avoid multiple resampling steps when using `Pipeline`, intensity transforms should be grouped at the start and/or end of the pipeline. This is because intensity transforms require resolving (resampling) of all previous `GridTransform` or `SpatialTransform` objects to determine the intensities passed to `transform_intensities()`.

### SpatialTransform
A `SpatialTransform` changes the positions of objects within the image, e.g. rotation, elastic deformation. These transforms implement the `backward_transform_points()` method that accepts and returns a `PointsTensor`.
