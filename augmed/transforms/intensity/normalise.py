from typing import *

from ...typing import *
from .intensity import IntensityTransform, RandomIntensityTransform

class RandomNormalise(RandomIntensityTransform):
    def __init__(
        self,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        # Randomise mean/std using some range.

    def freeze(self) -> 'Normalise':
        pass

class Normalise(IntensityTransform):
    def __init__(
        self,
        mean: Number = 0,
        std: Number = 1,
        ) -> None:
        pass

    # How on earth do we deal with this type of transform?

    # New thoughts.
    # 1. Some grid/spatial transforms will change the distribution of intensities.
    # - Resize can halve the number of intensities which will alter the histogram.
    # - Elastic deformation can move objects around in the image which will change
    #   image histogram and spatial positions of intensities.
    # 2. Intensity transforms will change depending upon the image histogram and spatial positions of intensities.
    # - Min/max normalisation is calculated using min/max histogram statistics.
    # - Gaussian blurring at a particular voxel depends on the intensity values of neighbouring voxels.

    # Approaches.
    # 1. We could try to be clever and use approximate intensity transforms.
    # - Resize might halve the number of samples - coarsening the histogram - but we could approximate
    # by using the initial histogram.
    # - Crop might alter the histogram, but we could assume that the transform is calculated on the full image
    # (not just onscreen) as a simplifying assumption.
    # - Not sure about elastic deformation, as we really need to know the resulting intensity distribution before
    # applying normalisation (for example) and intensity locations before apply intensity transforms that require
    # spatial locations.
    # 2. Enfore intensity transformations at either beginning/end of pipeline. This might be the best approach
    # (for now). What use cases are there for requiring intensity transforms in the middle of the pipeline?
    # 3. Preferred option: Allow intensity transforms to be placed anywhere in the pipeline - but, warn that
    # this will result in multiple resampling steps and they should be placed first/last.
    
    # Should we differentiate between intensity transforms that require histogram statistics and those
    # that require spatial locations of intensities?
    # - Pairing transforms that only require hist stats with some simplifiying assumptions, might mean that we can
    # put intensity transforms anywhere in the pipeline.
    
    # The real question here is how do we handle intensity transforms that are in the middle of the pipeline.
    


    # 1. If the intensity transform (IT) is at the start of a pipeline, then there is no issue,
    # we just scale all the intensity values before resampling the image.
    # 1b. Similarly, if the IT is at the end of the pipeline, we can just perform the spatial transforms
    # and then scale intensity values as needed - no issues.
    # 2. What if we apply grid transforms and then apply normalisation? Grid transforms will change
    # the image histogram by either removing or adding values.
    # - e.g. crop reduces lower-end values, pad might introduce more lower-end.
    # - resize could for example halve the number of intensity values.
    # 2 answer. I think we can simplify this by making the assumptions that objects aren't added or 
    # removed from the image until the final resample (i.e. just because an object goes out of view
    # mid-pipeline, this doesn't mean it won't be included for intensity calculation purposes.)
    # - But what about changing numbers of samples (e.g. resize)? Any resize is just going to reduce
    # image information, i.e. coarsen the histogram. I think we should assume that intensity transforms
    # operate at the highest number of samples.
    # Otherwise, we'd have to perform parts of the pipeline to calculate intermediate intensities and
    # then perform intensity normalisation before performing the next part of the pipeline - multiple resamples?
    # E.g. Translation -> Rotation -> Resize(2x downsample) -> Norm -> Elastic. If we're normalising mid-pipeline, 
    # we need to resolve the intensity values after the resize transform, which means 'backward_transform_points'
    # for Translation -> Rotation -> Resize, then perform Norm transform, then backward_transform_points for Elastic.
    # 2x resampling.
    # Alternatively, we could push the Norm transform to the front (or make this an overridable default):
    # Norm -> Translation -> Rotation -> Resize -> Elastic. In this format, we can perform the intensity normalisation
    # which doesn't involve resampling, followed by a single resample using 'backward_transform_points' of the 
    # remaining spatial transforms.
    # Now we run into the issue of spatial transforms that deform the intensity histogram, i.e. non-rigid transforms.
    # Obvious example is elastic deformation, which could expand bony regions and increase the high-end of the histogram.
    # Does scaling deform the histogram? It does, but I think only by a factor, as doubling the size of objects along an
    # axis will double the number of voxels with each intensity, i.e. the histogram is twice as high.
    # So scaling transforms won't change any of the histogram stats (i.e. min/max/mean/median) used for normalisation.
    # TODO: could easily check this for some sample images.
    # So perhaps, elastic deformation is a special case.
    # So if we perform an elastic deformation before our normalisation, what can we do?
    # E.g. Translation -> Rotation -> Elastic -> Norm -> Crop
    # The image histogram presented to Norm could be very different to that presented to Translation.
    # Possible solutions:
    # 1. 
    # Is there a way to forward propagate the image histogram - even if it's just an approximation.

    # What to do?
    # 1. For each spatial transform, test whether it affects the image histogram. How do we define a transform
    # that changes the image histogram - using a mixin/subclass?
    # 1a. Implement normalisation and test using non-elastic spatial transforms.
    # 2. If we only have a problem with Elastic transforms, then just let people do whatever, but warn them
    # that multiple resamples will be performed when using Norm transforms after Elastic and before other spatial transforms. 
    # We can then suggest a re-ordering of their pipeline. This is a pretty simple solution, and I don't see
    # how the order matters that much - i.e. why not just normalise first or last.
    # 3. Some forms of normalisation only require histogram stats. Is there a way to forward propagate these 
    # stats? I.e. we know the min/max values after the elastic deformation, and these can be used to apply the
    # normalisation as the first step, and this will be quite similar to performing it mid-way.
    # I'm not sure how this would work.
    # Other forms of normalisation might need intensity values in their spatial locations. I.e. some sort of 
    # gaussian smoothing applied over the image. We can't forward propagate stats here.
    # 

