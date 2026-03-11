from .intensity import IntensityTransform

from ...typing import *

# This is really just a utility class for triggering resamples in the pipeline
# for testing purposes. It doesn't actually change intensities.
class ForceResample(IntensityTransform):
    def __init__(
        self,
        **kwargs) -> None:
        super().__init__(**kwargs)
        self._params = dict(
            type=self.__class__.__name__,
            dim=self._dim,
        )

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__, {})

    def transform_intensity(
        self,
        image: ImageTensor,
        ) -> ImageTensor:
        return image
