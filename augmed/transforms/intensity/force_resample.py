from ...typing import ImageTensor
from ...utils.python import set_private_attr
from .intensity import IntensityTransform

# This is really just a utility class for triggering resamples in the pipeline
# for testing purposes. It doesn't actually change intensities.
class ForceResample(IntensityTransform):
    def __init__(
        self,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        set_private_attr(self, '__params', dict(
            dim=self.__dim,
            type=self.__class__.__name__,
        ))

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().__str__(self.__class__.__name__, subtransform=subtransform)

    def transform_intensity(
        self,
        image: ImageTensor,
        ) -> ImageTensor:
        return image
