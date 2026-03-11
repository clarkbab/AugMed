from .affine import Affine, RandomAffine

# This does preserve parallel lines.
class Shear(Affine):
    pass

class RandomShear(RandomAffine):
    pass
