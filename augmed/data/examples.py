from importlib.resources import files
import numpy as np
import os
from typing import Tuple

from ..typing import Affine3D, Image3D, LabelImage3D, Points3D
from ..utils.io import load_nifti, load_numpy

def load_example_ct() -> Tuple[Image3D, Affine3D, LabelImage3D, Points3D]:
    data_dir = files("augmed.data")
    filepath = os.path.join(data_dir, "example_ct.nii.gz")
    image, affine = load_nifti(filepath)
    filepath = os.path.join(data_dir, "example_labels.npz")
    labels = load_numpy(filepath)
    filepath = os.path.join(data_dir, "example_points.npz")
    points = load_numpy(filepath)

    # Convert types.
    image = image.astype(np.float32)
    affine = affine.astype(np.float32)
    labels = labels.astype(np.bool_)
    points = points.astype(np.float32)

    return image, affine, labels, points
