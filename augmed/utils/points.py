import numpy as np
import torch

from ..typing import Indices, Points
from .args import arg_to_list
from .conversion import to_numpy, to_tensor

def filter_points(
    points: Points,
    indices: Indices,
    ) -> Points:
    points, return_type = to_tensor(points, return_type=True)
    indices = arg_to_list(indices, int)
    mask = torch.ones(points.shape[0], dtype=torch.bool)
    mask[indices] = False
    points = points[mask]
    if return_type is np.ndarray:
        points = to_numpy(points)
    return points
