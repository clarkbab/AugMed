import numpy as np
import sys
import torch
from typing import List

from ..typing import Image, Points

def get_group_device(
    data: Image | Points | List[Image | Points],
    device: torch.device | None = None,
    ) -> torch.device:
    if device is not None:
        return device

    data = arg_to_list(data, (np.ndarray, torch.Tensor))
    if isinstance(data[0], torch.Tensor):
        return data[0].device

    return torch.device('cpu')

def is_windows() -> bool:
    return 'win' in sys.platform
