import numpy as np
import torch
from typing import List

from ..typing import Number
from .conversion import to_list, to_tensor

def round(
    x: Number | List[Number] | np.ndarray | torch.Tensor,
    # Can round either by number of decimal places, or by tolerance.
    dp: int | None = None,
    tol: Number | None = None,
    ) -> Number | List[Number] | np.ndarray | torch.Tensor:
    x, return_type = to_tensor(x, dtype=torch.float64, return_type=True)
    assert (dp is not None) ^ (tol is not None), "Specify either dp or tol, not both."
    if dp is not None:
        x = torch.round(x * (10 ** dp)) / (10 ** dp)
    else:
        x = tol * torch.round(x / tol)
    if return_type is int or return_type is float:
        x = return_type(x[0])
    elif return_type is list:
        x = to_list(x)
    elif return_type is np.ndarray:
        x = x.cpu().numpy()
    return x
