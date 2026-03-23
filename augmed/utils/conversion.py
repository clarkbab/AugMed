import numpy as np
import torch
from typing import List, Tuple

from ..typing import AffineMatrix, Image, Number, Points, TransformParams
from .python import delegates_to

def to_numpy(
    data: bool | Number | str | List[bool | Number | str] | np.ndarray | torch.Tensor | torch.Size,
    broadcast: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    ) -> np.ndarray | None:
    if data is None:
        return None

    # Convert data to array.
    if isinstance(data, (bool, float, int, str)):
        data = np.array([data])
    if isinstance(data, (list, tuple)):
        data = np.array(data)
    elif isinstance(data, torch.Size):
        data = np.array(data)
    elif isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # Set data type.
    if dtype is not None:
        data = data.astype(dtype)

    # Broadcast if required.
    if broadcast is not None and len(data) == 1:
        data = np.repeat(data, broadcast)

    return data

@delegates_to(to_numpy)
def to_list(
    data: bool | Number | str | List[bool | Number | str] | np.ndarray | torch.Tensor | torch.Size,
    **kwargs,
    ) -> List[bool | Number | str] | None:
    if data is None:
        return None 
    return to_numpy(data, **kwargs).tolist()

def to_return_format(
    data: Points | List[Image],
    return_single: bool = True,
    return_types: type | List[type] | None = None,
    other_data: List[AffineMatrix | TransformParams] | None = None,
) -> Image | Points | List[Image | Points]:
    # Can't use "arg_to_list" because of circular dependencies.
    if isinstance(data, (np.ndarray, torch.Tensor)):
        data = [data]
    if isinstance(return_types, type):
        return_types = [return_types] * len(data)
        assert len(return_types) == len(data), f"Length of 'return_types' must match length of 'data'. Expected {len(data)}, got {len(return_types)}."

    # Convert data items to return types.
    if return_types is not None:
        for i, (d, rt) in enumerate(zip(data, return_types)):
            if rt is np.ndarray:
                data[i] = to_numpy(d)
            elif rt is torch.Tensor:
                data[i] = to_tensor(d)
            else:
                raise ValueError(f"Unsupported return type '{rt}'. Supported types are 'np.ndarray' and 'torch.Tensor'.")

    # Add "other data". This could be affines or transform params.
    if other_data is not None:
        data += other_data

    # Convert to a single value if appropriate.
    if return_single and len(data) == 1:
        data = data[0]

    return data

def to_tensor(
    data: bool | Number | str | List[bool | Number | str] | np.ndarray | torch.Tensor | torch.Size,
    broadcast: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    return_type: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor | None, type] | None:
    # Record input type.
    if return_type:
        input_type = type(data)

    # Convert to tensor.
    if isinstance(data, (bool, float, int, str)):
        device = torch.device('cpu') if device is None else device  
        data = torch.tensor([data], device=device, dtype=dtype)
    elif isinstance(data, (list, tuple, np.ndarray, torch.Size)):
        device = torch.device('cpu') if device is None else device  
        data = torch.tensor(data, device=device, dtype=dtype)
    elif isinstance(data, torch.Tensor):
        device = data.device if device is None else device
        dtype = data.dtype if dtype is None else dtype
        data = data.to(device=device, dtype=dtype)

    # Broadcast if required.
    if broadcast is not None and len(data) == 1:
        data = data.repeat(broadcast)

    if return_type:
        return data, input_type
    else:
        return data

@delegates_to(to_numpy)
def to_tuple(
    data: bool | Number | str | List[bool | Number | str] | np.ndarray | torch.Tensor | torch.Size,
    decimals: int | None = None,
    **kwargs,
    ) -> Tuple[bool | Number | str, ...] | None:
    if data is None:
        return None 
    # Convert to tuple.
    data = tuple(to_numpy(data, **kwargs).tolist())

    # Round elements if required.
    if decimals is not None:
        data = tuple(round(x, decimals) if isinstance(x, float) else x for x in data)

    return data
