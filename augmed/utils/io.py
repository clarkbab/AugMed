import json
import numpy as np
import os
import torch
from typing import Any, List
import yaml

from ..typing import FilePath, Image

def load_numpy(
    filepath: str,
    keys: str | List[str] = 'data',
    ) -> Image | List[Image]:
    assert filepath.endswith('.npy') or filepath.endswith('.npz'), "Filepath must end with .npy or .npz"
    data = np.load(filepath)
    if filepath.endswith('.npz'):
        keys = arg_to_list(keys, str)
        data = [data[k] for k in keys]
        data = data[0] if len(data) == 1 else data
    return data

# JSON/YAML don't know how to serialize tensors, so we need to convert them to lists first.
def make_serialisable(
    param: Any,
    ) -> Any:
    if isinstance(param, dict):
        return {k: make_serialisable(v) for k, v in param.items()}
    elif isinstance(param, list):
        return [make_serialisable(v) for v in param]
    elif isinstance(param, tuple):
        return tuple(make_serialisable(v) for v in param)
    elif isinstance(param, torch.Tensor):
        return param.detach().cpu().tolist()
    else:
        return param

def save_json(
    data: Any,
    filepath: FilePath,
    overwrite: bool = True,
    ) -> None:
    if os.path.exists(filepath) and not overwrite:
        raise ValueError(f"File '{filepath}' already exists, use overwrite=True.")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = make_serialisable(data)
    if filepath.endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
        with open(filepath, 'w') as f:
            yaml.dump(data, f)
