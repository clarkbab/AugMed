import json
import os
import yaml
from typing import *

from ..typing import *

# JSON/YAML don't know how to serialize tensors, so we need to convert them to lists first.
def make_serialisable(
    param: Dict | List | Tuple | torch.Tensor,
    ) -> Dict | List | Tuple:
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

def save_params(
    params: Dict[int | str, Any],
    filepath: FilePath,
    overwrite: bool = True,
    ) -> None:
    if os.path.exists(filepath) and not overwrite:
        raise ValueError(f"File '{filepath}' already exists, use overwrite=True.")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    params = make_serialisable(params)
    if filepath.endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=4)
    elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
        with open(filepath, 'w') as f:
            yaml.dump(params, f)
