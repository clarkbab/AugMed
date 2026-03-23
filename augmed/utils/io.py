import json
import nibabel as nib
import nrrd
import numpy as np
import os
import torch
from typing import Any, List, Tuple
import yaml

from ..typing import AffineMatrix3D, FilePath, Image3D
from .args import arg_to_list

def load_nifti(
    filepath: FilePath,
    **kwargs,
    ) -> Tuple[Image3D, AffineMatrix3D]:
    assert filepath.endswith('.nii') or filepath.endswith('.nii.gz'), "Filepath must end with .nii or .nii.gz"
    img = nib.load(filepath)
    data = img.get_fdata()
    affine = img.affine
    return data, affine

def load_nrrd(
    filepath: FilePath,
    ) -> Tuple[Image3D, AffineMatrix3D]:
    data, header = nrrd.read(filepath)
    affine = np.zeros((4, 4), dtype=np.float32)
    affine[:3, :3] = header['space directions']
    affine[:3, 3] = header['space origin']
    affine[3, 3] = 1.0
    return data, affine

def load_numpy(
    filepath: FilePath,
    keys: str | List[str] = 'data',
    ) -> np.ndarray | List[np.ndarray]:
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
