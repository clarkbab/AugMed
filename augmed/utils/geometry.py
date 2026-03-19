import torch
from typing import *

from ..typing import *
from .args import alias_kwargs
from .conversion import to_numpy, to_tensor
from .matrix import affine_spacing, affine_origin

def foreground_fov(
    data: LabelImage,
    affine: Affine | None = None,
    ) -> Box | None:
    data, return_type = to_tensor(data, return_type=True)
    if data.sum() == 0:
        return None

    # Get fov of foreground objects.
    non_zero = torch.argwhere(data != 0).type(torch.int)
    fov_vox = torch.stack([
        non_zero.min(dim=0),
        non_zero.max(dim=0)
    ])
    if affine is None:
        if return_type is np.ndarray:
            fov_vox = to_numpy(fov_vox)
        return fov_vox

    # Get fov in mm.
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    fov_mm = fov_vox * spacing + origin

    if return_type is np.ndarray:
         fov_mm = to_numpy(fov_mm)

    return fov_mm

def foreground_fov_centre(
    data: LabelImage,
    affine: Affine | None = None,
    **kwargs,
    ) -> Point | Pixel | Voxel | None:
    data, return_type = to_tensor(data, return_type=True)

    fov_d = foreground_fov(data, affine=affine, **kwargs)
    if fov_d is None:
        return None
    fov_c = torch.tensor(fov_d).sum(dim=0) / 2
    if affine is None:
        fov_c = torch.round(fov_c).type(torch.int32)

    if return_type is np.ndarray:
        fov_c = to_numpy(fov_c)
        
    return fov_c

def foreground_fov_width(
    data: LabelImage,
    **kwargs,
    ) -> Size | None:
    data, return_type = to_tensor(data, return_type=True)

    # Get foreground fov.
    fov_fg = foreground_fov(data, **kwargs)
    if fov_fg is None:
        return None
    min, max = fov_fg
    fov_w = max - min

    if return_type is np.ndarray:
        fov_w = to_numpy(fov_w)

    return fov_w

def fov(
    size: Size,
    affine: Affine | None = None,
    ) -> BoxTensor:
    size, return_type = to_tensor(size, return_type=True)

    # Get fov in voxels.
    n_dims = len(size)
    fov_vox = torch.stack([
        torch.zeros(n_dims, device=size.device, dtype=torch.int32),
        torch.tensor(size),
    ])
    if affine is None:
        return fov_vox

    # Get fov in mm.
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    fov_mm = fov_vox * spacing + origin

    if return_type is np.ndarray:
        fov_mm = to_numpy(fov_mm)

    return fov_mm

def fov_centre(
    size: Size,
    affine: Affine | None = None,
    **kwargs,
    ) -> Point | Pixel | Voxel:
    size, return_type = to_tensor(size, return_type=True)

    # Get FOV.
    fov_d = fov(size, affine=affine, **kwargs)

    # Get FOV centre.
    fov_c = fov_d.sum(dim=0) / 2
    if affine is None:
        fov_c = torch.round(fov_c).type(torch.int32)

    if return_type is np.ndarray:
        fov_c = to_numpy(fov_c)

    return fov_c

def fov_width(
    size: Size,
    affine: Affine | None = None,
    **kwargs,
    ) -> Size:
    size, return_type = to_tensor(size, return_type=True)

    fov_d = fov(size, affine=affine, **kwargs)
    
    # Get width.
    min, max = fov_d
    fov_w = max - min

    if return_type is np.ndarray:
        fov_w = to_numpy(fov_w)

    return fov_w
