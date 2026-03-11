import torch
from typing import *

from ..typing import *
from .args import alias_kwargs
from .matrix import affine_spacing, affine_origin

def foreground_fov(
    data: LabelImageTensor,
    affine: AffineTensor | None = None,
    ) -> BoxTensor | None:
    if data.sum() == 0:
        return None

    # Get fov of foreground objects.
    non_zero = torch.argwhere(data != 0).type(torch.int)
    fov_vox = torch.stack([
        non_zero.min(dim=0),
        non_zero.max(dim=0)
    ])
    if affine is None:
        return fov_vox

    # Get fov in mm.
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    fov_mm = fov_vox * spacing + origin
    return fov_mm

def foreground_fov_centre(
    data: LabelImageTensor,
    affine: AffineTensor | None = None,
    **kwargs,
    ) -> PointTensor | PixelTensor | VoxelTensor | None:
    fov_d = foreground_fov(data, affine=affine, **kwargs)
    if fov_d is None:
        return None
    fov_c = torch.tensor(fov_d).sum(dim=0) / 2
    if affine is None:
        fov_c = torch.round(fov_c).type(torch.int32)
    return fov_c

def foreground_fov_width(
    data: LabelImageTensor,
    **kwargs,
    ) -> SizeTensor | None:
    # Get foreground fov.
    fov_fg = foreground_fov(data, **kwargs)
    if fov_fg is None:
        return None
    min, max = fov_fg
    fov_w = max - min
    return fov_w

def fov(
    size: SizeTensor,
    affine: AffineTensor | None = None,
    raise_error: bool = True,
    ) -> BoxTensor:
    # Get fov in voxels.
    n_dims = len(size)
    fov_vox = torch.stack([
        torch.zeros(n_dims, dtype=torch.int32, device=size.device),
        torch.tensor(size),
    ])
    if affine is None:
        return fov_vox

    # Get fov in mm.
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    fov_mm = fov_vox * spacing + origin
    return fov_mm

def fov_centre(
    size: SizeTensor,
    affine: AffineTensor | None = None,
    **kwargs,
    ) -> PointTensor | PixelTensor | VoxelTensor:
    # Get FOV.
    fov_d = fov(size, affine=affine, **kwargs)

    # Get FOV centre.
    fov_c = fov_d.sum(dim=0) / 2
    if affine is None:
        fov_c = torch.round(fov_c).type(torch.int32)
    return fov_c

def fov_width(
    data: LabelImageTensor,
    **kwargs,
    ) -> SizeTensor:
    fov_d = fov(data, **kwargs)
    
    # Get width.
    min, max = fov_d
    fov_w = max - min
    return fov_w
