import numpy as np
import scipy
import torch

from ..typing import AffineMatrix, AffineMatrixTensor, Box, BoxTensor, Image, LabelImage, Pixel, PixelsTensor, PixelTensor, Point, PointsTensor, PointTensor, Size, Spacing, SpatialDim, Voxel, VoxelsTensor, VoxelTensor
from .conversion import to_numpy, to_tensor

def affine_origin(
    affine: AffineMatrix,
    ) -> Point:
    # Get origin.
    dim = affine.shape[0] - 1
    if dim == 2:
        origin = (affine[0, 2], affine[1, 2])
    else:
        origin = (affine[0, 3], affine[1, 3], affine[2, 3])

    return origin

def affine_spacing(
    affine: AffineMatrix,
    ) -> Spacing:
    # Get spacing.
    dim = affine.shape[0] - 1
    if dim == 2:
        spacing = (affine[0, 0], affine[1, 1])
    else:
        spacing = (affine[0, 0], affine[1, 1], affine[2, 2])

    return spacing

def centre_of_mass(
    data: Image,
    affine: AffineMatrix | None = None,
    ) -> Point | Pixel | Voxel:
    if data.sum() == 0:
        return None 

    data, return_type, return_device = to_numpy(data, return_device=True, return_type=True)

    # Compute the centre of mass.
    com = scipy.ndimage.center_of_mass(data)
    if affine is not None:
        com = to_world_coords(com, affine)

    if return_type is torch.Tensor:
        com = to_tensor(com, device=return_device, dtype=torch.float32)

    return com

def create_affine(
    spacing: Spacing,
    origin: Point,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
    dim = len(spacing)
    affine = create_eye(dim, device=device, dtype=dtype)
    if dim == 2:
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1]
        affine[0, 2] = origin[0]
        affine[1, 2] = origin[1]
    else:
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1]
        affine[2, 2] = spacing[2]
        affine[0, 3] = origin[0]
        affine[1, 3] = origin[1]
        affine[2, 3] = origin[2]
    return affine

def create_eye(
    dim: SpatialDim,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
    return torch.eye(dim + 1, device=device, dtype=dtype)

def foreground_fov(
    data: LabelImage,
    affine: AffineMatrix | None = None,
    ) -> Box | None:
    data, return_type = to_tensor(data, return_type=True)
    if data.sum() == 0:
        return None

    # Get fov of foreground objects.
    non_zero = torch.argwhere(data != 0).type(torch.int)
    fov_vox = torch.stack([
        non_zero.min(dim=0).values,
        non_zero.max(dim=0).values,
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
    affine: AffineMatrix | None = None,
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
    affine: AffineMatrix | None = None,
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
    affine: AffineMatrix | None = None,
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
    affine: AffineMatrix | None = None,
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

def to_image_coords(
    point: PointTensor | PointsTensor,
    affine: AffineMatrixTensor,
    ) -> PixelTensor | PixelsTensor | VoxelTensor | VoxelsTensor:
    point, return_type = to_tensor(point, return_type=True)
    affine = to_tensor(affine, return_type=False)
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    point = torch.round((point - origin) / spacing).type(torch.int32)
    if return_type is np.ndarray:
        point = to_numpy(point)
    return point

def to_world_coords(
    point: PixelTensor | PixelsTensor | VoxelTensor | VoxelsTensor,
    affine: AffineMatrixTensor,
    ) -> PointTensor | PointsTensor:
    point, return_type = to_tensor(point, return_type=True)
    affine = to_tensor(affine, return_type=False)
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    point = (point * spacing + origin).type(torch.float32)
    if return_type is np.ndarray:
        point = to_numpy(point)
    return point
