import numpy as np
import scipy
import torch

from ..defaults import get_dim
from ..typing import AffineMatrix, Box, BoxTensor, Image, LabelImage, Pixel, Pixels, Point, Points, SamplingGrid, Size, Spacing, SpatialDim, Voxel, Voxels
from .args import arg_default
from .conversion import to_numpy, to_tensor

# Don't work with tuple types, only handle numpy or tensors.
# These two are necessary - tuples aren't. Just convert tuples
# to numpy of tensors as needed.
def affine_origin(
    affine: AffineMatrix,
    ) -> Point:
    affine, return_type = to_tensor(affine, return_type=True)

    # Get origin.
    dim = affine.shape[0] - 1
    if dim == 2:
        origin = torch.tensor((affine[0, 2], affine[1, 2]), device=affine.device, dtype=torch.float32)
    else:
        origin = torch.tensor((affine[0, 3], affine[1, 3], affine[2, 3]), device=affine.device, dtype=torch.float32)

    if return_type is np.ndarray:
         origin = to_numpy(origin)
    return origin

def affine_spacing(
    affine: AffineMatrix,
    ) -> Spacing:
    affine, return_type = to_tensor(affine, return_type=True)

    # Get spacing.
    dim = affine.shape[0] - 1
    if dim == 2:
        spacing = torch.tensor((affine[0, 0], affine[1, 1]), device=affine.device, dtype=torch.float32)
    else:
        spacing = torch.tensor((affine[0, 0], affine[1, 1], affine[2, 2]), device=affine.device, dtype=torch.float32)

    if return_type is np.ndarray:
         spacing = to_numpy(spacing)
    return spacing

# Input: np.ndarray, or torch.Tensor.
# Output: np.ndarray, or torch.Tensor.
def centre_of_mass(
    data: Image,
    affine: AffineMatrix | None = None,
    ) -> Point | Pixel | Voxel | None:
    data, return_type, return_device = to_numpy(data, return_device=True, return_type=True)

    if data.sum() == 0:
        return None 

    # Compute the centre of mass.
    com = scipy.ndimage.center_of_mass(data)
    com = to_numpy(com)
    if affine is not None:
        com = to_world_coords(com, affine)

    if return_type is torch.Tensor:
        com = to_tensor(com, device=return_device, dtype=torch.float32)
    return com

def create_affine(
    spacing: Spacing | None = None,
    origin: Point | None = None,
    device: torch.device = torch.device('cpu'),
    dim: SpatialDim | None = None,
    ) -> AffineMatrix:
    # Resolve dim.
    def get_dim_local():
        if spacing is not None:
            return len(spacing)
        elif origin is not None:
            return len(origin)
        else:
            return get_dim()
    dim = arg_default(dim, dim, get_dim_local()) 

    # Use identify spacing/origin if not provided.
    return_type = torch.Tensor
    if spacing is not None:
        spacing, return_type = to_tensor(spacing, device=device, dtype=torch.float32, return_type=True)
    else:
        spacing = torch.ones(dim, device=device, dtype=torch.float32)
    if origin is not None:
        origin = to_tensor(origin, device=device, dtype=torch.float32)
    else:
        origin = torch.zeros(dim, device=device, dtype=torch.float32)

    # Create affine.
    affine = torch.eye(dim + 1, device=device, dtype=torch.float32)
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

    if return_type is np.ndarray:
        affine = to_numpy(affine)

    return affine

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
    fov_c = fov_d.sum(dim=0) / 2
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
    fov_w = max - min + 1

    if return_type is np.ndarray:
        fov_w = to_numpy(fov_w)

    return fov_w

# Returns the box defining the image FOV.
def fov(
    grid: SamplingGrid,
    ) -> BoxTensor:
    size, affine = grid
    size, return_type = to_tensor(size, return_type=True)

    # Get fov in voxels.
    dim = len(size)
    fov_vox = torch.stack([
        torch.zeros(dim, device=size.device, dtype=torch.int32),
        size - 1,
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

def fov_centre(
    grid: SamplingGrid,
    ) -> Point | Pixel | Voxel:
    size, affine = grid
    size, return_type = to_tensor(size, return_type=True)

    # Get FOV.
    fov_d = fov((size, affine))

    # Get FOV centre.
    fov_c = fov_d.sum(dim=0) / 2
    if affine is None:
        fov_c = torch.round(fov_c).type(torch.int32)

    if return_type is np.ndarray:
        fov_c = to_numpy(fov_c)

    return fov_c

# Returns the width of the FOV box in mm
def fov_width(
    grid: SamplingGrid,
    ) -> Size:
    size, affine = grid
    size, return_type = to_tensor(size, return_type=True)

    fov_d = fov((size, affine))
    
    # Get width.
    min, max = fov_d
    fov_w = max - min

    if return_type is np.ndarray:
        fov_w = to_numpy(fov_w)

    return fov_w

def spatial_size(
    image: Image,
    dim: SpatialDim
    ) -> Size:
    image, return_type = to_tensor(image, return_type=True)
    size = image.shape[-dim:]
    if return_type is np.ndarray:
        size = to_numpy(size)
    return size

# Input: np.ndarray, or torch.Tensor.
# Output: np.ndarray, or torch.Tensor.
def to_image_coords(
    point: Point | Points,
    affine: AffineMatrix,
    ) -> Pixel | Pixels | Voxel | Voxels:
    point, return_type = to_tensor(point, return_type=True)
    affine = to_tensor(affine, return_type=False)
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    point = torch.round((point - origin) / spacing).type(torch.int32)
    if return_type is np.ndarray:
        point = to_numpy(point)
    return point

def to_world_coords(
    point: Pixel | Pixels | Voxel | Voxels,
    affine: AffineMatrix,
    ) -> Point | Points:
    point, return_type = to_tensor(point, return_type=True)
    affine = to_tensor(affine, return_type=False)
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    point = (point * spacing + origin).type(torch.float32)
    if return_type is np.ndarray:
        point = to_numpy(point)
    return point
