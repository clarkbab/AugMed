import torch
import torch
from typing import Literal

from ..typing import AffineMatrixTensor, ImageTensor, Number, PointsTensor, SizeTensor, SpatialDim
from ..utils.conversion import to_tensor
from ..utils.geometry import affine_origin, affine_spacing

# 'grid_sample' can be used for interpolating at points (Nx3) or on image grids (3xXxYxZ).
# We don't need to know the spatial coordinates of the resampling grid for the interpolation,
# this information can be added back in after resampling to create the moved image.
# For 'grid_sample' we just need to know the coordinates of each sample (points, mm) in the moving
# image (image) and the coordinates of the moving image grid (affine).
def grid_points(
    size: SizeTensor,
    affine: AffineMatrixTensor | None = None,
    return_superset: bool = False,
    ) -> PointsTensor:
    # Get grid points.
    dim = len(size)
    grids = torch.meshgrid([torch.arange(s.item()) for s in size], indexing='ij')
    points = torch.stack(grids, dim=-1).reshape(-1, dim).to(size.device)
    # Convert from voxel to world coords if affine is provided.
    if affine is not None:
        origin = affine_origin(affine)
        spacing = affine_spacing(affine)
        points = points * spacing + origin

    # Create superset.
    if return_superset:
        # Move points to device for concatenation.
        # Which device should this go on? Use first GPU if available, because
        # we're going to have to calculate 'backward_transform_points' on these points.
        device_types = [d.type for d in devices]
        super_device = devices[device_types.index('cuda')] if 'cuda' in device_types else devices[0]
        points = [p.to(super_device) for p in pointses]

        # Get superset of points.
        # While it might seem like a good idea to create a superset of points to 
        # reduce transform processing, in practice the act of getting unique points
        # takes much longer than just transforming them all.
        # !!! This 'unique' op, over millions of points, takes a looooong time.
        # This kind of makes the superset idea non-viable.
        # Apparently it sorts the array first, which might be the slow part.
        # Our stacked array is not sorted by default.
        super_points = torch.vstack(points).unique(dim=0)

        # For each image, get the indices of it's points within the superset.
        # This is required for creating subsets later on after 'backward_transform_points'.
        indices = []
        for p, d in zip(pointses, devices):
            matches = (p[:, None, :].to(super_device) == super_points[None, :, :])
            matches = matches.all(dim=-1)
            index = matches.float().argmax(dim=1)
            index = index.to(d)
            indices.append(index)

        return super_points, indices

    return points

def grid_sample(
    image: ImageTensor,
    affine: AffineMatrixTensor,
    points: PointsTensor | ImageTensor,
    mode: Literal['bicubic', 'bilinear', 'nearest'] = 'bilinear',
    padding: Number | Literal['border', 'max', 'min', 'reflection', 'zeros'] = 'min',
    dim: SpatialDim = 3,
    **kwargs,
    ) -> ImageTensor | PointsTensor:
    if points.shape[-1] == 2 or points.shape[-1] == 3:
        points_type = 'points'
    else:
        points_type = 'image'

    # We use 'float32' for resample points and maintain the original dtype of 'image'.
    points = to_tensor(points, device=image.device, dtype=torch.float32)
    size = to_tensor(image.shape[-dim:], device=image.device, dtype=torch.float32)
    affine = to_tensor(affine, device=image.device, dtype=torch.float32)
    origin = affine_origin(affine)
    spacing = affine_spacing(affine)

    # Normalise to range [-1, 1] expected by 'torch.grid_sample'.
    if points_type == 'image':
        points = points.moveaxis(0, -1)     # Move channels to end - expected by 'torch.grid_sample'.
    points = 2 * (points - origin) / ((size - 1) * spacing) - 1      

    # Add image channels expected by 'torch.grid_sample'.
    image_dims_to_add = dim + 2 - len(image.shape)
    image = image.reshape(*(1,) * image_dims_to_add, *image.shape) if image_dims_to_add > 0 else image

    # Add points channels expected by 'torch.grid_sample'.
    points_dims_to_add = dim + 2 - len(points.shape)
    points = points.reshape(*(1,) * points_dims_to_add, *points.shape)

    # Transpose image spatial axes as expected by 'torch.grid_sample'.
    image_src_dims = list(range(-dim, 0))    # Image should have channels first anyway.
    image_dest_dims = list(reversed(image_src_dims))
    image = torch.moveaxis(image, image_src_dims, image_dest_dims)

    # Convert bool types to float as required.
    return_dtype = image.dtype
    if return_dtype is not torch.float32:
        image = image.type(torch.float32)

    # Convert padding to float.
    if isinstance(padding, str):
        if padding == 'min':
            padding = float(image.min())
        elif padding == 'max':
            padding = float(image.max())
        else:
            padding_mode = padding      # Pass values such as 'border' directly to 'grid_sample'.

    # For number padding, translate intensities as 'grid_sample' only provides zero-padding.
    if isinstance(padding, (int, float)):
        image = image - padding
        padding_mode = 'zeros'

    # Determine interpolation mode.
    mode = 'nearest' if return_dtype is torch.bool else mode

    # Resample image.
    image_t = torch.nn.functional.grid_sample(image, points, align_corners=True, mode=mode, padding_mode=padding_mode, **kwargs)

    # Convert to return format.
    if return_dtype is not torch.float32:
        image_t = image_t.type(return_dtype)

    # Reverse intensity translation for padding.
    if isinstance(padding, (int, float)):
        image_t = image_t + padding

    # Remove channels that were added for 'grid_sample'.
    image_t = image_t.squeeze(axis=tuple(range(image_dims_to_add))) if image_dims_to_add > 0 else image_t

    return image_t
