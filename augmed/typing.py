from jaxtyping import Bool, Float, Int
import numpy as np
import torch 
from typing import Dict, Literal, NamedTuple, Optional, Tuple

# First-order types (composed of basic types).
# Splitting by 'order' allows for easier managing of type dependencies.
Affine2D = Float[np.ndarray | torch.Tensor, "3 3"]
Affine2DTensor = Float[torch.Tensor, "3 3"]
Affine3D = Float[np.ndarray | torch.Tensor, "4 4"]
Affine3DArray = Float[np.ndarray, "4 4"]
Affine3DTensor = Float[torch.Tensor, "4 4"]
BatchImage2D = Float[np.ndarray | torch.Tensor, "B X Y"]
BatchImage2DTensor = Float[torch.Tensor, "B X Y"]
BatchImage3D = Float[np.ndarray | torch.Tensor, "B X Y Z"]
BatchImage3DTensor = Float[torch.Tensor, "B X Y Z"]
BatchChannelImage2D = Float[np.ndarray | torch.Tensor, "B C X Y"]
BatchChannelImage2DTensor = Float[torch.Tensor, "B C X Y"]
BatchChannelImage3D = Float[np.ndarray | torch.Tensor, "B C X Y Z"]
BatchChannelImage3DTensor = Float[torch.Tensor, "B C X Y Z"]
BatchChannelLabelImage2D = Bool[np.ndarray | torch.Tensor, "B C X Y"]
BatchChannelLabelImage2DTensor = Bool[torch.Tensor, "B C X Y"]
BatchChannelLabelImage3D = Bool[np.ndarray | torch.Tensor, "B C X Y Z"]
BatchChannelLabelImage3DTensor = Bool[torch.Tensor, "B C X Y Z"]
Box2D = Float[np.ndarray | torch.Tensor, "2 2"]
Box2DTensor = Float[torch.Tensor, "2 2"]
Box3D = Float[np.ndarray | torch.Tensor, "2 3"]
Box3DTensor = Float[torch.Tensor, "2 3"]
BatchLabelImage2D = Bool[np.ndarray | torch.Tensor, "B X Y"]
BatchLabelImage2DTensor = Bool[torch.Tensor, "B X Y"]
BatchLabelImage3D = Bool[np.ndarray | torch.Tensor, "B X Y Z"]
BatchLabelImage3DArray = Bool[np.ndarray, "B X Y Z"]
BatchLabelImage3DTensor = Bool[torch.Tensor, "B X Y Z"]
ChannelImage2D = Float[np.ndarray | torch.Tensor, "C X Y"]
ChannelImage2DTensor = Float[torch.Tensor, "C X Y"]
ChannelImage3D = Float[np.ndarray | torch.Tensor, "C X Y Z"]
ChannelImage3DTensor = Float[torch.Tensor, "C X Y Z"]
ChannelLabelImage2D = Bool[np.ndarray | torch.Tensor, "C X Y"]
ChannelLabelImage2DTensor = Bool[torch.Tensor, "C X Y"]
ChannelLabelImage3D = Bool[np.ndarray | torch.Tensor, "C X Y Z"]
ChannelLabelImage3DTensor = Bool[torch.Tensor, "C X Y Z"]
SpatialDim = Literal[2, 3]
FilePath = str
Image2D = Float[np.ndarray | torch.Tensor, "X Y"]
Image2DTensor = Float[torch.Tensor, "X Y"]
Image3D = Float[np.ndarray | torch.Tensor, "X Y Z"]
Image3DTensor = Float[torch.Tensor, "X Y Z"]
LabelImage2D = Bool[np.ndarray | torch.Tensor, "X Y"]
LabelImage2DTensor = Bool[torch.Tensor, "X Y"]
LabelImage3D = Bool[np.ndarray | torch.Tensor, "X Y Z"]
LabelImage3DArray = Bool[np.ndarray, "X Y Z"]
LabelImage3DTensor = Bool[torch.Tensor, "X Y Z"]
Number = int | float
Orientation = Literal['LAI', 'LAS', 'LPI', 'LPS', 'RAI', 'RAS', 'RPI', 'RPS']
Point2D = Tuple[float, float] | Float[np.ndarray | torch.Tensor, "2"]
Point2DTensor = Float[torch.Tensor, "2"]
Point3D = Tuple[float, float, float] | Float[np.ndarray | torch.Tensor, "3"]
Point3DTensor = Float[torch.Tensor, "3"]
Points2D = Float[np.ndarray | torch.Tensor, "N 2"]
Points2DTensor = Float[torch.Tensor, "N 2"]
Points3D = Float[np.ndarray | torch.Tensor, "N 3"]
Points3DArray = Float[np.ndarray, "N 3"]
Points3DTensor = Float[torch.Tensor, "N 3"]
Pixel = Tuple[int, int] | Int[np.ndarray | torch.Tensor, "2"]
PixelTensor = Int[torch.Tensor, "2"]
Voxel = Tuple[int, int, int] | Int[np.ndarray | torch.Tensor, "3"]
VoxelTensor = Int[torch.Tensor, "3"]
Size2D = Tuple[int, int] | Int[np.ndarray | torch.Tensor, "2"]
Size2DTensor = Int[torch.Tensor, "2"]
Size3D = Tuple[int, int, int] | Int[np.ndarray | torch.Tensor, "3"]
Size3DArray = Int[np.ndarray, "3"]
Size3DTensor = Int[torch.Tensor, "3"]
Spacing2D = Tuple[float, float] | Float[np.ndarray | torch.Tensor, "2"]
Spacing2DTensor = Float[torch.Tensor, "2"]
Spacing3D = Tuple[float, float, float] | Float[np.ndarray | torch.Tensor, "3"]
Spacing3DTensor = Float[torch.Tensor, "3"]
# Had to use Literal['TransformParams'] for recursive type definition.
TransformParams = Dict[int | str, int | str | float | np.ndarray | torch.Tensor | Literal['TransformParams']]

# Second-order types (composed of first-order types).
Affine = Affine2D | Affine3D
AffineTensor = Affine2DTensor | Affine3DTensor
BatchChannelImage = BatchChannelImage2D | BatchChannelImage3D
BatchChannelImageTensor = BatchChannelImage2DTensor | BatchChannelImage3DTensor
BatchChannelLabelImage = BatchChannelLabelImage2D | BatchChannelLabelImage3D
BatchChannelLabelImageTensor = BatchChannelLabelImage2DTensor | BatchChannelLabelImage3DTensor
BatchImage = BatchImage2D | BatchImage3D
BatchImageTensor = BatchImage2DTensor | BatchImage3DTensor
BatchLabelImage = BatchLabelImage2D | BatchLabelImage3D
BatchLabelImageTensor = BatchLabelImage2DTensor | BatchLabelImage3DTensor
Box = Box2D | Box3D
BoxTensor = Box2DTensor | Box3DTensor
ChannelImage = ChannelImage2D | ChannelImage3D
ChannelImageTensor = ChannelImage2DTensor | ChannelImage3DTensor
Image = Image2D | Image3D
ImageTensor = Image2DTensor | Image3DTensor
LabelImage = LabelImage2D | LabelImage3D
LabelImageTensor = LabelImage2DTensor | LabelImage3DTensor
Point = Point2D | Point3D
PointTensor = Point2DTensor | Point3DTensor
Points = Points2D | Points3D
PointsTensor = Points2DTensor | Points3DTensor
Size = Size2D | Size3D
SizeTensor = Size2DTensor | Size3DTensor
Spacing = Spacing2D | Spacing3D 
SpacingTensor = Spacing2DTensor | Spacing3DTensor

# Third-order types (you get it).
class SamplingGrid(NamedTuple):
    size: Size
    affine: Optional[Affine] = None

class SamplingGridTensor(NamedTuple):
    size: SizeTensor
    affine: Optional[AffineTensor] = None
