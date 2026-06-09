from jaxtyping import Bool, Float32, Int
import numpy as np
import torch
from typing import Dict, List, Literal, NamedTuple, Tuple

# First-order types (composed of basic types).
# Splitting by 'order' allows for easier managing of type dependencies.
AffineMatrix2D = Float32[np.ndarray | torch.Tensor, "3 3"]
AffineMatrix2DTensor = Float32[torch.Tensor, "3 3"]
AffineMatrix3D = Float32[np.ndarray | torch.Tensor, "4 4"]
AffineMatrix3DArray = Float32[np.ndarray, "4 4"]
AffineMatrix3DTensor = Float32[torch.Tensor, "4 4"]
BatchImage2D = Float32[np.ndarray | torch.Tensor, "B X Y"]
BatchImage2DTensor = Float32[torch.Tensor, "B X Y"]
BatchImage3D = Float32[np.ndarray | torch.Tensor, "B X Y Z"]
BatchImage3DTensor = Float32[torch.Tensor, "B X Y Z"]
BatchChannelImage2D = Float32[np.ndarray | torch.Tensor, "B C X Y"]
BatchChannelImage2DTensor = Float32[torch.Tensor, "B C X Y"]
BatchChannelImage3D = Float32[np.ndarray | torch.Tensor, "B C X Y Z"]
BatchChannelImage3DTensor = Float32[torch.Tensor, "B C X Y Z"]
BatchChannelLabelImage2D = Bool[np.ndarray | torch.Tensor, "B C X Y"]
BatchChannelLabelImage2DTensor = Bool[torch.Tensor, "B C X Y"]
BatchChannelLabelImage3D = Bool[np.ndarray | torch.Tensor, "B C X Y Z"]
BatchChannelLabelImage3DTensor = Bool[torch.Tensor, "B C X Y Z"]
BatchPoints2D = Float32[np.ndarray | torch.Tensor, "B N 2"]
BatchPoints2DTensor = Float32[torch.Tensor, "B N 2"]
BatchPoints3D = Float32[np.ndarray | torch.Tensor, "B N 3"]
BatchPoints3DTensor = Float32[torch.Tensor, "B N 3"]
Box2D = Float32[np.ndarray | torch.Tensor, "2 2"]
Box2DTensor = Float32[torch.Tensor, "2 2"]
Box3D = Float32[np.ndarray | torch.Tensor, "2 3"]
Box3DTensor = Float32[torch.Tensor, "2 3"]
BatchBox2D = Float32[np.ndarray | torch.Tensor, "B 2 2"]
BatchBox3D = Float32[np.ndarray | torch.Tensor, "B 2 3"]
BatchLabelImage2D = Bool[np.ndarray | torch.Tensor, "B X Y"]
BatchLabelImage2DTensor = Bool[torch.Tensor, "B X Y"]
BatchLabelImage3D = Bool[np.ndarray | torch.Tensor, "B X Y Z"]
BatchLabelImage3DArray = Bool[np.ndarray, "B X Y Z"]
BatchLabelImage3DTensor = Bool[torch.Tensor, "B X Y Z"]
ChannelImage2D = Float32[np.ndarray | torch.Tensor, "C X Y"]
ChannelImage2DTensor = Float32[torch.Tensor, "C X Y"]
ChannelImage3D = Float32[np.ndarray | torch.Tensor, "C X Y Z"]
ChannelImage3DTensor = Float32[torch.Tensor, "C X Y Z"]
ChannelLabelImage2D = Bool[np.ndarray | torch.Tensor, "C X Y"]
ChannelLabelImage2DTensor = Bool[torch.Tensor, "C X Y"]
ChannelLabelImage3D = Bool[np.ndarray | torch.Tensor, "C X Y Z"]
ChannelLabelImage3DTensor = Bool[torch.Tensor, "C X Y Z"]
SpatialDim = Literal[2, 3]
FilePath = str
Image2D = Float32[np.ndarray | torch.Tensor, "X Y"]
Image2DArray = Float32[np.ndarray, "X Y"]
Image2DTensor = Float32[torch.Tensor, "X Y"]
Image3D = Float32[np.ndarray | torch.Tensor, "X Y Z"]
Image3DArray = Float32[np.ndarray, "X Y Z"]
Image3DTensor = Float32[torch.Tensor, "X Y Z"]
Indices = int | List[int] | Int[np.ndarray | torch.Tensor, "N"]
IndicesTensor = Int[torch.Tensor, "N"]
LabelImage2D = Bool[np.ndarray | torch.Tensor, "X Y"]
LabelImage2DTensor = Bool[torch.Tensor, "X Y"]
LabelImage3D = Bool[np.ndarray | torch.Tensor, "X Y Z"]
LabelImage3DArray = Bool[np.ndarray, "X Y Z"]
LabelImage3DTensor = Bool[torch.Tensor, "X Y Z"]
Number = int | float
Orientation2D = Literal['LI', 'LS', 'RI', 'RS']
Orientation3D = Literal['LAI', 'LAS', 'LPI', 'LPS', 'RAI', 'RAS', 'RPI', 'RPS']
Point2D = Tuple[float, float] | Float32[np.ndarray | torch.Tensor, "2"]
Point2DTensor = Float32[torch.Tensor, "2"]
Point3D = Tuple[float, float, float] | Float32[np.ndarray | torch.Tensor, "3"]
Point3DTensor = Float32[torch.Tensor, "3"]
Points2D = Float32[np.ndarray | torch.Tensor, "N 2"]
Points2DTensor = Float32[torch.Tensor, "N 2"]
Points3D = Float32[np.ndarray | torch.Tensor, "N 3"]
Points3DArray = Float32[np.ndarray, "N 3"]
Points3DTensor = Float32[torch.Tensor, "N 3"]
Pixel = Tuple[int, int] | Int[np.ndarray | torch.Tensor, "2"]
PixelTensor = Int[torch.Tensor, "2"]
Pixels = Int[np.ndarray | torch.Tensor, "N 2"]
PixelsTensor = Int[torch.Tensor, "N 2"]
Voxel = Tuple[int, int, int] | Int[np.ndarray | torch.Tensor, "3"]
VoxelTensor = Int[torch.Tensor, "3"]
Voxels = Int[np.ndarray | torch.Tensor, "N 3"]
VoxelsTensor = Int[torch.Tensor, "N 3"]
Range2PerDim = Number | Tuple[Number, Number] | Tuple[Number, ...] | Float32[np.ndarray | torch.Tensor, "2 | 2 * <dim>"]
Range4PerDim = Number | Tuple[Number, Number, Number, Number] | Tuple[Number, ...] | Float32[np.ndarray | torch.Tensor, "4 | 4 * <dim>"]
Size2D = Tuple[int, int] | Int[np.ndarray | torch.Tensor, "2"]
Size2DTensor = Int[torch.Tensor, "2"]
Size3D = Tuple[int, int, int] | Int[np.ndarray | torch.Tensor, "3"]
Size3DArray = Int[np.ndarray, "3"]
Size3DTensor = Int[torch.Tensor, "3"]
Spacing2D = Tuple[float, float] | Float32[np.ndarray | torch.Tensor, "2"]
Spacing2DTensor = Float32[torch.Tensor, "2"]
Spacing3D = Tuple[float, float, float] | Float32[np.ndarray | torch.Tensor, "3"]
Spacing3DTensor = Float32[torch.Tensor, "3"]
# Had to use Literal['TransformParams'] for recursive type definition.
TransformParams = Dict[int | str, int | str | float | np.ndarray | torch.Tensor | Literal['TransformParams']]
View = Literal[0, 1, 2]

# Second-order types (composed of first-order types).
AffineMatrix = AffineMatrix2D | AffineMatrix3D
AffineMatrixTensor = AffineMatrix2DTensor | AffineMatrix3DTensor
BatchChannelImage = BatchChannelImage2D | BatchChannelImage3D
BatchChannelImageTensor = BatchChannelImage2DTensor | BatchChannelImage3DTensor
BatchChannelLabelImage = BatchChannelLabelImage2D | BatchChannelLabelImage3D
BatchChannelLabelImageTensor = BatchChannelLabelImage2DTensor | BatchChannelLabelImage3DTensor
BatchImage = BatchImage2D | BatchImage3D
BatchImageTensor = BatchImage2DTensor | BatchImage3DTensor
BatchLabelImage = BatchLabelImage2D | BatchLabelImage3D
BatchLabelImageTensor = BatchLabelImage2DTensor | BatchLabelImage3DTensor
BatchPoints = BatchPoints2D | BatchPoints3D
BatchPointsTensor = BatchPoints2DTensor | BatchPoints3DTensor
Box = Box2D | Box3D
BoxTensor = Box2DTensor | Box3DTensor
BatchBox = BatchBox2D | BatchBox3D
ChannelImage = ChannelImage2D | ChannelImage3D
ChannelImageTensor = ChannelImage2DTensor | ChannelImage3DTensor
Image = Image2D | Image3D
ImageTensor = Image2DTensor | Image3DTensor
LabelImage = LabelImage2D | LabelImage3D
LabelImageTensor = LabelImage2DTensor | LabelImage3DTensor
Orientation = Orientation2D | Orientation3D
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
    affine: AffineMatrix | None = None

class SamplingGridTensor(NamedTuple):
    size: SizeTensor
    affine: AffineMatrixTensor | None = None
