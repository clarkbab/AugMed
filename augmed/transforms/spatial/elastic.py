from typing import *

from ...typing import *
from ...utils.args import expand_range_arg
from ...utils.conversion import to_numpy, to_tensor, to_tuple
from ...utils.grid import grid_sample
from ...utils.logging import logger
from ...utils.matrix import affine_origin, affine_spacing, create_affine
from ..identity import Identity
from .spatial import RandomSpatialTransform, SpatialTransform

# Defines a coarse grid of control points.
# Random displacements are assigned at each control point.
class Elastic(SpatialTransform):
    def __init__(
        self,
        control_spacing: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 50.0,
        control_origin: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 20.0,
        displacement: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 20.0,
        method: Literal['bspline', 'cubic', 'linear'] = 'bspline',
        seed: int = 42,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        assert method in ['bspline', 'cubic', 'linear'], "Only 'bspline', 'cubic', and 'linear' elastic methods are supported."
        self.__method = method
        self.__control_spacing = to_tensor(control_spacing, broadcast=self._dim)
        assert len(self.__control_spacing) == self._dim, f"Expected 'control_spacing' of length '{self._dim}' for dim={self._dim}, got {len(self.__control_spacing)}."
        self.__control_origin = to_tensor(control_origin, broadcast=self._dim)
        assert len(self.__control_origin) == self._dim, f"Expected 'control_origin' of length '{self._dim}' for dim={self._dim}, got {len(self.__control_origin)}."
        # Disps aren't known until presented with the image.
        disp_range = expand_range_arg(displacement, dim=self._dim, negate_lower=True)
        assert len(disp_range) == 2 * self._dim, f"Expected 'displacement' of length {2 * self._dim}, got {len(disp_range)}."
        self.__disp_range = to_tensor(disp_range).reshape(self._dim, 2)
        self.__seed = seed
        self.__warn_folding()
        self._params = dict(
            type=self.__class__.__name__,
            control_origin=self.__control_origin,
            control_spacing=self.__control_spacing,
            dim=self._dim,
            displacement=self.__disp_range,
            method=self.__method,
            seed=self.__seed,
        )

    def __warn_folding(self, control_spacing: torch.Tensor | None = None) -> None:
        if control_spacing is None:
            control_spacing = self.__control_spacing
        disp_widths = self.__disp_range[:, 1] - self.__disp_range[:, 0]
        if (disp_widths >= control_spacing).any():
            logger.warning(f"Elastic transforms with displacement widths ({to_tuple(disp_widths)}) >= "
                f"control spacings ({to_tuple(control_spacing)}) may produce folding transforms. Such transforms may "
                f"be non-invertible and could raise errors when performing forward points transform.")

    def backward_transform_points(
        self,
        points: PointsTensor,
        **kwargs,
        ) -> PointsTensor:
        cp_disps, cp_affine = self.control_grid(points)
        cp_spacing = affine_spacing(cp_affine)
        cp_origin = affine_origin(cp_affine)
        cp_disps = torch.moveaxis(cp_disps, 0, -1)  # Move channels dim to back.

        # Normalise points to the control grid integer coords.
        points_norm = (points - cp_origin) / cp_spacing

        # Floor index: leftmost stencil point for linear, second stencil point for cubic.
        floor_idx = torch.stack([torch.searchsorted(torch.arange(cp_disps.shape[a]).to(points.device), points_norm[:, a]) - 1 for a in range(self._dim)], dim=-1)

        # Fractional position within the cell [0, 1).
        t = points_norm - floor_idx

        # Compute basis weights and stencil offsets per method.
        if self.__method == 'linear':
            # Linear basis (C0, interpolating). 2 weights, offsets {0, 1}.
            b = torch.stack([1 - t, t], dim=-2)
            stencil_range = torch.tensor([0, 1])
        else:
            # Cubic methods. 4 weights, offsets {-1, 0, 1, 2}.
            t2 = t * t
            t3 = t2 * t
            if self.__method == 'cubic':
                # Catmull-Rom basis (C1, interpolating).
                w0 = -0.5 * t3 + t2 - 0.5 * t
                w1 =  1.5 * t3 - 2.5 * t2 + 1.0
                w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
                w3 =  0.5 * t3 - 0.5 * t2
            else:
                # Cubic B-spline basis (C2, approximating).
                u = 1.0 - t
                w0 = u * u * u / 6.0
                w1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0
                w2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0
                w3 = t3 / 6.0
            b = torch.stack([w0, w1, w2, w3], dim=-2)
            stencil_range = torch.tensor([-1, 0, 1, 2])

        n = len(stencil_range)

        # Build the n^dim stencil offsets.
        offsets = torch.stack(torch.meshgrid([stencil_range] * self._dim, indexing='ij'), dim=-1)
        offsets = offsets.reshape(-1, self._dim).to(points.device)

        # Gather stencil indices for each point.
        corners = floor_idx[:, None, :] + offsets[None, :, :]

        # Index into the displacement grid.
        idxs = corners.unbind(-1)
        corner_disps = cp_disps[*idxs]

        # Reshape to (N, n, n, [n,] dim) for einsum.
        V = corner_disps.reshape(-1, *(n, ) * self._dim, self._dim)

        # Tensor-product interpolation via einsum.
        if self._dim == 2:
            disps = torch.einsum('ni,nj,nijd->nd', b[:, :, 0], b[:, :, 1], V)
        elif self._dim == 3:
            disps = torch.einsum('ni,nj,nk,nijkd->nd', b[:, :, 0], b[:, :, 1], b[:, :, 2], V)

        # Get displaced input points.
        points_t = points + disps

        return points_t

    def __backward_transform_points_linear_gs(
        self,
        points: PointsTensor,
        ) -> PointsTensor:
        # Get control grid.
        cp_disps, cp_affine = self.control_grid(points)

        # Interpolate the displacement grid.
        disps = grid_sample(cp_disps, cp_affine, points, dim=self._dim)
        disps = torch.moveaxis(disps, 0, -1)[0, 0]    # Disps come back from 'grid_sample' as 3-channel image.

        # Get displaced input points.
        points_t = points + disps

        return points_t
        
    # Vectorised spatial hash over world-space control point coordinates.
    # Reinterprets float32 coords as int32 bit patterns for hashing — deterministic
    # as long as the same arithmetic path produces the coordinates.
    # Returns draws in [0, 1) of shape (N, dim).
    def __control_grid_draws(
        self,
        points: PointsTensor,
        ) -> PointsTensor:
        bits = points.float().contiguous().view(torch.int32)
        primes = (73856093, 19349663, 83492791)[:self._dim]
        h = bits[..., 0].long() * primes[0]
        for a in range(1, self._dim):
            h = h ^ (bits[..., a].long() * primes[a])
        h = h ^ self.__seed

        # Generate dim independent draws by mixing h with a per-dimension offset.
        draws = []
        for d in range(self._dim):
            hd = h ^ (d * 2654435761)
            # Finalisation mix for better distribution.
            hd = hd ^ (hd >> 16)
            hd = (hd * 0x45d9f3b) & 0xFFFFFFFF
            hd = hd ^ (hd >> 16)
            draws.append((hd & 0x7FFFFFFF).float() / 0x7FFFFFFF)
        return torch.stack(draws, dim=-1)

    # This method returns the control point locations, displacements
    # The control grid random displacements must not change depending on the passed points.
    # That is, if we pass the point (0.5, 0.5, 0.5) this must give the same transformed
    # point regardless of the other points we pass.
    # This means that subsequent calls to transform_points and backward_transform_points
    # will align points and images properly.
    def control_grid(
        self,
        points: PointsTensor,
        ) -> Tuple[ChannelImageTensor, AffineTensor]:
        # Get the origin/spacing for this point cloud.
        cp_spacing = self.__control_spacing.to(points.device)
        cp_global_origin = self.__control_origin.to(points.device)
        point_min, _ = points.min(dim=0)
        point_max, _ = points.max(dim=0)
        cp_idx_min = torch.floor((point_min - cp_global_origin) / cp_spacing)
        cp_idx_max = torch.ceil((point_max - cp_global_origin) / cp_spacing)
        if self.__method in ('cubic', 'bspline'):
            # Add an extra boundary point on each end of each axis.
            cp_idx_min -= 1
            cp_idx_max += 1
        cp_origin = cp_idx_min * cp_spacing + cp_global_origin

        # Create integer index grid — indices uniquely identify control points
        # and are used for the spatial hash (avoids floating-point sensitivity).
        cp_indices = torch.stack(torch.meshgrid([
            torch.arange(cp_idx_min[a].item(), cp_idx_max[a].item() + 1) for a in range(self._dim)
        ], indexing='ij'), dim=-1)
        cp_indices = cp_indices.to(device=points.device)

        # Convert indices to world coordinates.
        cps = cp_indices * cp_spacing + cp_global_origin

        # Generate reproducible displacements via vectorised spatial hash.
        cp_size = cp_indices.shape[:-1]
        draws = self.__control_grid_draws(cps.reshape(-1, self._dim))
        draws = draws.reshape(*cp_size, self._dim)
        disp_range = self.__disp_range.to(points.device)
        cp_disps = draws * (disp_range[:, 1] - disp_range[:, 0]) + disp_range[:, 0]

        # Bring channels to the front.
        cp_disps = torch.moveaxis(cp_disps, -1, 0)
        cp_affine = create_affine(cp_spacing, cp_origin)
        
        return cp_disps, cp_affine

    def __str__(self) -> str:
        params = dict(
            control_origin=to_tuple(self.__control_origin, decimals=3),
            control_spacing=to_tuple(self.__control_spacing, decimals=3),
            displacement=to_tuple(self.__disp_range.flatten(), decimals=3),
        )
        return super().__str__(self.__class__.__name__, params)

    def transform_points(
        self,
        points: Points,
        filter_offgrid: bool = True,
        grid: SamplingGrid | None = None,   # Required for 'image-centre' rotation/scale.
        return_filtered: bool = False,
        **kwargs,
        ) -> Points | List[Points | np.ndarray | torch.Tensor]:
        points, return_type = to_tensor(points, return_type=True)

        # Define the backward transform.

        # Let: y = x + b(x) be the location of the back-transformed point x.  
        # Let: F(x) = x + b(x) - y, and solve for F(x) = 0 (using Newton-Rahpson) to find x for a given y.
        n_iter = 100     # Log the required number of iterations for solve and adjust.
        x_i = points.clone().requires_grad_()
        b = self.backward_transform_points
        for i in range(n_iter):
            # Perform transform.
            y_i = x_i + b(x_i)

            # Check convergence.
            if torch.isclose(y_i, points).all():
                break
            elif i == n_iter - 1:
                raise ValueError('No convergence after max iterations: ', i)

            # Get Jacobians for batch of points.
            grads = []
            for a in range(self._dim):
                grad_a, = torch.autograd.grad(y_i[:, a], x_i, grad_outputs=torch.ones(len(x_i)).to(x_i.device), retain_graph=True)
                grads.append(grad_a)
            J = torch.stack(grads, dim=1)

            # Batch solve for deltas for each point.
            r = y_i - points
            dx = torch.linalg.solve(J, r)

            # Update guess.
            x_i = x_i.detach()  # How does it get 'requires_grad_' again, must be through 'dx'.
            x_i = x_i - dx

        x_i = x_i.detach()

        # Forward transformed points could end up off-screen and should be filtered.
        # However, we need to know which points are returned for loss calc for example.
        points_t = x_i
        if filter_offgrid:
            size, affine = grid if grid is not None else (None, None)
            assert size is not None
            assert affine is not None
            size = to_tensor(size, device=points.device, dtype=points.dtype)
            affine = to_tensor(affine, device=points.device, dtype=points.dtype)
            fov = torch.stack([affine[:3, 3], affine[:3, 3] + size * affine[:3, :3].diag()]).to(points.device)
            to_keep = (points_t >= fov[0]) & (points_t < fov[1])
            to_keep = to_keep.all(axis=1)
            points_t = points_t[to_keep]
            indices = torch.where(to_keep)[0]

        # Convert return types.
        if return_type is np.ndarray:
            points_t = to_numpy(points_t)
            indices = to_numpy(indices) if filter_offgrid else None

        # Format returned values.
        results = points_t
        if filter_offgrid and return_filtered:
            results = [points_t, indices]
        return results

class RandomElastic(RandomSpatialTransform):
    def __init__(
        self, 
        control_spacing: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 50.0,
        control_origin: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 20.0,
        displacement: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 20.0,
        # Can we randomise the fitting method too?
        method: Literal['bspline', 'cubic', 'linear'] = 'linear',
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        self.__method = method
        control_spacing_range = expand_range_arg(control_spacing, dim=self._dim)
        assert len(control_spacing_range) == 2 * self._dim, f"Expected 'control_spacing' of length {2 * self._dim}, got {len(control_spacing_range)}."
        self.__control_spacing_range = to_tensor(control_spacing_range).reshape(self._dim, 2)
        control_origin_range = expand_range_arg(control_origin, dim=self._dim, negate_lower=True)
        assert len(control_origin_range) == 2 * self._dim, f"Expected 'control_origin' of length {2 * self._dim}, got {len(control_origin_range)}."
        self.__control_origin_range = to_tensor(control_origin_range).reshape(self._dim, 2)
        disp_range = expand_range_arg(displacement, dim=self._dim, negate_lower=True)
        assert len(disp_range) == 2 * self._dim, f"Expected 'displacement' of length {2 * self._dim}, got {len(disp_range)}."
        self.__disp_range = to_tensor(disp_range).reshape(self._dim, 2)
        self._params = dict(
            type=self.__class__.__name__,
            control_origin=self.__control_origin_range,
            control_spacing=self.__control_spacing_range,
            dim=self._dim,
            displacement=self.__disp_range,
            method=self.__method,
            p=self._p,
        )

        self.__warn_folding()

    def __warn_folding(self, control_spacing: torch.Tensor | None = None) -> None:
        if control_spacing is None:
            control_spacing, _ = self.__control_spacing_range.min(axis=1)
        disp_widths = self.__disp_range[:, 1] - self.__disp_range[:, 0]
        if (disp_widths >= control_spacing).any():
            logger.warning(f"RandomElastic transforms with displacement widths ({to_tuple(disp_widths)}) >= "
                f"control spacings ({to_tuple(control_spacing)}) may produce folding transforms. Such transforms may "
                f"be non-invertible and could raise errors when performing forward points transform.")

    def freeze(self) -> 'Elastic':
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random(self._dim))
        control_spacing_draw = draw * (self.__control_spacing_range[:, 1] - self.__control_spacing_range[:, 0]) + self.__control_spacing_range[:, 0]
        control_origin_draw = draw * (self.__control_origin_range[:, 1] - self.__control_origin_range[:, 0]) + self.__control_origin_range[:, 0]
        # We can't draw displacements here as we need the image to determine the number of control points.
        # However, we should pass a randomly-drawn seed.
        seed_draw = self._rng.integers(1e9)   # Requires upper bound.
        self.__warn_folding(control_spacing_draw)
        params = dict(
            control_origin=control_origin_draw,
            control_spacing=control_spacing_draw,
            displacement=self.__disp_range.flatten(),
            method=self.__method,
            seed=seed_draw,
        )
        return super().freeze(Elastic, params)

    def __str__(self) -> str:
        params = dict(
            control_origin=to_tuple(self.__control_origin_range.flatten(), decimals=3),
            control_spacing=to_tuple(self.__control_spacing_range.flatten(), decimals=3),
            displacement=to_tuple(self.__disp_range.flatten(), decimals=3),
        )
        return super().__str__(self.__class__.__name__, params)
