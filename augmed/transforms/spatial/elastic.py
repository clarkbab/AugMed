from __future__ import annotations

import numpy as np
import torch
from typing import List, Literal, Tuple

from ...typing import AffineMatrix, AffineMatrixTensor, ChannelImageTensor, Indices, Number, Points, PointsTensor, Size, SpatialDim, TransformParams
from ...utils.args import alias_kwargs, arg_to_list, expand_range_arg
from ...utils.assertions import assert_points_shapes
from ...utils.conversion import to_return_format, to_tensor, to_tuple
from ...utils.geometry import affine_origin, affine_spacing, create_affine, fov
from ...utils.logging import logger
from ...utils.python import get_group_device, wrap_quotes
from ..identity import Identity
from .spatial import RandomSpatialTransform, SpatialTransform

BATCHING_MEM_P = 0.25           # Proportion of total GPU used before batching kicks in.
BATCHING_MIN_POINTS = int(1e5)       # Number of points above which batching is considered.
N_ITER_MAX = 100                # Max iterations for forward points transform solve.
CLOSENESS_TOL = 1e-6           # Tolerance for closeness in forward points transform solve, in mm. 

# Defines a coarse grid of control points.
# Random displacements are assigned at each control point.
class Elastic(SpatialTransform):
    @alias_kwargs(
        ('bm', 'batching_mem_p'),
        ('cs', 'control_spacing'),
        ('co', 'control_origin'),
        ('d', 'displacement'),
        ('m', 'method'),
        ('n', 'n_iter_max'),
        ('s', 'seed'),
        ('ub', 'use_batching'),
    )
    def __init__(
        self,
        batching_mem_p: float = BATCHING_MEM_P,
        control_spacing: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 50.0,
        control_origin: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 20.0,
        displacement: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 20.0,
        method: Literal['bspline', 'cubic', 'linear'] = 'bspline',
        n_iter_max: int = N_ITER_MAX,
        seed: int = 42,
        use_batching: bool = True,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        assert method in ['bspline', 'cubic', 'linear'], "Only 'bspline', 'cubic', and 'linear' elastic methods are supported."
        self.__method = method
        self.__control_spacing = to_tensor(control_spacing, broadcast=self.__dim)
        assert len(self.__control_spacing) == self.__dim, f"Expected 'control_spacing' of length '{self.__dim}' for dim={self.__dim}, got {len(self.__control_spacing)}."
        self.__control_origin = to_tensor(control_origin, broadcast=self.__dim)
        assert len(self.__control_origin) == self.__dim, f"Expected 'control_origin' of length '{self.__dim}' for dim={self.__dim}, got {len(self.__control_origin)}."
        # Disps aren't known until presented with the image.
        disp_range = expand_range_arg(displacement, dim=self.__dim, negate_lower=True)
        assert len(disp_range) == 2 * self.__dim, f"Expected 'displacement' of length {2 * self.__dim}, got {len(disp_range)}."
        self.__disp_range = to_tensor(disp_range).reshape(self.__dim, 2).T
        self.__seed = seed
        self.__use_batching = use_batching
        self.__batching_mem_p = batching_mem_p
        self.__n_iter_max = n_iter_max
        self.__warn_folding()

    def back_transform_points(
        self,
        points: PointsTensor,
        **kwargs,
        ) -> PointsTensor:
        if self.__debug:
            print("=== Elastic.back_transform_points (start) ===")
            print('points: ', points.shape)

        # Get the control grid - will be large enough to cover all points.
        # This returns the displacements and the affine required to place them
        # in world coordinates.
        cp_disps, cp_affine = self.control_grid(points)
        cp_spacing = affine_spacing(cp_affine)
        cp_origin = affine_origin(cp_affine)
        cp_disps = torch.moveaxis(cp_disps, 0, -1)  # Move channels dim to back.
        if self.__debug:
            print('cp_disps: ', cp_disps.shape, cp_disps.dtype)
            print('cp_affine: ', cp_affine.shape, cp_affine.dtype)

        # Normalise points to the control grid integer coords.
        points_norm = (points - cp_origin) / cp_spacing
        if self.__debug:
            print('points_norm: ', points_norm.shape, points_norm.dtype)

        # Get lowest corner point.
        corner_min = points_norm.floor().type(torch.int32)
        if self.__debug:
            print('corner_min: ', corner_min.shape, corner_min.dtype)

        # Get distances from corner.
        t = points_norm - corner_min

        # Compute basis by method.
        if self.__method == 'linear':
            b = torch.stack([1 - t, t], dim=-2).type(torch.float32)
            corner_range = to_tensor([0, 1], dtype=torch.float32)
        else:
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
            b = torch.stack([w0, w1, w2, w3], dim=-2).type(torch.float32)
            corner_range = to_tensor([-1, 0, 1, 2], dtype=torch.int32)
        if self.__debug:
            print('b: ', b.shape, b.dtype)
            print('corner_range: ', corner_range, corner_range.dtype)

        # Get corner point offsets (small, independent of N).
        offsets = torch.stack(torch.meshgrid([corner_range] * self.__dim, indexing='ij'), dim=-1).type(torch.int32)
        offsets = offsets.reshape(-1, self.__dim).to(points.device)
        if self.__debug:
            print('offsets: ', offsets.shape, offsets.dtype)

        # Get the number of batches required to process points on GPU only.
        if self.__use_batching:
            n_points = points.shape[0]
            n_batches = self.__get_n_batches(n_points, points.device)
            if self.__debug and n_batches == 1:
                print("Batching not required.")

        if self.__use_batching and n_batches > 1:
            disps = torch.empty_like(points)
            batch_indices = torch.linspace(0, n_points, n_batches + 1, dtype=torch.long)

            # Process batches.
            for i in range(n_batches):
                if self.__debug:
                    print(f"Points batch {i + 1}/{n_batches}")
                start, end = batch_indices[i].item(), batch_indices[i + 1].item()
                corner_min_batch = corner_min[start:end]
                b_batch = b[start:end]
                if self.__debug:
                    print('corner_min_batch: ', corner_min_batch.shape, corner_min_batch.dtype)
                    print('b_batch: ', b_batch.shape, b_batch.dtype)

                # Calculate corners for each point.
                corners_batch = corner_min_batch[:, None, :] + offsets[None, :, :]
                if self.__debug:
                    print('corners_batch: ', corners_batch.shape, corners_batch.dtype)

                # Split into x/y/z indices to perform control point disp selection.
                idxs = corners_batch.unbind(-1)
                corner_disps_batch = cp_disps[tuple(idxs)]
                if self.__debug:
                    print('corner_disps_batch: ', corner_disps_batch.shape, corner_disps_batch.dtype)

                # Reshape for einsum.
                V_batch = corner_disps_batch.reshape(-1, *(len(corner_range), ) * self.__dim, self.__dim)
                if self.__debug:
                    print('V_batch: ', V_batch.shape, V_batch.dtype)

                # Tensor-product interpolation via einsum.
                # Contract one axis at a time (separable) to reduce cost from O(n^d) to O(n*d).
                for a in range(self.__dim):
                    V_batch = torch.einsum('ni,ni...d->n...d', b_batch[:, :, a], V_batch)

                disps[start:end] = V_batch

                # Free batch intermediates.
                del corner_min_batch, b_batch, corners_batch, idxs, corner_disps_batch, V_batch
        else:
            # Calculate corners for each point.
            corners = corner_min[:, None, :] + offsets[None, :, :]
            if self.__debug:
                print('corners: ', corners.shape, corners.dtype)

            # Split into x/y/z indices to perform control point disp selection.
            idxs = corners.unbind(-1)
            corner_disps = cp_disps[tuple(idxs)]
            if self.__debug:
                print('corner_disps: ', corner_disps.shape, corner_disps.dtype)

            # Reshape for einsum.
            V = corner_disps.reshape(-1, *(len(corner_range), ) * self.__dim, self.__dim)
            if self.__debug:
                print('V: ', V.shape, V.dtype)

            # Tensor-product interpolation via einsum.
            # Contract one axis at a time (separable) to reduce cost from O(n^d) to O(n*d).
            for a in range(self.__dim):
                V = torch.einsum('ni,ni...d->n...d', b[:, :, a], V)
            disps = V

        # Get displaced input points.
        points_t = points + disps

        if self.__debug:
            print("=== Elastic.back_transform_points (end) ===")

        return points_t

    def control_grid(
        self,
        points: PointsTensor,
        ) -> Tuple[ChannelImageTensor, AffineMatrixTensor]:
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
            torch.arange(cp_idx_min[a].item(), cp_idx_max[a].item() + 1) for a in range(self.__dim)
        ], indexing='ij'), dim=-1)
        cp_indices = cp_indices.to(device=points.device)

        # Convert indices to world coordinates.
        cps = cp_indices * cp_spacing + cp_global_origin

        # Generate reproducible displacements via vectorised spatial hash.
        cp_size = cp_indices.shape[:-1]
        draws = self.__control_grid_draws(cps.reshape(-1, self.__dim))
        draws = draws.reshape(*cp_size, self.__dim)
        disp_range = self.__disp_range.to(points.device)
        cp_disps = draws * (disp_range[1] - disp_range[0]) + disp_range[0]

        # Bring channels to the front.
        cp_disps = torch.moveaxis(cp_disps, -1, 0)
        cp_affine = create_affine(cp_spacing, cp_origin, device=points.device)
        
        return cp_disps, cp_affine

    def __control_grid_draws(
        self,
        points: PointsTensor,
        ) -> PointsTensor:
        bits = points.float().contiguous().view(torch.int32)
        primes = (73856093, 19349663, 83492791)[:self.__dim]
        h = bits[..., 0].long() * primes[0]
        for a in range(1, self.__dim):
            h = h ^ (bits[..., a].long() * primes[a])
        h = h ^ self.__seed

        # Generate dim independent draws by mixing h with a per-dimension offset.
        draws = []
        for d in range(self.__dim):
            hd = h ^ (d * 2654435761)
            # Finalisation mix for better distribution.
            hd = hd ^ (hd >> 16)
            hd = (hd * 0x45d9f3b) & 0xFFFFFFFF
            hd = hd ^ (hd >> 16)
            draws.append((hd & 0x7FFFFFFF).float() / 0x7FFFFFFF)
        return torch.stack(draws, dim=-1)

    # For each point in the control grid, a displacement vector is drawn in a 
    # pseudo-random manner. The same control point will always yield the same 
    # displacement vector for a given 'self.__seed'. We need to do this as we
    # don't know the required control grid size until transform time, and we
    # don't want to create an excessively large control grid to handle all possible
    # images/point clouds.
    # TODO: Claude wrote this bit - it might need expert eyes.
    def __estimate_back_transform_memory(
        self,
        n_points: int,
        ) -> int:
        d = self.__dim
        n = 4 if self.__method in ('cubic', 'bspline') else 2
        n_d = n ** d
        # (N, n^d, d) int32 — corner indices.
        corners = n_points * n_d * d * 4
        # (N, n^d, d) float32 — corner displacements; V is a view of this so no extra alloc until first einsum.
        corner_disps = n_points * n_d * d * 4
        # (N, n^{d-1}, d) float32 — output of first einsum contraction, coexists briefly with corner_disps input.
        einsum_out = n_points * (n_d // n) * d * 4
        # (N, n, d) float32 — interpolation basis weights.
        b = n_points * n * d * 4
        # t2, t3 for cubic/bspline: 2 * (N, d) float32.
        t_extra = 2 * n_points * d * 4 if self.__method in ('cubic', 'bspline') else 0
        # corner_min, t, points_norm, disps, points_t: 5 * (N, d).
        other = 5 * n_points * d * 4
        return corners + corner_disps + einsum_out + b + t_extra + other

    # Estimates the bytes of VRAM required per point for the back transform points
    # method. This is used to determine batching requirements.
    def __get_n_batches(
        self,
        n_points: int,
        device: torch.device,
        ) -> int:
        if self.__debug:
            print("=== Elastic.__get_n_batches (start) ===")

        if device.type == 'cuda':
            if n_points < BATCHING_MIN_POINTS:
                if self.__debug:
                    print(f"Number of points ({n_points}) below batching threshold ({BATCHING_MIN_POINTS}).")
                n_batches = 1
            else:
                mem_total = torch.cuda.get_device_properties(device).total_memory
                if self.__debug:
                    print(f"Total GPU memory: {mem_total / (1024 ** 3):.2f} GB")
                mem_budget = int(mem_total * self.__batching_mem_p)
                if self.__debug:
                    print(f"Allowing up to {mem_budget / (1024 ** 3):.2f} GB before applying batching.")
                bytes_est = self.__estimate_back_transform_memory(n_points)
                if self.__debug:
                    print(f"Estimated GPU memory usage: {bytes_est / (1024 ** 3):.2f} GB")
                n_batches = int(np.ceil(bytes_est / mem_budget))        
        else:
            # No batching required for CPU.
            n_batches = 1

        if self.__debug:
            print(f"Total batches required: {n_batches}.")
            print("=== Elastic.__get_n_batches (end) ===")

        return n_batches
        
    # Uses batching to reduce peak GPU memory usage and potential offloading to CPU.
    @property
    def params(self) -> TransformParams:
        return super().params(
            batching_mem_p=self.__batching_mem_p,
            control_origin=to_tuple(self.__control_origin),
            control_spacing=to_tuple(self.__control_spacing),
            displacement=to_tuple(self.__disp_range.T.flatten()),
            method=wrap_quotes(self.__method),
            n_iter_max=self.__n_iter_max,
            seed=self.__seed,
            use_batching=self.__use_batching,
        )

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().__str__(
            self.__class__.__name__,
            control_origin=to_tuple(self.__control_origin, dp=3),
            control_spacing=to_tuple(self.__control_spacing, dp=3),
            displacement=to_tuple(self.__disp_range.T.flatten(), dp=3),
            method=wrap_quotes(self.__method),
            n_iter_max=self.__n_iter_max,
            seed=self.__seed,
            subtransform=subtransform,
        )

    @alias_kwargs(
        ('a', 'affine'),
        ('fo', 'filter_offgrid'),
        ('rf', 'return_filtered'),
        ('rs', 'return_single'),
        ('s', 'size'),
    )
    def transform_points(
        self,
        points: Points | List[Points],
        affine: AffineMatrix | None = None,       # Required for some transforms, e.g. Rotate, to get centre of rotation.
        filter_offgrid: bool | None = None,
        # grid: SamplingGrid | None = None,   # Required for filtering off-grid points and some transforms, e.g. Rotate.
        return_filtered: bool = False,
        return_single: bool = True,
        size: Size | None = None,           # Required for filtering off-grid points.
        **kwargs,
        ) -> Points | List[Points | Indices | List[Indices]]:
        assert_points_shapes(points, self.__dim)
        points, points_was_single = arg_to_list(points, (np.ndarray, torch.Tensor), return_matched=True)
        device = get_group_device(points, device=self.__device)
        return_types = [type(p) for p in points]
        points = [to_tensor(p, device=device) for p in points]
        size = to_tensor(size, device=device, dtype=torch.int32)
        affine = to_tensor(affine, device=device, dtype=torch.float32)
        filter_offgrid = filter_offgrid if filter_offgrid is not None else self.__filter_offgrid

        points_ts = []
        indiceses = []
        for p in points:
            # Method:
            # - Let: y = x + b(x) be the location of the back-transformed point x.  
            # - Let: F(x) = x + b(x) - y
            # - Solve F(x) = 0, for x (using Newton-Raphson).
            x_i = p.clone().requires_grad_()   # Use y as initial guess for x.
            b = self.back_transform_points      # Gives x + b(x).
            for i in range(self.__n_iter_max):
                # Perform transform.
                y_i = b(x_i)

                # Check convergence.
                if torch.isclose(y_i, p, atol=CLOSENESS_TOL).all():
                    break
                elif i == self.__n_iter_max - 1:
                    raise ValueError(f"Elastic.transform_points failed to converge after {self.__n_iter_max} iterations.")

                # Get Jacobians for batch of points.
                grads = []
                for a in range(self.__dim):
                    grad_a, = torch.autograd.grad(y_i[:, a], x_i, grad_outputs=torch.ones(len(x_i)).to(device), retain_graph=True)
                    grads.append(grad_a)
                J = torch.stack(grads, dim=1)

                # Batch solve for deltas for each point.
                r = y_i - p
                dx = torch.linalg.solve(J, r)

                # Update guess.
                x_i = x_i.detach()  # How does it get 'requires_grad_' again, must be through 'dx'.
                x_i = x_i - dx

            if self.__debug:
                print(f"Elastic.transform_points converged after {i} iterations.")

            points_t = x_i.detach()

            # Forward transformed points could end up off-screen and should be filtered.
            # However, we need to know which points are returned for loss calc for example.
            if filter_offgrid:
                assert size is not None, "Size must be provided for filtering off-grid points."
                assert affine is not None, "Affine must be provided for filtering off-grid points."
                fov_mm = fov((size, affine))
                to_keep = (points_t >= fov_mm[0]) & (points_t < fov_mm[1])
                to_keep = to_keep.all(axis=1)
                points_t = points_t[to_keep]
                indices = torch.where(~to_keep)[0].type(torch.int32)
                indiceses.append(indices)

            points_ts.append(points_t)

        # Convert to return format.
        other_data = []
        if filter_offgrid and return_filtered:
            indiceses = to_return_format(indiceses, return_single=False, return_types=return_types)
            other_data.append(indiceses)
        return to_return_format(points_ts, other_data=other_data, return_single=return_single and points_was_single, return_types=return_types)

    def __warn_folding(self, control_spacing: torch.Tensor | None = None) -> None:
        if control_spacing is None:
            control_spacing = self.__control_spacing
        disp_widths = self.__disp_range[1] - self.__disp_range[0]
        if (disp_widths >= control_spacing).any():
            logger.warn(f"Elastic transforms with displacement widths ({to_tuple(disp_widths)}) >= "
                f"control spacings ({to_tuple(control_spacing)}) may produce folding transforms. Such transforms may "
                f"be non-invertible and could raise errors when performing forward points transform.")

class RandomElastic(RandomSpatialTransform):
    @alias_kwargs(
        ('bm', 'batching_mem_p'),
        ('cs', 'control_spacing'),
        ('co', 'control_origin'),
        ('d', 'displacement'),
        ('m', 'method'),
        ('n', 'n_iter_max'),
        ('ub', 'use_batching'),
    )
    def __init__(
        self, 
        batching_mem_p: float = BATCHING_MEM_P,
        control_spacing: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 50.0,
        control_origin: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 20.0,
        displacement: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 20.0,
        # Can we randomise the fitting method too?
        method: Literal['bspline', 'cubic', 'linear'] = 'bspline',
        n_iter_max: int = N_ITER_MAX,
        use_batching: bool = True,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        self.__method = method
        self.__control_spacing = control_spacing
        self.__control_origin = control_origin
        self.__displacement = displacement
        self.__use_batching = use_batching
        self.__batching_mem_p = batching_mem_p
        self.__n_iter_max = n_iter_max
        self.__expand_range_args()

    def __expand_range_args(self) -> None:
        control_spacing_range = expand_range_arg(self.__control_spacing, dim=self.__dim)
        assert len(control_spacing_range) == 2 * self.__dim, f"Expected 'control_spacing' of length {2 * self.__dim}, got {len(control_spacing_range)}."
        self.__control_spacing_range = to_tensor(control_spacing_range).reshape(self.__dim, 2).T
        control_origin_range = expand_range_arg(self.__control_origin, dim=self.__dim, negate_lower=True)
        assert len(control_origin_range) == 2 * self.__dim, f"Expected 'control_origin' of length {2 * self.__dim}, got {len(control_origin_range)}."
        self.__control_origin_range = to_tensor(control_origin_range).reshape(self.__dim, 2).T
        disp_range = expand_range_arg(self.__displacement, dim=self.__dim, negate_lower=True)
        assert len(disp_range) == 2 * self.__dim, f"Expected 'displacement' of length {2 * self.__dim}, got {len(disp_range)}."
        self.__displacement_range = to_tensor(disp_range).reshape(self.__dim, 2).T

    def freeze(self) -> Elastic | Identity:
        should_apply = self.__rng.random(1) < self.__p
        if not should_apply:
            return Identity(dim=self.__dim)

        draw = to_tensor(self.__rng.random(self.__dim))
        control_spacing_draw = draw * (self.__control_spacing_range[1] - self.__control_spacing_range[0]) + self.__control_spacing_range[0]
        control_origin_draw = draw * (self.__control_origin_range[1] - self.__control_origin_range[0]) + self.__control_origin_range[0]
        # We can't draw displacements here as we need the image to determine the number of control points.
        # However, we should pass a randomly-drawn seed.
        seed_draw = self.__rng.integers(1e9)   # Requires upper bound.
        self.__warn_folding(control_spacing_draw, self.__displacement_range)

        params = dict(
            batching_mem_p=self.__batching_mem_p,
            control_origin=control_origin_draw,
            control_spacing=control_spacing_draw,
            displacement=self.__displacement_range.T.flatten(),
            method=self.__method,
            n_iter_max=self.__n_iter_max,
            seed=seed_draw,
            use_batching=self.__use_batching,
        )
        return super().freeze(Elastic, params)

    @property
    def params(self) -> TransformParams:
        return super().params(
            batching_mem_p=self.__batching_mem_p,
            control_origin=to_tuple(self.__control_origin_range.T.flatten()),
            control_spacing=to_tuple(self.__control_spacing_range.T.flatten()),
            displacement=to_tuple(self.__displacement_range.T.flatten()),
            method=wrap_quotes(self.__method),
            n_iter_max=self.__n_iter_max,
            use_batching=self.__use_batching,
        )

    def set_dim(
        self,
        dim: SpatialDim,
        ) -> None:
        super().set_dim(dim)
        self.__expand_range_args()

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
        self,
        subtransform: bool = False,
        ) -> str:
        return super().__str__(
            self.__class__.__name__,
            batching_mem_p=self.__batching_mem_p,
            control_origin=to_tuple(self.__control_origin_range.T.flatten(), dp=3),
            control_spacing=to_tuple(self.__control_spacing_range.T.flatten(), dp=3),
            displacement=to_tuple(self.__displacement_range.T.flatten(), dp=3),
            method=wrap_quotes(self.__method),
            n_iter_max=self.__n_iter_max,
            subtransform=subtransform,
            use_batching=self.__use_batching,
        )

    def __warn_folding(self, control_spacing, disp_range) -> None:
        disp_widths = disp_range[1] - disp_range[0]
        if (disp_widths >= control_spacing).any():
            logger.warn(f"RandomElastic transforms with displacement widths ({to_tuple(disp_widths)}) >= "
                f"control spacings ({to_tuple(control_spacing)}) may produce folding transforms. Such transforms may "
                f"be non-invertible and could raise errors when performing forward points transform.")
