import numpy as np
import torch
from typing import List, Literal, Tuple

from ...typing import Affine, AffineTensor, ChannelImageTensor, Indices, Number, Points, PointsTensor, Size
from ...utils.args import expand_range_arg
from ...utils.conversion import to_return_format, to_tensor, to_tuple
from ...utils.logging import logger
from ...utils.matrix import affine_origin, affine_spacing, create_affine
from ..identity import Identity
from .spatial import RandomSpatialTransform, SpatialTransform

BATCHING_MEM_P = 0.25
N_ITER_MAX = 100

# Defines a coarse grid of control points.
# Random displacements are assigned at each control point.
class Elastic(SpatialTransform):
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
        self.__control_spacing = to_tensor(control_spacing, broadcast=self._dim)
        assert len(self.__control_spacing) == self._dim, f"Expected 'control_spacing' of length '{self._dim}' for dim={self._dim}, got {len(self.__control_spacing)}."
        self.__control_origin = to_tensor(control_origin, broadcast=self._dim)
        assert len(self.__control_origin) == self._dim, f"Expected 'control_origin' of length '{self._dim}' for dim={self._dim}, got {len(self.__control_origin)}."
        # Disps aren't known until presented with the image.
        disp_range = expand_range_arg(displacement, dim=self._dim, negate_lower=True)
        assert len(disp_range) == 2 * self._dim, f"Expected 'displacement' of length {2 * self._dim}, got {len(disp_range)}."
        self.__disp_range = to_tensor(disp_range).reshape(self._dim, 2)
        self.__seed = seed
        self.__use_batching = use_batching
        self.__batching_mem_p = batching_mem_p
        self.__n_iter_max = n_iter_max
        self.__warn_folding()
        self._params = dict(
            batching_mem_p=self.__batching_mem_p,
            control_origin=self.__control_origin,
            control_spacing=self.__control_spacing,
            dim=self._dim,
            displacement=self.__disp_range,
            method=self.__method,
            n_iter_max=self.__n_iter_max,
            seed=self.__seed,
            type=self.__class__.__name__,
            use_batching=self.__use_batching,
        )

    def backward_transform_points(
        self,
        points: PointsTensor,
        **kwargs,
        ) -> PointsTensor:
        if self._debug:
            print("=== Elastic.backward_transform_points (start) ===")

        # Get the control grid - will be large enough to cover all points.
        cp_disps, cp_affine = self.control_grid(points)
        cp_spacing = affine_spacing(cp_affine)
        cp_origin = affine_origin(cp_affine)
        cp_disps = torch.moveaxis(cp_disps, 0, -1)  # Move channels dim to back.
        if self._debug:
            print('cp_disps: ', cp_disps.shape)

        # Normalise points to the control grid integer coords.
        points_norm = (points - cp_origin) / cp_spacing
        if self._debug:
            print('points_norm: ', points_norm.shape)

        # Get lowest corner point.
        corner_min = points_norm.floor().type(torch.int32)
        if self._debug:
            print('corner_min: ', corner_min.shape)

        # Get distances from corner.
        t = points_norm - corner_min

        # Compute basis by method.
        if self.__method == 'linear':
            b = torch.stack([1 - t, t], dim=-2)
            corner_range = to_tensor([0, 1])
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
            b = torch.stack([w0, w1, w2, w3], dim=-2)
            corner_range = to_tensor([-1, 0, 1, 2])
        if self._debug:
            print('b: ', b.shape)
            print('corner_range: ', corner_range)

        # Get corner point offsets (small, independent of N).
        offsets = torch.stack(torch.meshgrid([corner_range] * self._dim, indexing='ij'), dim=-1)
        offsets = offsets.reshape(-1, self._dim).to(points.device)
        if self._debug:
            print('offsets: ', offsets.shape)

        # Get the number of batches required to process points on GPU only.
        if self.__use_batching:
            n_points = points.shape[0]
            n_batches = self.__get_n_batches(n_points, points.device)
            if self._debug and n_batches == 1:
                print("Batching not required.")

        if self.__use_batching and n_batches > 1:
            disps = torch.empty_like(points)
            batch_indices = torch.linspace(0, n_points, n_batches + 1, dtype=torch.long)

            # Process batches.
            for i in range(n_batches):
                if self._debug:
                    print(f"Points batch {i + 1}/{n_batches}")
                start, end = batch_indices[i].item(), batch_indices[i + 1].item()
                corner_min_batch = corner_min[start:end]
                b_batch = b[start:end]
                if self._debug:
                    print('corner_min_batch: ', corner_min_batch.shape)
                    print('b_batch: ', b_batch.shape)

                # Calculate corners for each point.
                corners_batch = corner_min_batch[:, None, :] + offsets[None, :, :]
                if self._debug:
                    print('corners_batch: ', corners_batch.shape)

                # Split into x/y/z indices to perform control point disp selection.
                idxs = corners_batch.unbind(-1)
                corner_disps_batch = cp_disps[tuple(idxs)]
                if self._debug:
                    print('corner_disps_batch: ', corner_disps_batch.shape)

                # Reshape for einsum.
                V_batch = corner_disps_batch.reshape(-1, *(len(corner_range), ) * self._dim, self._dim)
                if self._debug:
                    print('V_batch: ', V_batch.shape)

                # Tensor-product interpolation via einsum.
                # Contract one axis at a time (separable) to reduce cost from O(n^d) to O(n*d).
                for a in range(self._dim):
                    V_batch = torch.einsum('ni,ni...d->n...d', b_batch[:, :, a], V_batch)

                disps[start:end] = V_batch

                # Free batch intermediates.
                del corner_min_batch, b_batch, corners_batch, idxs, corner_disps_batch, V_batch
        else:
            # Calculate corners for each point.
            corners = corner_min[:, None, :] + offsets[None, :, :]
            if self._debug:
                print('corners: ', corners.shape)

            # Split into x/y/z indices to perform control point disp selection.
            idxs = corners.unbind(-1)
            corner_disps = cp_disps[tuple(idxs)]
            if self._debug:
                print('corner_disps: ', corner_disps.shape)

            # Reshape for einsum.
            V = corner_disps.reshape(-1, *(len(corner_range), ) * self._dim, self._dim)
            if self._debug:
                print('V: ', V.shape)

            # Tensor-product interpolation via einsum.
            # Contract one axis at a time (separable) to reduce cost from O(n^d) to O(n*d).
            for a in range(self._dim):
                V = torch.einsum('ni,ni...d->n...d', b[:, :, a], V)
            disps = V

        # Get displaced input points.
        points_t = points + disps

        if self._debug:
            print("=== Elastic.backward_transform_points (end) ===")

        return points_t

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
        cp_affine = create_affine(cp_spacing, cp_origin, device=points.device)
        
        return cp_disps, cp_affine

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

    def __estimate_bytes_per_point(self) -> int:
        """Estimate peak GPU bytes per point for the interpolation hot loop."""
        n = 4 if self.__method in ('cubic', 'bspline') else 2
        d = self._dim
        n_d = n ** d
        # Key per-point tensors alive simultaneously (float32=4 bytes, int32=4 bytes):
        # corners (int32): n^d * d * 4
        # corner_disps (float32): n^d * d * 4
        # V (float32): n^d * d * 4 (same memory as corner_disps after reshape, but separate during einsum)
        # b (float32): n * d * 4
        # corner_min, t, points_norm slices, disps, points_t: ~5 * d * 4
        return (3 * n_d * d + n * d + 5 * d) * 4
        
    # Vectorised spatial hash over world-space control point coordinates.
    # Reinterprets float32 coords as int32 bit patterns for hashing — deterministic
    # as long as the same arithmetic path produces the coordinates.
    # Returns draws in [0, 1) of shape (N, dim).
    # Comp
    def __get_n_batches(
        self,
        n_points: int,
        device: torch.device,
        ) -> int:
        if self._debug:
            print("=== Elastic.__get_n_batches (start) ===")
        if device.type == 'cuda':
            mem_total = torch.cuda.get_device_properties(device).total_memory
            if self._debug:
                print(f"Total GPU memory: {mem_total / (1024 ** 3):.2f} GB")
            mem_budget = int(mem_total * self.__batching_mem_p)
            if self._debug:
                print(f"Allowing up to {mem_budget / (1024 ** 3):.2f} GB before applying batching.")
            bpp = self.__estimate_bytes_per_point()
            if self._debug:
                print(f"Estimated bytes per point: {bpp} B.")
            points_per_batch = max(mem_budget // bpp, 1)
            if self._debug:
                print(f"Processing {points_per_batch} points per batch.")
            n_batches = max((n_points + points_per_batch - 1) // points_per_batch, 1)
            if self._debug:
                print(f"Total batches required: {n_batches}.")
        else:
            n_batches = 1

        if self._debug:
            print("=== Elastic.__get_n_batches (end) ===")

        return n_batches

    # This method returns the control point locations, displacements
    # The control grid random displacements must not change depending on the passed points.
    # That is, if we pass the point (0.5, 0.5, 0.5) this must give the same transformed
    # point regardless of the other points we pass.
    # This means that subsequent calls to transform_points and backward_transform_points
    # will align points and images properly.
    def __str__(self) -> str:
        params = dict(
            control_origin=to_tuple(self.__control_origin, decimals=3),
            control_spacing=to_tuple(self.__control_spacing, decimals=3),
            displacement=to_tuple(self.__disp_range.flatten(), decimals=3),
            method=self.__method,
            n_iter_max=self.__n_iter_max,
            seed=self.__seed,
        )
        return super().__str__(self.__class__.__name__, params)

    def transform_points(
        self,
        points: Points | List[Points],
        affine: Affine | None = None,       # Required for some transforms, e.g. Rotate, to get centre of rotation.
        filter_offgrid: bool = True,
        # grid: SamplingGrid | None = None,   # Required for filtering off-grid points and some transforms, e.g. Rotate.
        return_filtered: bool = False,
        size: Size | None = None,           # Required for filtering off-grid points.
        **kwargs,
        ) -> Points | List[Points | Indices | List[Indices]]:
        pointses, points_was_single = arg_to_list(points, (np.ndarray, torch.Tensor), return_expanded=True)
        device = get_group_device(pointses, device=self._device)
        return_types = [type(p) for p in pointses]
        pointses = [to_tensor(p, device=device) for p in pointses]
        size = to_tensor(size, device=device, dtype=torch.int32)
        affine = to_tensor(affine, device=device, dtype=torch.float32)

        points_ts = []
        indiceses = []
        for p in pointses:
            # Method:
            # - Let: y = x + b(x) be the location of the back-transformed point x.  
            # - Let: F(x) = x + b(x) - y
            # - Solve F(x) = 0, for x (using Newton-Raphson).
            x_i = p.clone().requires_grad_()   # Use y as initial guess for x.
            b = self.backward_transform_points      # Gives x + b(x).
            for i in range(self.__n_iter_max):
                # Perform transform.
                y_i = b(x_i)

                # Check convergence.
                if torch.isclose(y_i, p).all():
                    break
                elif i == self.__n_iter_max - 1:
                    raise ValueError(f"Elastic.transform_points failed to converge after {self.__n_iter_max} iterations.")

                # Get Jacobians for batch of points.
                grads = []
                for a in range(self._dim):
                    grad_a, = torch.autograd.grad(y_i[:, a], x_i, grad_outputs=torch.ones(len(x_i)).to(device), retain_graph=True)
                    grads.append(grad_a)
                J = torch.stack(grads, dim=1)

                # Batch solve for deltas for each point.
                r = y_i - points
                dx = torch.linalg.solve(J, r)

                # Update guess.
                x_i = x_i.detach()  # How does it get 'requires_grad_' again, must be through 'dx'.
                x_i = x_i - dx

            if self._debug:
                print(f"Elastic.transform_points converged after {i} iterations.")

            points_t = x_i.detach()

            # Forward transformed points could end up off-screen and should be filtered.
            # However, we need to know which points are returned for loss calc for example.
            if filter_offgrid:
                assert size is not None, "Size must be provided for filtering off-grid points."
                assert affine is not None, "Affine must be provided for filtering off-grid points."
                fov = torch.stack([affine[:3, 3], affine[:3, 3] + size * affine[:3, :3].diag()]).to(device)
                to_keep = (points_t >= fov[0]) & (points_t < fov[1])
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
        results = to_return_format(points_ts, other_data=other_data, return_single=points_was_single, return_types=return_types)
        return results

    def __warn_folding(self, control_spacing: torch.Tensor | None = None) -> None:
        if control_spacing is None:
            control_spacing = self.__control_spacing
        disp_widths = self.__disp_range[:, 1] - self.__disp_range[:, 0]
        if (disp_widths >= control_spacing).any():
            logger.warning(f"Elastic transforms with displacement widths ({to_tuple(disp_widths)}) >= "
                f"control spacings ({to_tuple(control_spacing)}) may produce folding transforms. Such transforms may "
                f"be non-invertible and could raise errors when performing forward points transform.")

class RandomElastic(RandomSpatialTransform):
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
        control_spacing_range = expand_range_arg(control_spacing, dim=self._dim)
        assert len(control_spacing_range) == 2 * self._dim, f"Expected 'control_spacing' of length {2 * self._dim}, got {len(control_spacing_range)}."
        self.__control_spacing_range = to_tensor(control_spacing_range).reshape(self._dim, 2)
        control_origin_range = expand_range_arg(control_origin, dim=self._dim, negate_lower=True)
        assert len(control_origin_range) == 2 * self._dim, f"Expected 'control_origin' of length {2 * self._dim}, got {len(control_origin_range)}."
        self.__control_origin_range = to_tensor(control_origin_range).reshape(self._dim, 2)
        disp_range = expand_range_arg(displacement, dim=self._dim, negate_lower=True)
        assert len(disp_range) == 2 * self._dim, f"Expected 'displacement' of length {2 * self._dim}, got {len(disp_range)}."
        self.__disp_range = to_tensor(disp_range).reshape(self._dim, 2)
        self.__use_batching = use_batching
        self.__batching_mem_p = batching_mem_p
        self.__n_iter_max = n_iter_max
        self.__warn_folding()
        self._params = dict(
            batching_mem_p=self.__batching_mem_p,
            control_origin=self.__control_origin_range,
            control_spacing=self.__control_spacing_range,
            dim=self._dim,
            displacement=self.__disp_range,
            method=self.__method,
            n_iter_max=self.__n_iter_max,
            p=self._p,
            type=self.__class__.__name__,
            use_batching=self.__use_batching,
        )

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
            batching_mem_p=self.__batching_mem_p,
            control_origin=control_origin_draw,
            control_spacing=control_spacing_draw,
            displacement=self.__disp_range.flatten(),
            method=self.__method,
            n_iter_max=self.__n_iter_max,
            seed=seed_draw,
            use_batching=self.__use_batching,
        )
        return super().freeze(Elastic, params)

    def __str__(self) -> str:
        params = dict(
            batching_mem_p=self.__batching_mem_p,
            control_origin=to_tuple(self.__control_origin_range.flatten(), decimals=3),
            control_spacing=to_tuple(self.__control_spacing_range.flatten(), decimals=3),
            displacement=to_tuple(self.__disp_range.flatten(), decimals=3),
            method=self.__method,
            n_iter_max=self.__n_iter_max,
            use_batching=self.__use_batching,
        )
        return super().__str__(self.__class__.__name__, params)

    def __warn_folding(self, control_spacing: torch.Tensor | None = None) -> None:
        if control_spacing is None:
            control_spacing, _ = self.__control_spacing_range.min(axis=1)
        disp_widths = self.__disp_range[:, 1] - self.__disp_range[:, 0]
        if (disp_widths >= control_spacing).any():
            logger.warning(f"RandomElastic transforms with displacement widths ({to_tuple(disp_widths)}) >= "
                f"control spacings ({to_tuple(control_spacing)}) may produce folding transforms. Such transforms may "
                f"be non-invertible and could raise errors when performing forward points transform.")
