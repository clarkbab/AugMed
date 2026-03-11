import struct
from typing import *

from ...typing import *
from ...utils.logging import logger
from ..identity import Identity
from .spatial import RandomSpatialTransform, SpatialTransform

# control spacing:
# - c=5 -> c=(5, 5, 5, 5) for 2D, c=(5, 5, 5, 5, 5, 5) for 3D. I.e deterministic control spacing.
# - c=(3, 5) -> c=(3, 5, 3, 5) for 2D, c=(3, 5, 3, 5, 3, 5) for 3D.
# disp:
# - d=5 -> d=(-5, 5, -5, 5) for 2D, d=(-5, 5, -5, 5, -5, 5) for 3D.
# - d=(3, 5) -> d=(3, 5, 3, 5) for 2D, d=(3, 5, 3, 5, 3, 5) for 3D.
# control origin:
# - c=5 -> c=(-5, 5, -5, 5) for 2D, c=(-5, 5, -5, 5, -5, 5) for 3D.
# - c=(3, 5) -> c=(3, 5, 3, 5) for 2D, c=(3, 5, 3, 5, 3, 5) for 3D.
class RandomElastic(RandomSpatialTransform):
    def __init__(
        self, 
        # How do we deal with elastic deformations that may fold?
        #   - prevent by requiring 'disp' to be < half of the control spacing.
        #   - allow but warn that folding may occur and forward point transforms may not converge.
        #      will also need to raise error upon no convergence.
        # 1. Control spacing/origin defines the grid of control points.
        # 2. Displacement controls the deformation vector magnitudes.
        control_spacing: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 50.0,
        control_origin: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 20.0,
        displacement: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 20.0,
        # Would other curve-fitting methods have desirable properties for certain situations?
        # E.g. 'bezier' allows sharp discontinuities which could be good for shearing in lung.
        # Can we randomise the curve-fitting method to increase the space of data augmentation?
        # This is the method for interpolating between points on the coarse control grid.
        interp_method: Literal['bspline', 'cubic', 'linear', 'linear-gs'] = 'linear',
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        self.__interp_method = interp_method
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
            interp_method=self.__interp_method,
            p=self._p,
        )

        # Warn about displacement ranges.
        disp_widths = self.__disp_range[:, 1] - self.__disp_range[:, 0]
        min_control_spacing, _ = self.__control_spacing_range.min(axis=1)
        if (disp_widths >= min_control_spacing).any():
            logger.warning(f"RandomElastic transforms with larger displacement widths (widths={to_tuple(disp_widths)}) larger than \
(or equal to) control spacings (min. spacings={to_tuple(min_control_spacing)}) may produce folding transforms. Such transforms may \
be non-invertible and could raise errors when performing forward points transform.")

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

# Defines a coarse grid of control points.
# Random displacements are assigned at each control point.
# A b-spline is fitted (input: control point locations, output: perturbed control point positions)
# and this b-spline is used as the back transform.
# Cubic b-splines require a min of 4 control points per axis.
# Define max displacement, with some reasonable default.
# An option to set border displacements to zero. Maybe?

# control spacing:
# - c=5 -> c=(5, 5) for 2D, c=(5, 5, 5) for 3D.
# disp:
# - d=5 -> d=(-5, 5, -5, 5) for 2D, d=(-5, 5, -5, 5, -5, 5) for 3D.
# - d=(3, 5) -> d=(3, 5, 3, 5) for 2D, d=(3, 5, 3, 5, 3, 5) for 3D.
# control origin:
# - c=5 -> c=(5, 5) for 2D, c=(5, 5, 5) for 3D.
class Elastic(SpatialTransform):
    def __init__(
        self,
        control_spacing: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 50.0,
        control_origin: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 20.0,
        displacement: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor = 20.0,
        interp_method: Literal['bspline', 'cubic', 'linear'] = 'linear',
        seed: int = 42,
        **kwargs,
        ) -> None:
        super().__init__(**kwargs)
        assert interp_method in ['bspline', 'cubic', 'linear', 'linear-gs'], "Only 'bspline', 'cubic', 'linear', and 'linear-gs' elastic methods are supported."
        self.__interp_method = interp_method
        self.__control_spacing = to_tensor(control_spacing, broadcast=self._dim)
        assert len(self.__control_spacing) == self._dim, f"Expected 'control_spacing' of length '{self._dim}' for dim={self._dim}, got {len(self.__control_spacing)}."
        self.__control_origin = to_tensor(control_origin, broadcast=self._dim)
        assert len(self.__control_origin) == self._dim, f"Expected 'control_origin' of length '{self._dim}' for dim={self._dim}, got {len(self.__control_origin)}."
        # Disps aren't known until presented with the image.
        disp_range = expand_range_arg(displacement, dim=self._dim, negate_lower=True)
        assert len(disp_range) == 2 * self._dim, f"Expected 'displacement' of length {2 * self._dim}, got {len(disp_range)}."
        self.__disp_range = to_tensor(disp_range).reshape(self._dim, 2)
        self.__seed = seed
        self._params = dict(
            type=self.__class__.__name__,
            control_origin=self.__control_origin,
            control_spacing=self.__control_spacing,
            dim=self._dim,
            displacement=self.__disp_range,
            method=self.__interp_method,
            seed=self.__seed,
        )

    def backward_transform_points(
        self,
        points: PointsTensor,
        interp_method: Optional[Literal['bspline', 'cubic', 'linear', 'linear-gs']] = None,
        **kwargs,
        ) -> PointsTensor:
        method = self.__interp_method if interp_method is None else interp_method
        if method == 'bspline':
            return self.__backward_transform_points_bspline(points, **kwargs)
        elif method == 'cubic':
            return self.__backward_transform_points_cubic(points, **kwargs)
        elif method == 'linear':
            return self.__backward_transform_points_linear(points, **kwargs)
        elif method == 'linear-gs':
            return self.__backward_transform_points_linear_gs(points, **kwargs)
        else:
            raise ValueError(f"Unrecognised elastic method '{method}'.")

    def __backward_transform_points_cubic(
        self,
        points: PointsTensor,
        **kwargs,
        ) -> PointsTensor:
        # Get control grid.
        # Make this just large enough to interpolate all 'points'.
        _, cp_disps, cp_spacing, cp_origin = self.control_grid(points, method='cubic')
        cp_disps = torch.moveaxis(cp_disps, 0, -1)  # Move channels dim to back.

        # Normalise points to the control grid integer coords.
        points_norm = (points - cp_origin) / cp_spacing

        # Get lowest corner point.
        corner_min = torch.stack([torch.searchsorted(torch.arange(cp_disps.shape[a]).to(points.device), points_norm[:, a]) - 1 for a in range(self._dim)], dim=-1)

        # Get distances from corner.
        u = points_norm - corner_min
        b = torch.stack([1 - u, u], dim=-2)

        # Get corner point offsets.
        offsets = torch.stack(torch.meshgrid([torch.tensor([0, 1]) for _ in range(self._dim)], indexing='ij'), dim=-1)
        offsets = offsets.reshape(-1, self._dim).to(points.device)

        # Calculate corners for each point.
        corners = corner_min[:, None, :] + offsets[None, :, :]

        # Split into x/y/z indices to perform control point disp selection.
        idxs = corners.unbind(-1)
        corner_disps = cp_disps[*idxs]

        # Create V of corner point displacements.
        V = corner_disps.reshape(-1, *(2, ) * self._dim, self._dim)

        # Compute interpolated displacements.
        if self._dim == 2:
            disps = torch.einsum('ni,nj,nijd->nd', b[:, :, 0], b[:, :, 1], V)
        elif self._dim == 3:
            disps = torch.einsum('ni,nj,nk,nijkd->nd', b[:, :, 0], b[:, :, 1], b[:, :, 2], V)

        # Get displaced input points.
        points_t = points + disps

        return points_t

    def __backward_transform_points_linear(
        self,
        points: PointsTensor,
        **kwargs,
        ) -> PointsTensor:
        print('linear elastic')
        print(points.dtype)

        # Get control grid.
        # Make this just large enough to interpolate all 'points'.
        _, cp_disps, cp_spacing, cp_origin = self.control_grid(points, method='linear')
        cp_disps = torch.moveaxis(cp_disps, 0, -1)  # Move channels dim to back.
        print(cp_disps.dtype, cp_spacing.dtype, cp_origin.dtype)

        # Normalise points to the control grid integer coords.
        points_norm = (points - cp_origin) / cp_spacing
        print(points_norm.dtype)

        # Get lowest corner point.
        corner_min = torch.stack([torch.searchsorted(torch.arange(cp_disps.shape[a]).to(points.device), points_norm[:, a]) - 1 for a in range(self._dim)], dim=-1)
        print(corner_min.dtype)

        # Get distances from corner.
        u = points_norm - corner_min
        b = torch.stack([1 - u, u], dim=-2)
        print(u.dtype, b.dtype)

        # Get corner point offsets.
        offsets = torch.stack(torch.meshgrid([torch.tensor([0, 1]) for _ in range(self._dim)], indexing='ij'), dim=-1)
        offsets = offsets.reshape(-1, self._dim).to(points.device)
        print(offsets.dtype)

        # Calculate corners for each point.
        corners = corner_min[:, None, :] + offsets[None, :, :]
        print(corners.dtype)

        # Split into x/y/z indices to perform control point disp selection.
        idxs = corners.unbind(-1)
        corner_disps = cp_disps[*idxs]

        # Create V of corner point displacements.
        V = corner_disps.reshape(-1, *(2, ) * self._dim, self._dim)

        # Compute interpolated displacements.
        print('elastic types')
        print(b.dtype)
        print(V.dtype)
        if self._dim == 2:
            disps = torch.einsum('ni,nj,nijd->nd', b[:, :, 0], b[:, :, 1], V)
        elif self._dim == 3:
            disps = torch.einsum('ni,nj,nk,nijkd->nd', b[:, :, 0], b[:, :, 1], b[:, :, 2], V)
        print(disps.dtype)

        # Get displaced input points.
        points_t = points + disps

        return points_t

    def __backward_transform_points_linear_gs(
        self,
        points: PointsTensor,
        **kwargs,
        ) -> PointsTensor:
        # Get control grid.
        _, cp_disps, cp_spacing, cp_origin = self.control_grid(points, method='linear')

        # Interpolate the displacement grid.
        disps = grid_sample(cp_disps, cp_spacing, cp_origin, points, dim=self._dim)
        disps = torch.moveaxis(disps, 0, -1)[0, 0]    # Disps come back from 'grid_sample' as 3-channel image.

        # Get displaced input points.
        points_t = points + disps

        return points_t
        
    # For the same location in a control grid (set spacing, origin) we should always get the
    # same displacement for the same 'seed'. This is so that subsequent calls to the
    # transform methods will return the same results.
    def __control_point_seed(
        self,
        point: PointsTensor,
        ) -> int:
        primes = (73856093, 19349663, 83492791)[:self._dim]
        def f2i(f: float) -> int:   
            return struct.unpack('<I', struct.pack('<f', float(f)))[0]
        point = [f2i(f) for f in point]
        h = point[0] * primes[0]
        for p, pr in zip(point[1:], primes[1:]):
            h = h ^ (p * pr)
        h = h ^ self.__seed
        return h & 0x7fffffff

    # The control grid random displacements must not change depending on the passed points.
    # That is, if we pass the point (0.5, 0.5, 0.5) this must give the same transformed
    # point regardless of the other points we pass.
    # This means that subsequent calls to transform_points and backward_transform_points
    # will align points and images properly.
    def control_grid(
        self,
        points: PointsTensor,
        interp_method: Literal['bspline', 'cubic', 'linear'] | None = None,
        ) -> Tuple[ImageTensor, ChannelImageTensor, AffineTensor]:
        if isinstance(points, torch.Tensor):
            return_type = 'torch'
        else:
            points = to_tensor(points)
            return_type = 'numpy'

        print('control grid')

        # Get the origin/spacing for this point cloud.
        cp_spacing = self.__control_spacing.to(points.device)
        cp_global_origin = self.__control_origin.to(points.device)
        point_min, _ = points.min(dim=0)
        point_max, _ = points.max(dim=0)
        cp_idx_min = torch.floor((point_min - cp_global_origin) / cp_spacing)
        cp_idx_max = torch.ceil((point_max - cp_global_origin) / cp_spacing)
        method = self.__method if interp_method is None else interp_method
        if method == 'cubic':
            # Add extra boundary points for cubic splines.
            cp_idx_min -= 1
            cp_idx_max += 1
        cp_origin = cp_idx_min * cp_spacing + cp_global_origin

        # Create grid of control points.
        cps = torch.stack(torch.meshgrid([
            torch.arange(cp_idx_min[a].item(), cp_idx_max[a].item() + 1) for a in range(self._dim)
        ], indexing='ij'), dim=-1)
        cps = to_tensor(cps, device=points.device)
        cps = cps * cp_spacing + cp_global_origin

        # Generate reproducible displacements.
        cp_points = cps.reshape(-1, self._dim)
        draws = []
        for p in cp_points:
            seed = self.__control_point_seed(p)
            rng = np.random.default_rng(seed=seed)
            draw = to_tensor(rng.random(self._dim), device=points.device).type(torch.float32)
            draws.append(draw)
        draws = torch.stack(draws).reshape(*cps.shape[:-1], self._dim)
        disp_range = self.__disp_range.to(points.device)
        cp_disps = draws * (disp_range[:, 1] - disp_range[:, 0]) + disp_range[:, 0]

        # Bring channels dimension to fore as these are images - i.e. not just an Nx2/3 list of points.
        cps, cp_disps = torch.moveaxis(cps, -1, 0), torch.moveaxis(cp_disps, -1, 0)

        cp_affine = create_affine(cp_spacing, cp_origin)
        if return_type == 'numpy':
            cps, cp_disps, cp_affine = to_array(cps), to_array(cp_disps), to_array(cp_affine)
        
        return cps, cp_disps, cp_affine

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
        interp_method: Optional[Literal['bspline', 'cubic', 'linear', 'linear-gs']] = None,
        **kwargs,
        ) -> Points:
        if isinstance(points, np.ndarray):
            points = to_tensor(points)
            return_type = 'numpy'
        else:
            return_type = 'torch'

        # Get the back transform.
        method = self.__interp_method if interp_method is None else interp_method
        if method == 'bspline':
            backward_transform_fn = self.__backward_transform_points_bspline
        elif method == 'cubic':
            backward_transform_fn = self.__backward_transform_points_cubic
        elif method == 'linear':
            backward_transform_fn = self.__backward_transform_points_linear
        elif method == 'linear-gs':
            backward_transform_fn = self.__backward_transform_points_linear_gs
        else:
            raise ValueError(f"Unrecognised elastic method '{method}'.")

        # Let: T_back(x) = x + u(x), where x is the point to find (fixed image) and T_back is known.
        # Let: F(x) = T_back(x) - y, where y is the target point we know (moving image).
        # Solve for F(x) = 0 using an iterative method, e.g. Newton-Raphson.
        max_i = 100     # Log the required number of iterations for solve and adjust.
        x_i = points.clone().requires_grad_()
        for i in range(max_i):
            # Perform transform.
            t_x = backward_transform_fn(x_i)

            # Check convergence.
            print(t_x.dtype, points.dtype)
            if torch.isclose(t_x, points).all():
                print('Newton-Raphson for inverse transform converged after iterations: ', i)
                break
            elif i == max_i - 1:
                raise ValueError('No convergence after iterations: ', i)

            # Get Jacobians for batch of points.
            grads = []
            for a in range(self._dim):
                grad_a, = torch.autograd.grad(t_x[:, a], x_i, grad_outputs=torch.ones(len(x_i)).to(x_i.device), retain_graph=True)
                grads.append(grad_a)
            J = torch.stack(grads, dim=1)

            # Batch solve for deltas for each point.
            r = t_x - points
            dx = torch.linalg.solve(J, r)

            # Update guess.
            x_i = x_i.detach()  # How does it get 'requires_grad_' again, must be through 'dx'.
            x_i = x_i - dx

        x_i = x_i.detach()
        if return_type == 'numpy':
            x_i = to_array(x_i)

        return x_i
