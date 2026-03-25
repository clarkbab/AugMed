#!/usr/bin/env python3
"""
Profile augmed transform pipelines to identify performance bottlenecks.

Runs the same pipeline configurations used in benchmark.py through Python's
cProfile and also provides a phase-level timing breakdown of the augmed
Pipeline internals (freeze, grid generation, backward-transform points,
grid_sample resampling, intensity transforms).

Usage
-----
    python scripts/performance/profile.py
    python scripts/performance/profile.py --dim 3 --device cpu --pipeline crop+affine+elastic+minmax
    python scripts/performance/profile.py --input image-labels --top 30
    python scripts/performance/profile.py --dim 2 --device cuda

Output
------
For each configuration the script prints:
1. A phase-level breakdown (time per stage inside the pipeline).
2. Top cProfile entries sorted by cumulative time.
3. An optional .prof file for loading in snakeviz / py-spy / etc.
"""

import argparse
from datetime import datetime
import gc
import io
import pandas as pd
from pathlib import Path
import pstats
import sys
import time
import torch
from typing import Any, Dict, List

# Temporarily remove script directory from sys.path so that this file
# (profile.py) doesn't shadow the stdlib 'profile' module that cProfile needs.
_script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _script_dir:
    sys.path.pop(0)
    import cProfile
    sys.path.insert(0, _script_dir)
else:
    import cProfile

# ---------------------------------------------------------------------------
# Shared configuration (mirrors benchmark.py)
# ---------------------------------------------------------------------------

PIPELINE_DEFS = {
    'crop+affine+minmax': ['crop', 'affine', 'minmax'],
    'crop+affine+elastic+minmax': ['crop', 'affine', 'elastic', 'minmax'],
}

CROP_REMOVE_MM = 20.0
ROTATION_DEG = 15.0
SCALING_RANGE = (0.8, 1.2)
TRANSLATION_MM = 20.0
ELASTIC_SPACING_MM = 50.0
ELASTIC_DISP_MM = 20.0
MINMAX_RANGE = (0.0, 1.0)

INPUTS = ['image', 'image-labels', 'image-labels-points']
DIMS = [2, 3]


# ---------------------------------------------------------------------------
# Data loading (same as benchmark.py)
# ---------------------------------------------------------------------------

def load_data(device: torch.device, dim: int) -> Dict[str, Any]:
    if dim == 3:
        return _load_data_3d(device)
    return _load_data_2d(device)


def _load_data_2d(device: torch.device) -> Dict[str, Any]:
    """Load the example CT and extract the middle axial slice."""
    try:
        from augmed.data.examples import load_example_ct
        ct_3d, affine_3d, labels_3d, points_3d = load_example_ct()

        # Middle axial slice.
        mid = ct_3d.shape[2] // 2
        ct_data = torch.as_tensor(ct_3d[:, :, mid], device=device, dtype=torch.float32)
        labels = torch.as_tensor(labels_3d[..., mid], device=device, dtype=torch.bool)

        # 2D affine: rows/cols 0-1 of the 3D affine, plus translation.
        a = torch.as_tensor(affine_3d, device=device, dtype=torch.float32)
        affine = torch.zeros(3, 3, device=device, dtype=torch.float32)
        affine[:2, :2] = a[:2, :2]
        affine[:2, 2] = a[:2, 3]
        affine[2, 2] = 1.0

        # Points: filter those near the middle slice, keep only x/y.
        p = torch.as_tensor(points_3d, device=device, dtype=torch.float32)
        slice_z = float(a[2, 3]) + mid * float(a[2, 2])
        half_thick = abs(float(a[2, 2])) / 2
        mask = (p[:, 2] >= slice_z - half_thick) & (p[:, 2] <= slice_z + half_thick)
        points = p[mask][:, :2]
        if len(points) == 0:
            points = p[:, :2]  # fallback: just drop Z
    except Exception:
        print('  [info] load_example_ct() unavailable — using synthetic 2D data.')
        size = (256, 256)
        spacing = (1.0, 1.0)
        ct_data = torch.randn(size, device=device, dtype=torch.float32) * 400 + 40
        affine = torch.eye(3, device=device, dtype=torch.float32)
        for i in range(2):
            affine[i, i] = spacing[i]
        labels = (ct_data > 200).unsqueeze(0)
        n_points = 500
        points = torch.rand(n_points, 2, device=device) * torch.tensor(
            [s * d for s, d in zip(spacing, size)], device=device,
        )
    return dict(affine=affine, ct_data=ct_data, labels=labels, points=points)


def _load_data_3d(device: torch.device) -> Dict[str, Any]:
    try:
        from augmed.data.examples import load_example_ct
        ct_data, affine, labels, points = load_example_ct()
        ct_data = torch.as_tensor(ct_data, device=device, dtype=torch.float32)
        affine = torch.as_tensor(affine, device=device, dtype=torch.float32)
        labels = torch.as_tensor(labels, device=device, dtype=torch.bool)
        points = torch.as_tensor(points, device=device, dtype=torch.float32)
    except Exception:
        print('  [info] load_example_ct() unavailable — using synthetic data.')
        size = (256, 256, 128)
        spacing = (1.0, 1.0, 2.5)
        ct_data = torch.randn(size, device=device, dtype=torch.float32) * 400 + 40
        affine = torch.eye(4, device=device, dtype=torch.float32)
        for i in range(3):
            affine[i, i] = spacing[i]
        labels = (ct_data > 200).unsqueeze(0)
        n_points = 500
        points = torch.rand(n_points, 3, device=device) * torch.tensor(
            [s * d for s, d in zip(spacing, size)], device=device,
        )
    return dict(affine=affine, ct_data=ct_data, labels=labels, points=points)


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def build_pipeline(steps: List[str], *, device: torch.device, dim: int):
    from augmed import (
        MinMax,
        Pipeline,
        RandomAffine,
        RandomCrop,
        RandomElastic,
    )
    transforms = []
    for s in steps:
        if s == 'crop':
            transforms.append(RandomCrop(dim=dim, remove=CROP_REMOVE_MM))
        elif s == 'affine':
            transforms.append(RandomAffine(
                dim=dim,
                rotation=ROTATION_DEG,
                scaling=SCALING_RANGE,
                translation=TRANSLATION_MM,
            ))
        elif s == 'elastic':
            transforms.append(RandomElastic(
                control_spacing=ELASTIC_SPACING_MM,
                dim=dim,
                displacement=ELASTIC_DISP_MM,
            ))
        elif s == 'minmax':
            transforms.append(MinMax(dim=dim, max=MINMAX_RANGE[1], min=MINMAX_RANGE[0]))
    return Pipeline(transforms, device=device, dim=dim, seed=42)


def build_inputs(data: Dict[str, Any], input_mode: str) -> tuple:
    inputs = [data['ct_data']]
    if input_mode in ('image-labels', 'image-labels-points'):
        inputs.append(data['labels'])
    if input_mode == 'image-labels-points':
        inputs.append(data['points'])
    return tuple(inputs)


# ---------------------------------------------------------------------------
# Phase-level timing (instruments the internal pipeline stages)
# ---------------------------------------------------------------------------

def phase_profile(
    steps: List[str],
    *,
    data: Dict[str, Any],
    device: torch.device,
    dim: int,
    input_mode: str,
    n_runs: int,
) -> Dict[str, float]:
    """Time each major phase of the augmed pipeline over *n_runs* iterations.

    Phases measured:
      - freeze:   random sampling + pipeline construction
      - grid_pts: grid_points() — generating the dense coordinate grid
      - backward: backward_transform_points — affine chains + elastic interp
      - resample: grid_sample — the actual torch.nn.functional.grid_sample call
      - intensity: intensity transform application (e.g. MinMax)
      - total:    full end-to-end time
    """
    from augmed.transforms.pipeline import FrozenPipeline
    from augmed.utils.conversion import to_tensor, to_tuple
    from augmed.utils.grid import grid_points, grid_sample

    # Monkey-patch key functions to capture timing without changing behaviour.
    # We accumulate times per phase across runs for averaging.
    accum = {
        'freeze': 0.0,
        'grid_pts': 0.0,
        'backward': 0.0,
        'resample': 0.0,
        'intensity': 0.0,
        'total': 0.0,
    }

    pipeline = build_pipeline(steps, device=device, dim=dim)
    inputs = build_inputs(data, input_mode)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    for _ in range(n_runs):
        gc.collect()
        sync()

        t_total_start = time.perf_counter()

        # Phase 1: freeze (random parameter sampling)
        sync()
        t0 = time.perf_counter()
        frozen = pipeline.freeze()
        sync()
        accum['freeze'] += time.perf_counter() - t0

        # Now run the frozen pipeline's image transform logic manually to
        # time each sub-phase.  We replicate the essential steps of
        # FrozenPipeline.transform_images.
        images = list(inputs)
        # Separate images and non-images (points).
        img_list = []
        points_input = None
        for inp in images:
            if isinstance(inp, torch.Tensor) and inp.dim() >= dim:
                # Could be image or labels — check if it has more elements
                # than a point cloud would for this dim.
                if inp.ndim == dim or (inp.ndim == dim + 1 and inp.shape[0] < 10):
                    img_list.append(inp)
                else:
                    img_list.append(inp)
            else:
                points_input = inp

        # Just run the full transform for an end-to-end time, but also
        # get a finer breakdown via cProfile for the sub-functions.
        frozen.transform(*inputs, affine=data['affine'])
        sync()
        accum['total'] += time.perf_counter() - t_total_start

    # Average
    return {k: v / n_runs for k, v in accum.items()}


def phase_profile_detailed(
    steps: List[str],
    *,
    data: Dict[str, Any],
    device: torch.device,
    dim: int,
    input_mode: str,
    n_runs: int,
) -> Dict[str, float]:
    """Detailed phase profiling by instrumenting augmed internals.

    Uses wrapper functions around grid_points, grid_sample, and the backward
    transform to measure time spent in each phase.
    """
    import augmed.utils.grid as grid_mod
    from augmed.transforms.pipeline import FrozenPipeline

    accum = {
        'freeze': 0.0,
        'grid_pts': 0.0,
        'backward': 0.0,
        'resample': 0.0,
        'intensity': 0.0,
        'total': 0.0,
    }

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    pipeline = build_pipeline(steps, device=device, dim=dim)
    inputs = build_inputs(data, input_mode)

    # Save originals.
    orig_grid_points = grid_mod.grid_points
    orig_grid_sample = grid_mod.grid_sample

    for _ in range(n_runs):
        gc.collect()
        sync()

        # Timing accumulators for this run, gathered via monkey-patching.
        run_gp = [0.0]
        run_gs = [0.0]

        def timed_grid_points(*a, **kw):
            sync()
            t = time.perf_counter()
            r = orig_grid_points(*a, **kw)
            sync()
            run_gp[0] += time.perf_counter() - t
            return r

        def timed_grid_sample(*a, **kw):
            sync()
            t = time.perf_counter()
            r = orig_grid_sample(*a, **kw)
            sync()
            run_gs[0] += time.perf_counter() - t
            return r

        # Patch.
        grid_mod.grid_points = timed_grid_points
        grid_mod.grid_sample = timed_grid_sample
        # Also need to patch the import in the pipeline module.
        import augmed.transforms.pipeline as pipe_mod
        pipe_mod.grid_points = timed_grid_points
        pipe_mod.grid_sample = timed_grid_sample

        t_total = time.perf_counter()

        # Freeze phase.
        sync()
        t0 = time.perf_counter()
        frozen = pipeline.freeze()
        sync()
        accum['freeze'] += time.perf_counter() - t0

        # Run the full transform.
        frozen.transform(*inputs, affine=data['affine'])
        sync()
        accum['total'] += time.perf_counter() - t_total

        accum['grid_pts'] += run_gp[0]
        accum['resample'] += run_gs[0]

        # Restore originals.
        grid_mod.grid_points = orig_grid_points
        grid_mod.grid_sample = orig_grid_sample
        pipe_mod.grid_points = orig_grid_points
        pipe_mod.grid_sample = orig_grid_sample

    # Compute derived phases.
    avg = {k: v / n_runs for k, v in accum.items()}
    # backward ≈ total - freeze - grid_pts - resample - intensity
    # intensity is small for MinMax but we can approximate it.
    avg['backward'] = max(0.0, avg['total'] - avg['freeze'] - avg['grid_pts'] - avg['resample'])
    return avg


# ---------------------------------------------------------------------------
# cProfile wrapper
# ---------------------------------------------------------------------------

def cprofile_run(
    steps: List[str],
    *,
    data: Dict[str, Any],
    device: torch.device,
    dim: int,
    input_mode: str,
    n_runs: int,
    save_prof: str | None = None,
    top: int = 40,
) -> str:
    """Run augmed pipeline under cProfile and print/return top functions."""
    pipeline = build_pipeline(steps, device=device, dim=dim)
    inputs = build_inputs(data, input_mode)

    def run():
        for _ in range(n_runs):
            pipeline.transform(*inputs, affine=data['affine'])
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # Warmup.
    pipeline.transform(*inputs, affine=data['affine'])
    if device.type == 'cuda':
        torch.cuda.synchronize()

    prof = cProfile.Profile()
    prof.enable()
    run()
    prof.disable()

    if save_prof:
        prof.dump_stats(save_prof)
        print(f'  Saved profile to {save_prof}')

    # Print summary.
    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(top)
    print(s.getvalue())

    # Also print a callers view for the hot functions.
    s2 = io.StringIO()
    ps2 = pstats.Stats(prof, stream=s2)
    ps2.sort_stats('tottime')
    ps2.print_stats(20)
    print('\n--- Top by total time ---')
    print(s2.getvalue())

    return s.getvalue() + '\n--- Top by total time ---\n' + s2.getvalue()


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Profile augmed transform pipelines.',
    )
    parser.add_argument(
        '--n-runs', default=5, help='Number of profiled runs per config (default: 5)',
        type=int,
    )
    parser.add_argument(
        '--output', default=None, help='Output CSV path (default: profile_<timestamp>.csv next to this script)',
        type=str,
    )
    parser.add_argument(
        '--save-prof', action='store_true',
        help='Save .prof files for external tools (snakeviz, etc.)',
    )
    parser.add_argument(
        '--top', default=40, help='Number of top cProfile entries to show (default: 40)',
        type=int,
    )
    args = parser.parse_args()

    n_runs = args.n_runs

    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices.append(torch.device('cuda'))

    output = args.output
    if output is None:
        ts_file = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = str(Path(__file__).parent / f'profile_{ts_file}.csv')

    # Text file for detailed cProfile output (same base name as CSV).
    output_txt = output.rsplit('.', 1)[0] + '.txt'

    print(f'Profiling augmed: n_runs={n_runs}')
    wall_t0 = time.perf_counter()

    rows = []
    profile_sections: List[str] = []

    for device in devices:
        for dim in DIMS:
            print(f'\n{"=" * 70}')
            print(f'Device: {device}  |  Dimensionality: {dim}D')
            print(f'{"=" * 70}')

            data = load_data(device, dim)
            ct_shape = tuple(data['ct_data'].shape)
            print(f'  data: ct_data={ct_shape}, labels={tuple(data["labels"].shape)}, '
                  f'points={tuple(data["points"].shape)}')

            for pipe_name, steps in PIPELINE_DEFS.items():
                for input_mode in INPUTS:
                    label = f'{pipe_name} / {input_mode} / {dim}D / {device}'
                    print(f'\n  {label}')

                    # --- Phase-level timing ---
                    phases = phase_profile_detailed(
                        steps,
                        data=data,
                        device=device,
                        dim=dim,
                        input_mode=input_mode,
                        n_runs=n_runs,
                    )

                    total = phases['total']
                    for name in ['freeze', 'grid_pts', 'backward', 'resample']:
                        t = phases[name]
                        pct = 100 * t / total if total > 0 else 0
                        print(f'    {name:12s}: {t:8.4f}s  ({pct:5.1f}%)')
                    print(f'    {"total":12s}: {total:8.4f}s')

                    rows.append(dict(
                        device=str(device),
                        dim=dim,
                        input=input_mode,
                        n_runs=n_runs,
                        pipeline=pipe_name,
                        **{f'time_{k}': round(v, 6) for k, v in phases.items()},
                    ))

                    # --- cProfile ---
                    print(f'    --- cProfile (cumulative, top {args.top}) ---')
                    prof_path = None
                    if args.save_prof:
                        ts_prof = datetime.now().strftime('%Y%m%d_%H%M%S')
                        prof_path = str(Path(__file__).parent / f'profile_{pipe_name}_{dim}d_{device}_{input_mode}_{ts_prof}.prof')

                    profile_text = cprofile_run(
                        steps,
                        data=data,
                        device=device,
                        dim=dim,
                        input_mode=input_mode,
                        n_runs=n_runs,
                        save_prof=prof_path,
                        top=args.top,
                    )
                    profile_sections.append(
                        f'{"/" * 70}\n'
                        f'{label}\n'
                        f'{"/" * 70}\n\n'
                        f'{profile_text}\n'
                    )

    # Save results CSV.
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)

    # Save detailed cProfile output.
    with open(output_txt, 'w') as f:
        f.write('\n'.join(profile_sections))

    wall_elapsed = time.perf_counter() - wall_t0
    print(f'\nResults saved to {output}')
    print(f'cProfile details saved to {output_txt}')
    print(f'Total wall-clock time: {wall_elapsed:.1f}s')


if __name__ == '__main__':
    main()
