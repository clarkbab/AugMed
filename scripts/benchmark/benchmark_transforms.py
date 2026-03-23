#!/usr/bin/env python3
"""
Benchmark augmed vs MONAI vs TorchIO for two augmentation pipelines.

Pipelines
---------
1. Crop → Affine → MinMax normalisation
2. Crop → Affine → Elastic → MinMax normalisation

Data modes
----------
- ``image``                — CT volume only
- ``image-labels``         — CT volume + label masks
- ``image-labels-points``  — CT volume + label masks + point cloud (augmed only)

Dimensionality
--------------
- ``2d`` — single-slice images  (H, W)
- ``3d`` — volumetric images    (D, H, W)

Each combination of (library × pipeline × data_mode × ndim × device) is timed
with peak RAM / GPU VRAM recorded.  Results are saved to CSV.

Usage
-----
    python scripts/benchmark/benchmark_transforms.py
    python scripts/benchmark/benchmark_transforms.py --n-runs 5 --warmup 2
    python scripts/benchmark/benchmark_transforms.py --output results.csv

Prerequisites
-------------
    pip install augmed monai torchio pandas
"""

import argparse
from datetime import datetime
import gc
import numpy as np
import pandas as pd
from pathlib import Path
import time
import torch
import tracemalloc
from typing import Any, Callable, Dict, List

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PIPELINE_DEFS = {
    'crop+affine+minmax': ['crop', 'affine', 'minmax'],
    'crop+affine+elastic+minmax': ['crop', 'affine', 'elastic', 'minmax'],
}

INPUTS = ['image', 'image-labels', 'image-labels-points']

# Which inputs each library supports.
LIBRARY_INPUTS = {
    'augmed': ['image', 'image-labels', 'image-labels-points'],
    'monai': ['image', 'image-labels'],
    'torchio': ['image', 'image-labels'],
}

DIMS = [2, 3]

# Which dimensionalities each library supports.
LIBRARY_DIMS = {
    'augmed': [2, 3],
    'monai': [2, 3],
    'torchio': [2, 3],         # 2-D slices promoted to (C, 1, H, W) pseudo-volumes.
}

# Shared transform parameters that are as close as possible across
# libraries so the workloads are comparable.
CROP_REMOVE_MM = 20.0           # mm to remove per side
ROTATION_DEG = 15.0             # max rotation per axis in degrees
SCALING_RANGE = (0.8, 1.2)
TRANSLATION_MM = 20.0           # max translation per axis in mm
ELASTIC_SPACING_MM = 50.0       # control-point spacing
ELASTIC_DISP_MM = 20.0          # max displacement
MINMAX_RANGE = (0.0, 1.0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(device: torch.device, dim: int) -> Dict[str, Any]:
    """Load (or generate) representative data for the given dimensionality."""
    if dim == 3:
        return _load_data_3d(device)
    return _load_data_2d(device)


def _load_data_2d(device: torch.device) -> Dict[str, Any]:
    """Generate a synthetic 2-D slice."""
    size = (256, 256)
    spacing = (1.0, 1.0)
    ct_data = torch.randn(size, device=device, dtype=torch.float32) * 400 + 40
    affine = torch.eye(3, device=device, dtype=torch.float32)
    for i in range(2):
        affine[i, i] = spacing[i]
    labels = (ct_data > 200).unsqueeze(0)               # (1, H, W)
    n_points = 500
    points = torch.rand(n_points, 2, device=device) * torch.tensor(
        [s * d for s, d in zip(spacing, size)], device=device,
    )
    return dict(affine=affine, ct_data=ct_data, labels=labels, points=points)


def _load_data_3d(device: torch.device) -> Dict[str, Any]:
    """Load (or generate) a representative 3-D CT volume."""
    try:
        from augmed.utils.examples import load_example_ct
        ct_data, affine, labels, points = load_example_ct()
        ct_data = torch.as_tensor(ct_data, device=device, dtype=torch.float32)
        affine = torch.as_tensor(affine, device=device, dtype=torch.float32)
        labels = torch.as_tensor(labels, device=device, dtype=torch.bool)
        points = torch.as_tensor(points, device=device, dtype=torch.float32)
    except Exception:
        # Fallback: synthetic volume that exercises the same code paths.
        print('  [info] load_example_ct() unavailable — using synthetic data.')
        size = (256, 256, 128)
        spacing = (1.0, 1.0, 2.5)
        ct_data = torch.randn(size, device=device, dtype=torch.float32) * 400 + 40
        affine = torch.eye(4, device=device, dtype=torch.float32)
        for i in range(3):
            affine[i, i] = spacing[i]
        labels = (ct_data > 200).unsqueeze(0)           # (1, D, H, W)
        n_points = 500
        points = torch.rand(n_points, 3, device=device) * torch.tensor(
            [s * d for s, d in zip(spacing, size)], device=device,
        )
    return dict(affine=affine, ct_data=ct_data, labels=labels, points=points)


# ---------------------------------------------------------------------------
# augmed pipelines
# ---------------------------------------------------------------------------

def _build_augmed_pipeline(steps: List[str], *, device: torch.device, dim: int) -> Any:
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
        else:
            raise ValueError(f'Unknown step: {s}')

    return Pipeline(transforms, device=device, dim=dim, seed=42)


def run_augmed(
    data: Dict[str, Any],
    dim: int,
    device: torch.device,
    input_mode: str,
    steps: List[str],
) -> None:
    pipeline = _build_augmed_pipeline(steps, device=device, dim=dim)
    inputs = [data['ct_data']]
    if input_mode in ('image-labels', 'image-labels-points'):
        inputs.append(data['labels'])
    if input_mode == 'image-labels-points':
        inputs.append(data['points'])
    pipeline.transform(*inputs, affine=data['affine'])
    if device.type == 'cuda':
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# MONAI pipelines
# ---------------------------------------------------------------------------

def _build_monai_image_pipeline(
    steps: List[str],
    *,
    device: torch.device,
    dim: int,
) -> Any:
    """Build a MONAI pipeline for image-only mode (non-dict transforms)."""
    from monai.transforms import (
        Compose,
        RandAffine,
        RandSpatialCrop,
        ScaleIntensityRange,
    )

    roi_size = [156, 156] if dim == 2 else [156, 156, 78]
    n_spatial = dim
    rot_range = [np.deg2rad(ROTATION_DEG)] * (1 if dim == 2 else 3)

    transforms: list = []
    for s in steps:
        if s == 'crop':
            transforms.append(RandSpatialCrop(
                random_size=False,
                roi_size=roi_size,
            ))
        elif s == 'affine':
            transforms.append(RandAffine(
                mode='bilinear',
                prob=1.0,
                rotate_range=rot_range,
                scale_range=[(v - 1.0) for v in SCALING_RANGE],
                translate_range=[TRANSLATION_MM] * n_spatial,
            ))
        elif s == 'elastic':
            transforms.append(_monai_elastic_transform(dim=dim))
        elif s == 'minmax':
            transforms.append(ScaleIntensityRange(
                a_max=float(MINMAX_RANGE[1]),
                a_min=float(MINMAX_RANGE[0]),
                b_max=float(MINMAX_RANGE[1]),
                b_min=float(MINMAX_RANGE[0]),
                clip=True,
            ))

    return Compose(transforms)


def _build_monai_dict_pipeline(
    steps: List[str],
    *,
    device: torch.device,
    dim: int,
) -> Any:
    """Build a MONAI pipeline for image-labels mode (dict transforms)."""
    from monai.transforms import (
        Compose,
        RandAffined,
        RandSpatialCropd,
        ScaleIntensityRanged,
    )

    roi_size = [156, 156] if dim == 2 else [156, 156, 78]
    n_spatial = dim
    rot_range = [np.deg2rad(ROTATION_DEG)] * (1 if dim == 2 else 3)

    transforms: list = []
    for s in steps:
        if s == 'crop':
            transforms.append(RandSpatialCropd(
                keys=['image', 'label'],
                random_size=False,
                roi_size=roi_size,
            ))
        elif s == 'affine':
            transforms.append(RandAffined(
                keys=['image', 'label'],
                mode=['bilinear', 'nearest'],
                prob=1.0,
                rotate_range=rot_range,
                scale_range=[(v - 1.0) for v in SCALING_RANGE],
                translate_range=[TRANSLATION_MM] * n_spatial,
            ))
        elif s == 'elastic':
            transforms.append(_monai_elastic_transform_d(dim=dim))
        elif s == 'minmax':
            transforms.append(ScaleIntensityRanged(
                a_max=float(MINMAX_RANGE[1]),
                a_min=float(MINMAX_RANGE[0]),
                b_max=float(MINMAX_RANGE[1]),
                b_min=float(MINMAX_RANGE[0]),
                clip=True,
                keys=['image'],
            ))

    return Compose(transforms)


def _monai_elastic_transform(*, dim: int) -> Any:
    """Return the appropriate MONAI elastic transform for the dimensionality."""
    if dim == 2:
        from monai.transforms import Rand2DElastic
        return Rand2DElastic(
            magnitude_range=(0, ELASTIC_DISP_MM),
            mode='bilinear',
            prob=1.0,
            spacing=(30, 30),
        )
    from monai.transforms import Rand3DElastic
    return Rand3DElastic(
        magnitude_range=(0, ELASTIC_DISP_MM),
        mode='bilinear',
        prob=1.0,
        sigma_range=(5, 8),
    )


def _monai_elastic_transform_d(*, dim: int) -> Any:
    """Return the appropriate MONAI dict elastic transform."""
    if dim == 2:
        from monai.transforms import Rand2DElasticd
        return Rand2DElasticd(
            keys=['image', 'label'],
            magnitude_range=(0, ELASTIC_DISP_MM),
            mode=['bilinear', 'nearest'],
            prob=1.0,
            spacing=(30, 30),
        )
    from monai.transforms import Rand3DElasticd
    return Rand3DElasticd(
        keys=['image', 'label'],
        magnitude_range=(0, ELASTIC_DISP_MM),
        mode=['bilinear', 'nearest'],
        prob=1.0,
        sigma_range=(5, 8),
    )


def run_monai(
    data: Dict[str, Any],
    dim: int,
    device: torch.device,
    input_mode: str,
    steps: List[str],
) -> None:
    # Add channel dim — MONAI expects (C, *spatial).
    img = data['ct_data'].unsqueeze(0).cpu().numpy()
    if input_mode == 'image':
        pipeline = _build_monai_image_pipeline(steps, device=device, dim=dim)
        pipeline(img)
    else:
        pipeline = _build_monai_dict_pipeline(steps, device=device, dim=dim)
        sample = {
            'image': img,
            'label': data['labels'].cpu().numpy(),   # already (1, *spatial)
        }
        pipeline(sample)
    if device.type == 'cuda':
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# TorchIO pipelines
# ---------------------------------------------------------------------------

def _build_torchio_pipeline(steps: List[str], *, device: torch.device, dim: int) -> Any:
    import torchio as tio

    crop_amount = int(CROP_REMOVE_MM)
    transforms = []
    for s in steps:
        if s == 'crop':
            if dim == 2:
                # (D, H, W) bounds — don't crop singleton depth axis.
                transforms.append(tio.Crop((0, 0, crop_amount, crop_amount, crop_amount, crop_amount)))
            else:
                transforms.append(tio.Crop(crop_amount))
        elif s == 'affine':
            transforms.append(tio.RandomAffine(
                degrees=ROTATION_DEG,
                isotropic=False,
                scales=SCALING_RANGE,
                translation=TRANSLATION_MM,
            ))
        elif s == 'elastic':
            transforms.append(tio.RandomElasticDeformation(
                max_displacement=ELASTIC_DISP_MM,
                num_control_points=7,
            ))
        elif s == 'minmax':
            transforms.append(tio.RescaleIntensity(out_min_max=MINMAX_RANGE))

    return tio.Compose(transforms)


def run_torchio(
    data: Dict[str, Any],
    dim: int,
    device: torch.device,
    input_mode: str,
    steps: List[str],
) -> None:
    import torchio as tio

    pipeline = _build_torchio_pipeline(steps, device=device, dim=dim)
    # TorchIO expects 4-D tensors (C, D, H, W).  For 2-D data add a
    # singleton depth dimension so the transforms still apply.
    img_tensor = data['ct_data'].unsqueeze(0).cpu()     # (1, *spatial)
    if dim == 2:
        img_tensor = img_tensor.unsqueeze(1)             # (1, 1, H, W)
    kwargs = dict(
        image=tio.ScalarImage(tensor=img_tensor),
    )
    if input_mode == 'image-labels':
        lbl_tensor = data['labels'].cpu()                # (1, *spatial)
        if dim == 2:
            lbl_tensor = lbl_tensor.unsqueeze(1)         # (1, 1, H, W)
        kwargs['label'] = tio.LabelMap(tensor=lbl_tensor)
    subject = tio.Subject(**kwargs)
    pipeline(subject)
    if device.type == 'cuda':
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _measure_run(
    fn: Callable[[], None],
    device: torch.device,
) -> Dict[str, float]:
    """Run *fn* once and return timing + peak memory metrics."""
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    tracemalloc.start()
    t0 = time.perf_counter()

    fn()

    elapsed = time.perf_counter() - t0
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_vram = 0.0
    if device.type == 'cuda':
        peak_vram = torch.cuda.max_memory_allocated(device)

    return dict(
        peak_ram_mb=peak_ram / 1024**2,
        peak_vram_mb=peak_vram / 1024**2,
        time=elapsed,
    )


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

LIBRARY_RUNNERS = {
    'augmed': run_augmed,
    'monai': run_monai,
    'torchio': run_torchio,
}


def benchmark(
    *,
    n_runs: int = 3,
    warmup: int = 1,
) -> pd.DataFrame:
    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices.append(torch.device('cuda'))

    rows: List[Dict[str, Any]] = []

    for device in devices:
        for dim in DIMS:
            print(f'\n{"=" * 60}')
            print(f'Device: {device}  |  Dimensionality: {dim}D')
            print(f'{"=" * 60}')
            data = load_data(device, dim)
            print(f'  data shape: {tuple(data["ct_data"].shape)}, '
                  f'labels shape: {tuple(data["labels"].shape)}, '
                  f'points shape: {tuple(data["points"].shape)}')

            for pipe_name, steps in PIPELINE_DEFS.items():
                for input_mode in INPUTS:
                    for lib_name, runner in LIBRARY_RUNNERS.items():
                        # Skip unsupported inputs.
                        if input_mode not in LIBRARY_INPUTS[lib_name]:
                            continue
                        # Skip unsupported dimensionalities.
                        if dim not in LIBRARY_DIMS[lib_name]:
                            continue

                        label = (f'{lib_name} / {pipe_name} / {input_mode}'
                                 f' / {dim}D / {device}')
                        print(f'\n  {label}')

                        # ---- warmup ----
                        ok = True
                        last_exc = None
                        for w in range(warmup):
                            try:
                                runner(data, dim, device, input_mode, steps)
                            except Exception as exc:
                                print(f'    SKIP ({type(exc).__name__}: {exc})')
                                last_exc = exc
                                ok = False
                                break
                        if not ok:
                            rows.append(dict(
                                device=str(device),
                                dim=dim,
                                error=str(last_exc),
                                input=input_mode,
                                library=lib_name,
                                peak_ram_mb=None,
                                peak_vram_mb=None,
                                pipeline=pipe_name,
                                run=None,
                                time=None,
                            ))
                            continue

                        # ---- timed runs ----
                        for r in range(n_runs):
                            metrics = _measure_run(
                                lambda _r=runner, _d=data, _dm=dim,
                                       _dev=device, _im=input_mode, _s=steps:
                                    _r(_d, _dm, _dev, _im, _s),
                                device,
                            )
                            rows.append(dict(
                                device=str(device),
                                dim=dim,
                                error=None,
                                input=input_mode,
                                library=lib_name,
                                peak_ram_mb=round(metrics['peak_ram_mb'], 2),
                                peak_vram_mb=round(metrics['peak_vram_mb'], 2),
                                pipeline=pipe_name,
                                run=r + 1,
                                time=round(metrics['time'], 4),
                            ))
                            print(f'    run {r + 1}/{n_runs}: '
                                  f'{metrics["time"]:.3f}s  '
                                  f'RAM {metrics["peak_ram_mb"]:.1f} MB  '
                                  f'VRAM {metrics["peak_vram_mb"]:.1f} MB')

    df = pd.DataFrame(rows)
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print aggregated stats grouped by library/pipeline/input/dim/device."""
    successful = df.dropna(subset=['time'])
    if successful.empty:
        print('\nNo successful runs to summarise.')
        return

    summary = (
        successful
        .groupby(['device', 'dim', 'input', 'library', 'pipeline'])
        .agg(
            mean_time=('time', 'mean'),
            n_runs=('time', 'count'),
            peak_ram_mb=('peak_ram_mb', 'max'),
            peak_vram_mb=('peak_vram_mb', 'max'),
            std_time=('time', 'std'),
        )
        .reset_index()
        .sort_values(['device', 'dim', 'input', 'pipeline', 'mean_time'])
    )
    print('\n=== Summary ===')
    print(summary.to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark augmentation libraries.')
    parser.add_argument('--n-runs', default=3, help='Timed runs per config (default: 3)', type=int)
    parser.add_argument('--output', default=None, help='Output CSV path (default: next to this script)', type=str)
    parser.add_argument('--warmup', default=1, help='Warmup runs per config (default: 1)', type=int)
    args = parser.parse_args()

    output = args.output
    if output is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = str(Path(__file__).parent / f'results_{ts}.csv')

    print(f'Running benchmark: {args.warmup} warmup + {args.n_runs} timed runs per config')
    df = benchmark(n_runs=args.n_runs, warmup=args.warmup)
    df.to_csv(output, index=False)
    print(f'\nResults saved to {output}')
    print_summary(df)


if __name__ == '__main__':
    main()
