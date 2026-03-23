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

DATA_MODES = ['image', 'image-labels', 'image-labels-points']

# Which data modes each library supports.
LIBRARY_DATA_MODES = {
    'augmed': ['image', 'image-labels', 'image-labels-points'],
    'monai': ['image', 'image-labels'],
    'torchio': ['image', 'image-labels'],
}

NDIMS = ['2d', '3d']

# Which dimensionalities each library supports.
LIBRARY_NDIMS = {
    'augmed': ['2d', '3d'],
    'monai': ['2d', '3d'],
    'torchio': ['2d', '3d'],
}

# Shared transform parameters that are as close as possible across
# libraries so the workloads are comparable.
CROP_REMOVE_MM = 50.0           # mm to remove per side
ROTATION_DEG = 15.0             # max rotation per axis in degrees
SCALING_RANGE = (0.8, 1.2)
TRANSLATION_MM = 20.0           # max translation per axis in mm
ELASTIC_SPACING_MM = 50.0       # control-point spacing
ELASTIC_DISP_MM = 10.0          # max displacement
MINMAX_RANGE = (0.0, 1.0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(device: torch.device, ndim: str) -> Dict[str, Any]:
    """Load (or generate) representative data for the given dimensionality."""
    if ndim == '3d':
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

def _build_augmed_pipeline(steps: List[str], *, device: torch.device) -> Any:
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
            transforms.append(RandomCrop(crop_remove=CROP_REMOVE_MM))
        elif s == 'affine':
            transforms.append(RandomAffine(
                rotation=ROTATION_DEG,
                scaling=SCALING_RANGE,
                translation=TRANSLATION_MM,
            ))
        elif s == 'elastic':
            transforms.append(RandomElastic(
                control_spacing=ELASTIC_SPACING_MM,
                displacement=ELASTIC_DISP_MM,
            ))
        elif s == 'minmax':
            transforms.append(MinMax(max=MINMAX_RANGE[1], min=MINMAX_RANGE[0]))
        else:
            raise ValueError(f'Unknown step: {s}')

    return Pipeline(transforms, device=device, seed=42)


def run_augmed(
    data: Dict[str, Any],
    data_mode: str,
    device: torch.device,
    ndim: str,
    steps: List[str],
) -> None:
    pipeline = _build_augmed_pipeline(steps, device=device)
    inputs = [data['ct_data']]
    if data_mode in ('image-labels', 'image-labels-points'):
        inputs.append(data['labels'])
    if data_mode == 'image-labels-points':
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
    ndim: str,
) -> Any:
    """Build a MONAI pipeline for image-only mode (non-dict transforms)."""
    from monai.transforms import (
        Compose,
        EnsureChannelFirst,
        RandAffine,
        RandSpatialCrop,
        ScaleIntensityRange,
    )

    roi_size = [156, 156] if ndim == '2d' else [156, 156, 78]
    n_spatial = 2 if ndim == '2d' else 3
    rot_range = [np.deg2rad(ROTATION_DEG)] * (1 if ndim == '2d' else 3)

    transforms: list = [EnsureChannelFirst()]
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
            transforms.append(_monai_elastic_transform(ndim=ndim))
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
    ndim: str,
) -> Any:
    """Build a MONAI pipeline for image-labels mode (dict transforms)."""
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        RandAffined,
        RandSpatialCropd,
        ScaleIntensityRanged,
    )

    roi_size = [156, 156] if ndim == '2d' else [156, 156, 78]
    n_spatial = 2 if ndim == '2d' else 3
    rot_range = [np.deg2rad(ROTATION_DEG)] * (1 if ndim == '2d' else 3)

    transforms: list = [EnsureChannelFirstd(keys=['image', 'label'])]
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
            transforms.append(_monai_elastic_transform_d(ndim=ndim))
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


def _monai_elastic_transform(*, ndim: str) -> Any:
    """Return the appropriate MONAI elastic transform for the dimensionality."""
    if ndim == '2d':
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


def _monai_elastic_transform_d(*, ndim: str) -> Any:
    """Return the appropriate MONAI dict elastic transform."""
    if ndim == '2d':
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
    data_mode: str,
    device: torch.device,
    ndim: str,
    steps: List[str],
) -> None:
    if data_mode == 'image':
        pipeline = _build_monai_image_pipeline(steps, device=device, ndim=ndim)
        pipeline(data['ct_data'].cpu().numpy())
    else:
        pipeline = _build_monai_dict_pipeline(steps, device=device, ndim=ndim)
        sample = {
            'image': data['ct_data'].cpu().numpy(),
            'label': data['labels'].cpu().numpy(),
        }
        pipeline(sample)
    if device.type == 'cuda':
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# TorchIO pipelines
# ---------------------------------------------------------------------------

def _build_torchio_pipeline(steps: List[str], *, device: torch.device) -> Any:
    import torchio as tio

    transforms = []
    for s in steps:
        if s == 'crop':
            transforms.append(tio.Crop(CROP_REMOVE_MM))
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
    data_mode: str,
    device: torch.device,
    ndim: str,
    steps: List[str],
) -> None:
    import torchio as tio

    pipeline = _build_torchio_pipeline(steps, device=device)
    # TorchIO expects 4-D tensors (C, D, H, W).  For 2-D data add a
    # singleton depth dimension so the transforms still apply.
    img_tensor = data['ct_data'].unsqueeze(0).cpu()     # (1, *spatial)
    if ndim == '2d':
        img_tensor = img_tensor.unsqueeze(1)             # (1, 1, H, W)
    kwargs = dict(
        image=tio.ScalarImage(tensor=img_tensor),
    )
    if data_mode == 'image-labels':
        lbl_tensor = data['labels'].cpu()                # (1, *spatial)
        if ndim == '2d':
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
        time_s=elapsed,
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
        for ndim in NDIMS:
            print(f'\n{"=" * 60}')
            print(f'Device: {device}  |  Dimensionality: {ndim.upper()}')
            print(f'{"=" * 60}')
            data = load_data(device, ndim)
            print(f'  data shape: {tuple(data["ct_data"].shape)}, '
                  f'labels shape: {tuple(data["labels"].shape)}, '
                  f'points shape: {tuple(data["points"].shape)}')

            for pipe_name, steps in PIPELINE_DEFS.items():
                for data_mode in DATA_MODES:
                    for lib_name, runner in LIBRARY_RUNNERS.items():
                        # Skip unsupported data modes.
                        if data_mode not in LIBRARY_DATA_MODES[lib_name]:
                            continue
                        # Skip unsupported dimensionalities.
                        if ndim not in LIBRARY_NDIMS[lib_name]:
                            continue

                        label = (f'{lib_name} / {pipe_name} / {data_mode}'
                                 f' / {ndim} / {device}')
                        print(f'\n  {label}')

                        # ---- warmup ----
                        ok = True
                        for w in range(warmup):
                            try:
                                runner(data, data_mode, device, ndim, steps)
                            except Exception as exc:
                                print(f'    SKIP ({type(exc).__name__}: {exc})')
                                ok = False
                                break
                        if not ok:
                            rows.append(dict(
                                data_mode=data_mode,
                                device=str(device),
                                error=str(exc),
                                library=lib_name,
                                ndim=ndim,
                                peak_ram_mb=None,
                                peak_vram_mb=None,
                                pipeline=pipe_name,
                                run=None,
                                time_s=None,
                            ))
                            continue

                        # ---- timed runs ----
                        for r in range(n_runs):
                            metrics = _measure_run(
                                lambda _r=runner, _d=data, _dm=data_mode,
                                       _dev=device, _nd=ndim, _s=steps:
                                    _r(_d, _dm, _dev, _nd, _s),
                                device,
                            )
                            rows.append(dict(
                                data_mode=data_mode,
                                device=str(device),
                                error=None,
                                library=lib_name,
                                ndim=ndim,
                                peak_ram_mb=round(metrics['peak_ram_mb'], 2),
                                peak_vram_mb=round(metrics['peak_vram_mb'], 2),
                                pipeline=pipe_name,
                                run=r + 1,
                                time_s=round(metrics['time_s'], 4),
                            ))
                            print(f'    run {r + 1}/{n_runs}: '
                                  f'{metrics["time_s"]:.3f}s  '
                                  f'RAM {metrics["peak_ram_mb"]:.1f} MB  '
                                  f'VRAM {metrics["peak_vram_mb"]:.1f} MB')

    df = pd.DataFrame(rows)
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print aggregated stats grouped by library/pipeline/data_mode/ndim/device."""
    successful = df.dropna(subset=['time_s'])
    if successful.empty:
        print('\nNo successful runs to summarise.')
        return

    summary = (
        successful
        .groupby(['data_mode', 'device', 'library', 'ndim', 'pipeline'])
        .agg(
            mean_time_s=('time_s', 'mean'),
            n_runs=('time_s', 'count'),
            peak_ram_mb=('peak_ram_mb', 'max'),
            peak_vram_mb=('peak_vram_mb', 'max'),
            std_time_s=('time_s', 'std'),
        )
        .reset_index()
        .sort_values(['data_mode', 'device', 'ndim', 'pipeline', 'mean_time_s'])
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
