import sys as _sys
# Remove the script's directory from sys.path so that profile.py in this folder
# doesn't shadow the stdlib 'profile' module (used by cProfile → IPython → seaborn).
_script_dir = str(__import__('pathlib').Path(__file__).resolve().parent)
if _script_dir in _sys.path:
    _sys.path.remove(_script_dir)

import argparse
from augmed import Crop, MinMax, Pipeline, RandomAffine, RandomElastic, affine, crop, elastic, pipeline, transforms
from augmed.data import load_example_ct
from augmed.typing import AffineMatrix2DTensor, AffineMatrix3DTensor, AffineMatrixTensor, BatchLabelImage2DTensor, BatchLabelImage3DTensor, BatchLabelImageTensor, Image2DTensor, Image3DTensor, ImageTensor, Points2DTensor, Points3DTensor, PointsTensor, SpatialDim
from augmed.utils import args, config, to_tensor
from datetime import datetime
import gc
from monai.transforms import Compose, Rand2DElasticd, Rand3DElasticd, RandAffined, RandSpatialCropd, ScaleIntensityRanged
import numpy as np
import pandas as pd
from pathlib import Path
import time
import torch
from torchio import Compose as TioCompose, Crop as TioCrop, LabelMap, RandomAffine as TioRandomAffine, RandomElasticDeformation, RescaleIntensity, ScalarImage, Subject
import tracemalloc
from typing import Any, Callable, Dict, List, Tuple

# Test configs.
DIMS = [2, 3]
INPUTS = ['image', 'image-labels', 'image-labels-points']
LIBRARY_DEVICES = {
    'augmed': ['cpu', 'cuda'],
    'monai': ['cpu', 'cuda'],
    'torchio': ['cpu'],         # CPU only - SimpleITK under the hood.
}
LIBRARY_DIMS = {
    'augmed': [2, 3],
    'monai': [2, 3],
    'torchio': [2, 3],      # Uses 3D for 2D data.
}
LIBRARY_INPUTS = {
    'augmed': ['image', 'image-labels', 'image-labels-points'],
    'monai': ['image-labels'],        # Dict transforms always require label.
    'torchio': ['image', 'image-labels'],
}
PIPELINES = {
    'crop+affine+minmax': ['crop', 'affine', 'minmax'],
    'crop+affine+elastic+minmax': ['crop', 'affine', 'elastic', 'minmax'],
}

# Augmentation settings.
# Don't randomise the crop as this affects the output shape and makes it difficult
# to compare libraries.
CROP_REMOVE_MM = 20.0
ELASTIC_SPACING_MM = 50.0
ELASTIC_DISP_MM = 20.0
MINMAX_RANGE = (0.0, 1.0)
ROTATION_RANGE_DEG = (-15.0, 15.0)
SCALING_RANGE = (0.8, 1.2)
TRANSLATION_RANGE_MM = (-20.0, 20.0)

# Data assertions - make sure all pipelines are being evaluated on the same data shapes.
CROP_OUTPUT_2D = (476, 476)
CROP_OUTPUT_3D = (476, 476, 93)
CT_SHAPE_2D = (512, 512)
CT_SHAPE_3D = (512, 512, 107)
CT_SPACING_2D = (1.12, 1.12)
CT_SPACING_3D = (1.12, 1.12, 3.0)

def load_data(
    device: torch.device,
    dim: int,
    ) -> Tuple[AffineMatrixTensor, ImageTensor, BatchLabelImageTensor, PointsTensor]:
    if dim == 3:
        return load_3d_data(device)
    return load_2d_data(device)

def load_2d_data(device: torch.device) -> Tuple[AffineMatrix2DTensor, Image2DTensor, BatchLabelImage2DTensor, Points2DTensor]:
    ct_data, affine, labels, points = load_example_ct()

    # Extract middle slice for 2D test and project points.
    idx = ct_data.shape[2] // 2
    ct_data = ct_data[:, :, idx]
    labels = labels[:, :, :, idx]
    points = points[:, :2]

    # Get 2D affine.
    affine = to_tensor(affine, device=device, dtype=torch.float32)
    affine_2d = torch.zeros(3, 3, device=device, dtype=torch.float32)
    affine_2d[:2, :2] = torch.as_tensor(affine[:2, :2])
    affine_2d[:2, 2] = torch.as_tensor(affine[:2, 3])
    affine_2d[2, 2] = 1.0
    affine = affine_2d

    assert ct_data.shape == CT_SHAPE_2D, f'Expected CT shape {CT_SHAPE_2D}, got {ct_data.shape}'

    ct_data = to_tensor(ct_data, device=device, dtype=torch.float32)
    affine = to_tensor(affine, device=device, dtype=torch.float32)
    labels = to_tensor(labels, device=device, dtype=torch.bool)
    points = to_tensor(points, device=device, dtype=torch.float32)

    return affine, ct_data, labels, points

def load_3d_data(device: torch.device) -> Tuple[AffineMatrix3DTensor, Image3DTensor, BatchLabelImage3DTensor, Points3DTensor]:
    ct_data, affine, labels, points = load_example_ct()
    assert ct_data.shape == CT_SHAPE_3D, f'Expected CT shape {CT_SHAPE_3D}, got {ct_data.shape}'
    ct_data = to_tensor(ct_data, device=device, dtype=torch.float32)
    affine = to_tensor(affine, device=device, dtype=torch.float32)
    labels = to_tensor(labels, device=device, dtype=torch.bool)
    points = to_tensor(points, device=device, dtype=torch.float32)
    return affine, ct_data, labels, points

# Create augmed methods.

def build_augmed_pipeline(
    device: torch.device,
    dim: int,
    steps: List[str], 
    ) -> Pipeline:
    transforms = []
    for s in steps:
        if s == 'crop':
            transforms.append(Crop(dim=dim, remove=CROP_REMOVE_MM))
        elif s == 'affine':
            transforms.append(RandomAffine(
                dim=dim,
                rotation=ROTATION_RANGE_DEG,
                scaling=SCALING_RANGE,
                translation=TRANSLATION_RANGE_MM,
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

    return Pipeline(transforms, device=device, dim=dim)

def run_augmed_pipeline(
    pipeline: Pipeline,
    dim: int,
    image: torch.Tensor = None,
    affine: torch.Tensor = None,
    labels: torch.Tensor = None,
    points: torch.Tensor = None,
    ) -> Tuple[Any, ...]:
    inputs = [image]
    if labels is not None:
        inputs.append(labels)
    if points is not None:
        inputs.append(points)
    return pipeline.transform(*inputs, affine=affine)

# Build monai methods.

def build_monai_pipeline(
    device: torch.device,
    dim: int,
    steps: List[str],
    ) -> Compose:
    roi_size = list(CROP_OUTPUT_2D) if dim == 2 else list(CROP_OUTPUT_3D)
    n_spatial = dim
    rot_range = [np.deg2rad(ROTATION_RANGE_DEG[0]), np.deg2rad(ROTATION_RANGE_DEG[1])] * (1 if dim == 2 else 3)

    transforms = []
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
                translate_range=[TRANSLATION_RANGE_MM] * n_spatial,
            ))
        elif s == 'elastic':
            if dim == 2:
                transforms.append(Rand2DElasticd(
                    keys=['image', 'label'],
                    magnitude_range=(0, ELASTIC_DISP_MM),
                    mode=['bilinear', 'nearest'],
                    prob=1.0,
                    spacing=(30, 30),
                ))
            else:
                transforms.append(Rand3DElasticd(
                    keys=['image', 'label'],
                    magnitude_range=(0, ELASTIC_DISP_MM),
                    mode=['bilinear', 'nearest'],
                    prob=1.0,
                    sigma_range=(5, 8),
                ))
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

def run_monai_pipeline(
    pipeline: Compose,
    dim: int,
    affine: torch.Tensor = None,
    image: torch.Tensor = None,
    labels: torch.Tensor = None,
    points: torch.Tensor = None,
    ) -> dict:
    sample = {'image': image.unsqueeze(0)}
    if labels is not None:
        sample['label'] = labels
    result = pipeline(sample)
    if image.device.type == 'cuda':
        torch.cuda.synchronize()
    return result

# Build torchio methods.

def build_torchio_pipeline(
    device: torch.device,
    dim: int,
    steps: List[str],
    ) -> TioCompose:
    transforms = []
    for s in steps:
        if s == 'crop':
            if dim == 2:
                # 2-D slices are promoted to (C, 1, H, W) — don't crop the
                # singleton depth axis.
                cy = (CT_SHAPE_2D[0] - CROP_OUTPUT_2D[0]) // 2
                cx = (CT_SHAPE_2D[1] - CROP_OUTPUT_2D[1]) // 2
                transforms.append(TioCrop((0, 0, cy, cy, cx, cx)))
            else:
                cx = (CT_SHAPE_3D[0] - CROP_OUTPUT_3D[0]) // 2
                cy = (CT_SHAPE_3D[1] - CROP_OUTPUT_3D[1]) // 2
                cz = (CT_SHAPE_3D[2] - CROP_OUTPUT_3D[2]) // 2
                transforms.append(TioCrop((cx, cx, cy, cy, cz, cz)))
        elif s == 'affine':
            transforms.append(TioRandomAffine(
                degrees=ROTATION_RANGE_DEG,
                isotropic=False,
                scales=SCALING_RANGE,
                translation=TRANSLATION_RANGE_MM,
            ))
        elif s == 'elastic':
            transforms.append(RandomElasticDeformation(
                max_displacement=ELASTIC_DISP_MM,
                num_control_points=7,
            ))
        elif s == 'minmax':
            transforms.append(RescaleIntensity(out_min_max=MINMAX_RANGE))

    return TioCompose(transforms)

def run_torchio_pipeline(
    pipeline: TioCompose,
    dim: int,
    affine: torch.Tensor = None,
    image: torch.Tensor = None,
    labels: torch.Tensor = None,
    points: torch.Tensor = None,
    ) -> Subject:
    # TorchIO expects 4-D tensors (C, D, H, W). For 2-D data add a
    # singleton depth dimension so the transforms still apply.
    img_tensor = image.unsqueeze(0).cpu()
    if dim == 2:
        img_tensor = img_tensor.unsqueeze(1)
    kwargs = dict(image=ScalarImage(tensor=img_tensor))
    if labels is not None:
        lbl_tensor = labels.cpu()
        if dim == 2:
            lbl_tensor = lbl_tensor.unsqueeze(1)
        kwargs['label'] = LabelMap(tensor=lbl_tensor)
    subject = Subject(**kwargs)
    result = pipeline(subject)
    if image.device.type == 'cuda':
        torch.cuda.synchronize()
    return result

def measure_run(
    pipeline: Any,
    run_pipeline: Callable,
    input_kwargs: Dict[str, Any],
    device: torch.device,
    dim: SpatialDim,
) -> Dict[str, float]:
    # Perform cleanup.
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Start memory and time measurement.
    tracemalloc.start()
    t0 = time.perf_counter()

    # Perform pipeline.
    run_pipeline(pipeline, dim, **input_kwargs)

    # Measure time and memory.
    elapsed = time.perf_counter() - t0
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_vram = 0.0
    if device.type == 'cuda':
        peak_vram = torch.cuda.max_memory_allocated(device)

    stats = dict(
        peak_ram_mb=peak_ram / 1024**2,
        peak_vram_mb=peak_vram / 1024**2,
        time=elapsed,
    )

    return stats

LIBRARY_METHODS = {
    'augmed': (build_augmed_pipeline, run_augmed_pipeline),
    'monai': (build_monai_pipeline, run_monai_pipeline),
    'torchio': (build_torchio_pipeline, run_torchio_pipeline),
}

def benchmark(
    n_runs: int = 3,
    warmup: int = 1,
    ) -> pd.DataFrame:
    # Get available devices.
    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices.append(torch.device('cuda'))

    rows: List[Dict[str, Any]] = []

    # Run for each dim.
    for dim in DIMS:

        # Run for each device.
        for dev in devices:
            print(f'\n{"=" * 60}')
            print(f'Benchmarking on device: {dev}, dim: {dim}D')
            print(f'{"=" * 60}')
            affine, ct_data, labels, points = load_data(dev, dim)
            print(f"CT shape: {ct_data.shape}, affine shape: {affine.shape}, labels shape: {labels.shape}, points shape: {points.shape}")

            # Run for each pipeline (e.g. affine vs. elastic).
            for pipe_name, steps in PIPELINES.items():

                for lib_name, (build_pipeline, run_pipeline) in LIBRARY_METHODS.items():
                    # Skip unsupported dims, e.g. 3D for torchvision.
                    if dim not in LIBRARY_DIMS[lib_name]:
                        continue
                    # Skip unsupported devices.
                    if str(dev) not in LIBRARY_DEVICES[lib_name]:
                        continue

                    # Build the pipeline.
                    pipeline = build_pipeline(dev, dim, steps)

                    # Run for each input type (image vs. image+labels vs. image+labels+points).
                    for input_mode in INPUTS:
                        # Skip unsupported inputs, e.g. points for monai/torchio.
                        if input_mode not in LIBRARY_INPUTS[lib_name]:
                            continue

                        # Pass the required data for this input mode.
                        input_kwargs = dict(
                            image=ct_data,
                            affine=affine,
                        )
                        if input_mode in ('image-labels', 'image-labels-points'):
                            input_kwargs['labels'] = labels
                        if input_mode == 'image-labels-points':
                            input_kwargs['points'] = points

                        # Print run ID.
                        run_id = f'{lib_name}_{pipe_name}_{input_mode}_{dim}D_{dev}'
                        print(f'\n  {run_id}')

                        # Perform warmup stretches.
                        for _ in range(warmup):
                            run_pipeline(pipeline, dim=dim, **input_kwargs)

                        # Perform timed runs.
                        for i in range(n_runs):
                            metrics = measure_run(pipeline, run_pipeline, input_kwargs, dev, dim)
                            rows.append(dict(
                                device=str(dev),
                                dim=dim,
                                error=None,
                                input=input_mode,
                                library=lib_name,
                                peak_ram_mb=round(metrics['peak_ram_mb'], 2),
                                peak_vram_mb=round(metrics['peak_vram_mb'], 2),
                                pipeline=pipe_name,
                                run=i + 1,
                                time=round(metrics['time'], 4),
                            ))

                            # Log the results.
                            print(f'    run {i + 1}/{n_runs}: '
                                  f'{metrics["time"]:.3f}s  '
                                  f'RAM {metrics["peak_ram_mb"]:.1f} MB  '
                                  f'VRAM {metrics["peak_vram_mb"]:.1f} MB')

    df = pd.DataFrame(rows)

    # Copy torchio's CPU results into the cuda section as it's CPU-only.
    if torch.cuda.is_available():
        tio_cpu = df[(df['library'] == 'torchio') & (df['device'] == 'cpu')].copy()
        if not tio_cpu.empty:
            tio_cpu['device'] = 'cuda'
            tio_cpu['peak_vram_mb'] = 0.0
            df = pd.concat([df, tio_cpu], ignore_index=True)

    return df

def print_summary(df: pd.DataFrame) -> None:
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

def run_benchmark() -> None:
    parser = argparse.ArgumentParser(description='Benchmark augmentation libraries.')
    parser.add_argument('--n-runs', default=3, help='Timed runs per config (default: 3)', type=int)
    parser.add_argument('--warmup', default=1, help='Warmup runs per config (default: 1)', type=int)
    args = parser.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = Path(__file__).parent / f'benchmark_{args.n_runs}_{ts}'
    output_csv = str(base.with_suffix('.csv'))
    stdout_log = str(base.with_name(base.name + '_stdout.txt'))
    stderr_log = str(base.with_name(base.name + '_stderr.txt'))

    # Send stdout/stderr to log files as the benchmark runs.
    with open(stdout_log, 'w', encoding='utf-8') as fout, \
         open(stderr_log, 'w', encoding='utf-8') as ferr:
        old_stdout, old_stderr = _sys.stdout, _sys.stderr
        _sys.stdout = _Tee(old_stdout, fout)
        _sys.stderr = _Tee(old_stderr, ferr)
        try:
            print(f'Running benchmark: {args.warmup} warmup + {args.n_runs} timed runs per config')
            wall_t0 = time.perf_counter()
            df = benchmark(n_runs=args.n_runs, warmup=args.warmup)
            wall_elapsed = time.perf_counter() - wall_t0
            df.to_csv(output_csv, index=False)
            print(f'\nResults saved to {output_csv}')
            print_summary(df)
            print(f'\nTotal wall-clock time: {wall_elapsed:.1f}s')
        finally:
            _sys.stdout = old_stdout
            _sys.stderr = old_stderr

class _Tee:
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        self._log_file.write(data)
        self._log_file.flush()

    def flush(self):
        self._stream.flush()
        self._log_file.flush()

    def fileno(self):
        return self._stream.fileno()

if __name__ == '__main__':
    run_benchmark()
