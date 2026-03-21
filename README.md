# SplatLab

**SplatLab** is a Julia project for training and visualizing 3D Gaussian Splatting models.

This repository includes data loading, rasterization, optimization, evaluation, and reporting scripts for reproducible experiments.

## Key Capabilities

- End-to-end training workflow: dataset, Gaussian model, rasterizer, optimizer, and validation
- Multi-backend extension support: CUDA, Metal, and AMDGPU
- Training progress visualization and image export
- Metrics logging and report chart generation

## Repository Structure

```text
src/                      # Core implementation: camera, dataset, training, rasterization, GUI
ext/                      # GPU backend extensions
benchmark/pipeline.jl     # Benchmark pipeline script
train.jl                  # Main training script example
plot_report_charts.jl     # Training report plotting script
resize_images.jl          # Image preprocessing utility
assets/                   # Sample assets and dataset-related resources
```

## Requirements

- Julia 1.10 or later
- A supported GPU environment is recommended:
	- NVIDIA CUDA
	- Apple Metal
	- AMD ROCm

Install and configure backend dependencies according to your target hardware.

## Path Configuration

This project reads the dataset path from the environment variable `SPLATLAB_DATASET_PATH`.

Set this variable to the root directory of your COLMAP-style dataset before running scripts.

PowerShell (Windows):

```powershell
$env:SPLATLAB_DATASET_PATH = "D:/path/to/your/dataset"
```

Bash (Linux/macOS):

```bash
export SPLATLAB_DATASET_PATH="/path/to/your/dataset"
```

Example dataset layout:

```text
dataset/
	images/
	sparse/
		0/
```

## Quick Start

1. Prepare a COLMAP-style dataset directory.
2. Set `SPLATLAB_DATASET_PATH` to your dataset location.
3. Run training:

```bash
julia train.jl
```

4. Generate report charts:

```bash
julia plot_report_charts.jl
```

## Typical Outputs

- `training_metrics.csv`: recorded training metrics by iteration
- `chart_psnr.png`, `chart_ssim.png`: quality trend plots
- `chart_gaussians_growth.png`: Gaussian count growth curve
- `evolution_step_*.png`: intermediate rendering snapshots

## For Coursework Submission

- The repository is organized for EE5311's experimentation and reporting.
- The generated CSV and chart artifacts are used directly in our project reports.
- For reproducibility, keep the dataset path, scale, and iteration settings.
