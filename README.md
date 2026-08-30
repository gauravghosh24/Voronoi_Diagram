# GPU Discrete Voronoi Research Project

This project implements and tests a GPU/CPU-friendly discrete Voronoi construction workflow inspired by "A digital geometric approach for discrete Voronoi diagram construction using GPU".

The main goal is to compare a digital circle-growing Voronoi method against brute-force ground truth and a basic Jump Flood Algorithm (JFA) across site distributions, cutoff radii, grid sizes, and site counts.

## Algorithms Implemented

- Brute force Voronoi: exact squared Euclidean nearest-site assignment. This is the ground truth.
- Proposed digital circle growing: Taichi implementation using per-site radius frames and integer radius-band filling.
- Jump Flood Algorithm: basic Taichi 8-neighbor JFA with one final local correction pass.

Coordinate convention is always `(x, y)` for sites and `labels[y, x]` for arrays.

## Installation

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Test Taichi GPU initialization:

```powershell
python -c "import taichi as ti; ti.init(arch=ti.gpu); print('Taichi GPU OK')"
```

If GPU initialization fails during experiments, the project prints a warning and falls back to CPU.

## Run a Single Experiment

Small CPU test:

```powershell
python run_experiment.py --grid 128 --sites 10 --mode random --cutoff 40 --algorithm all --arch cpu --compare-bruteforce --save-images --save-diff
```

GPU test:

```powershell
python run_experiment.py --grid 512 --sites 100 --mode random --cutoff 80 --algorithm all --arch gpu --compare-bruteforce --save-images --save-diff
```

Proposed-only example:

```powershell
python run_experiment.py --grid 512 --sites 100 --mode random --cutoff 80 --algorithm proposed --arch gpu --compare-bruteforce --save-images --save-diff
```

Clustered CPU example:

```powershell
python run_experiment.py --grid 256 --sites 20 --mode clustered --cutoff 60 --algorithm proposed --arch cpu --compare-bruteforce --save-images
```

ML-predicted cutoff example:

```powershell
python run_experiment.py --grid 512 --sites 100 --mode clustered --cutoff auto_ml --algorithm proposed --arch gpu --compare-bruteforce --save-images --save-diff
```

Supported algorithms:

- `proposed`
- `brute_force`
- `jfa`
- `all`

Supported site modes:

- `random`
- `clustered`
- `boundary`
- `close`
- `collinear_horizontal`
- `collinear_diagonal`
- `grid`
- `symmetric_tie`

## Run Batch and Study Scripts

Cutoff study:

```powershell
python experiments/cutoff_study.py --arch gpu
```

Full default batch:

```powershell
python run_batch.py --arch gpu
```

Other studies:

```powershell
python experiments/distribution_study.py --arch gpu
python experiments/grid_size_study.py --arch gpu
python experiments/site_count_study.py --arch gpu
```

## ML Cutoff Prediction

The project can train a scikit-learn model to predict a good global
`cutoff_radius` for the proposed digital circle-growing algorithm. The target
is the oracle cutoff computed from brute-force distance maps:

```text
ceil(sqrt(max_squared_distance_to_nearest_site))
```

Build the cutoff training dataset:

```powershell
python experiments/build_cutoff_dataset.py --max-seed 30
```

Train the Random Forest cutoff model:

```powershell
python experiments/train_cutoff_model.py
```

This saves:

```text
models/cutoff_predictor.joblib
models/cutoff_feature_columns.json
outputs/csv/cutoff_model_metrics.csv
```

Evaluate fixed, ML-predicted, and oracle cutoffs:

```powershell
python experiments/evaluate_ml_cutoff.py --arch gpu
```

Run the proposed algorithm with the trained ML cutoff predictor:

```powershell
python run_experiment.py --grid 512 --sites 100 --mode clustered --cutoff auto_ml --algorithm proposed --arch gpu --compare-bruteforce --save-images --save-diff
```

If `models/cutoff_predictor.joblib` does not exist, `--cutoff auto_ml` exits
with:

```text
ML cutoff model not found. Run experiments/train_cutoff_model.py first.
```

## Outputs

Single-run image outputs use the requested naming pattern:

```text
outputs/images/random_512_100_cutoff80_proposed.png
outputs/images/random_512_100_cutoff80_bruteforce.png
outputs/images/random_512_100_cutoff80_jfa.png
```

Difference images:

```text
outputs/diff/random_512_100_cutoff80_proposed_vs_bruteforce.png
outputs/diff/random_512_100_cutoff80_jfa_vs_bruteforce.png
```

CSV and logs:

```text
outputs/csv/results.csv
outputs/csv/<prefix>_sites.csv
outputs/logs/<prefix>_stats.json
```

Cutoff plots:

```text
outputs/plots/cutoff_vs_runtime.png
outputs/plots/cutoff_vs_accuracy.png
outputs/plots/cutoff_vs_unassigned.png
outputs/plots/cutoff_study_summary.png
```

## Metrics

Each algorithm run appends a row to `outputs/csv/results.csv` with:

- `accuracy_vs_bruteforce`
- `mismatched_pixels`
- `unassigned_pixels`
- `unassigned_percent`
- `total_time`
- `fill_frames_time`
- `generate_result_time`
- `estimated_memory_mb`

Unassigned pixels are represented by label `-1`, shown as white in output images, and counted as mismatches.

## Recommended Experiment Order

1. Run the small CPU test to confirm the environment.
2. Run `algorithm all` on a small grid to compare all algorithms.
3. Run the cutoff study to understand completeness versus runtime.
4. Run the distribution study to find failure modes around boundary, close, and collinear sites.
5. Run grid-size and site-count studies to observe scaling and memory pressure.

## Implementation Notes

The proposed algorithm allocates `frames[S, H, W]`, where each entry stores the radius at which a site reaches a pixel. This is intentionally research-friendly because it exposes the paper-inspired frame structure, but it can be memory-heavy.

The current digital circle generator uses integer radius bands:

```text
(r - 0.5)^2 <= dx^2 + dy^2 < (r + 0.5)^2
```

This avoids absentee pixels inside the cutoff region. It is slower than a midpoint/octant-only boundary plotter, but it is a clearer correctness baseline and can be replaced later in `src/proposed_circle_growing.py`.

The proposed method stores integer radius layers, so it can disagree with exact brute force near Voronoi boundaries, especially when multiple sites land in the same radius layer. This is useful for studying the approximation behavior.

## Troubleshooting

If `taichi` is missing:

```powershell
pip install -r requirements.txt
```

If GPU initialization fails, rerun with CPU:

```powershell
python run_experiment.py --grid 128 --sites 10 --mode random --cutoff 40 --algorithm all --arch cpu --compare-bruteforce --save-images --save-diff
```

If memory usage is too high, reduce one of:

- `--sites`
- `--grid`
- `--cutoff`

The largest allocation in the proposed method is proportional to:

```text
num_sites * grid_size * grid_size * 4 bytes
```
