"""Evaluate fixed, ML-predicted, and oracle cutoff radii."""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CSV_DIR
from src.brute_force import run_brute_force
from src.cutoff_features import extract_cutoff_features
from src.cutoff_model import (
    calculate_oracle_cutoff_from_dist_map,
    load_cutoff_model,
    predict_cutoff,
)
from src.metrics import compare_to_ground_truth
from src.proposed_circle_growing import run_proposed
from src.site_generators import generate_sites
from src.utils import ensure_dirs


MODEL_NOT_FOUND = "ML cutoff model not found. Run experiments/train_cutoff_model.py first."
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODES = ["random", "clustered", "boundary", "close"]


def _diagonal_cutoff(grid_size):
    return int(math.ceil(math.sqrt(2.0) * (int(grid_size) - 1)))


def _append_rows(rows, output_path):
    pd.DataFrame(rows).to_csv(
        output_path,
        mode="a",
        header=not output_path.exists(),
        index=False,
    )


def _run_method(grid_size, sites, cutoff, arch, gt_labels):
    labels, _, _, stats = run_proposed(grid_size, sites, cutoff, arch=arch)
    metrics = compare_to_ground_truth(labels, gt_labels)
    return stats, metrics


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate ML cutoff predictor.")
    parser.add_argument("--arch", type=str, default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--output", type=str, default=str(CSV_DIR / "ml_cutoff_evaluation.csv"))
    parser.add_argument("--model", type=str, default=str(MODELS_DIR / "cutoff_predictor.joblib"))
    parser.add_argument("--grids", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--site-counts", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102])
    parser.add_argument("--fixed-cutoff", type=int, default=80)
    parser.add_argument("--block-grid", type=int, default=4)
    parser.add_argument("--safety-factor", type=float, default=1.10)
    parser.add_argument("--safety-add", type=int, default=5)
    return parser


def main():
    args = build_parser().parse_args()
    ensure_dirs()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(MODEL_NOT_FOUND)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_cutoff_model(model_path)
    cases = [
        (grid_size, num_sites, mode, seed)
        for grid_size in args.grids
        for num_sites in args.site_counts
        for mode in args.modes
        for seed in args.seeds
    ]

    for grid_size, num_sites, mode, seed in tqdm(cases, desc="ML cutoff evaluation"):
        sites = generate_sites(grid_size, num_sites, mode=mode, seed=seed)
        gt_labels, dist_map = run_brute_force(grid_size, sites)
        oracle_cutoff = calculate_oracle_cutoff_from_dist_map(dist_map)

        features = extract_cutoff_features(grid_size, sites, block_grid=args.block_grid)
        features["mode"] = mode
        ml_cutoff = predict_cutoff(
            model,
            features,
            safety_factor=args.safety_factor,
            safety_add=args.safety_add,
            max_cutoff=_diagonal_cutoff(grid_size),
        )

        rows = []
        for method, cutoff in (
            ("fixed", int(args.fixed_cutoff)),
            ("ml_predicted", int(ml_cutoff)),
            ("oracle", int(oracle_cutoff)),
        ):
            stats, metrics = _run_method(grid_size, sites, cutoff, args.arch, gt_labels)
            rows.append(
                {
                    "grid_size": int(grid_size),
                    "num_sites": int(num_sites),
                    "mode": mode,
                    "seed": int(seed),
                    "method": method,
                    "cutoff": int(cutoff),
                    "oracle_cutoff": int(oracle_cutoff),
                    "runtime": stats.get("total_time", ""),
                    "accuracy": metrics["accuracy"],
                    "mismatched_pixels": metrics["mismatched_pixels"],
                    "unassigned_pixels": metrics["unassigned_pixels"],
                }
            )
        _append_rows(rows, output_path)

    print("ML cutoff evaluation saved to {}".format(output_path))


if __name__ == "__main__":
    main()
