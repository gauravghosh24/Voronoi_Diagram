"""Build a supervised dataset for ML cutoff-radius prediction."""

import argparse
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
from src.cutoff_model import calculate_oracle_cutoff_from_dist_map
from src.site_generators import generate_sites
from src.utils import ensure_dirs


DEFAULT_GRIDS = [128, 256, 512]
DEFAULT_SITE_COUNTS = [5, 10, 20, 50, 100, 200]
DEFAULT_MODES = [
    "random",
    "clustered",
    "boundary",
    "close",
    "collinear_horizontal",
    "collinear_diagonal",
    "grid",
]


def _seed_values(args):
    if args.seeds is not None:
        return [int(seed) for seed in args.seeds]
    return list(range(0, int(args.max_seed) + 1))


def _append_row(row, output_path):
    pd.DataFrame([row]).to_csv(
        output_path,
        mode="a",
        header=not output_path.exists(),
        index=False,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Build cutoff training dataset.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(CSV_DIR / "cutoff_training_dataset.csv"),
    )
    parser.add_argument("--grids", type=int, nargs="+", default=DEFAULT_GRIDS)
    parser.add_argument("--site-counts", type=int, nargs="+", default=DEFAULT_SITE_COUNTS)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-seed", type=int, default=50)
    parser.add_argument("--block-grid", type=int, default=4)
    return parser


def main():
    args = build_parser().parse_args()
    ensure_dirs()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = [
        (grid_size, num_sites, mode, seed)
        for grid_size in args.grids
        for num_sites in args.site_counts
        for mode in args.modes
        for seed in _seed_values(args)
    ]

    for grid_size, num_sites, mode, seed in tqdm(cases, desc="Cutoff dataset"):
        sites = generate_sites(grid_size, num_sites, mode=mode, seed=seed)
        _, dist_map = run_brute_force(grid_size, sites)
        row = extract_cutoff_features(grid_size, sites, block_grid=args.block_grid)
        row["mode"] = mode
        row["seed"] = int(seed)
        row["target_cutoff"] = calculate_oracle_cutoff_from_dist_map(dist_map)
        _append_row(row, output_path)

    print("Cutoff dataset saved to {}".format(output_path))


if __name__ == "__main__":
    main()
