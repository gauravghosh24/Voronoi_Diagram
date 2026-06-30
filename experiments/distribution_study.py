"""Distribution study for site placement modes."""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CSV_DIR
from run_experiment import run_single_experiment
from src.utils import ensure_dirs, timestamp


DEFAULT_MODES = [
    "random",
    "clustered",
    "boundary",
    "close",
    "collinear_horizontal",
    "collinear_diagonal",
    "grid",
]


def build_parser():
    parser = argparse.ArgumentParser(description="Run a site-distribution study.")
    parser.add_argument("--arch", type=str, default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--grid", type=int, default=512, dest="grid_size")
    parser.add_argument("--sites", type=int, default=100, dest="num_sites")
    parser.add_argument("--cutoff", type=int, default=80, dest="cutoff_radius")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--save-diff", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    ensure_dirs()
    rows = []

    for mode in tqdm(args.modes, desc="Distribution study"):
        result = run_single_experiment(
            grid_size=args.grid_size,
            num_sites=args.num_sites,
            mode=mode,
            cutoff_radius=args.cutoff_radius,
            arch=args.arch,
            algorithm="proposed",
            seed=args.seed,
            output_prefix="distribution_{}_{}_{}_cutoff{}".format(
                mode, args.grid_size, args.num_sites, args.cutoff_radius
            ),
            compare_bruteforce=True,
            save_images=not args.no_images,
            save_diff=args.save_diff,
        )
        rows.extend([row for row in result["rows"] if row["algorithm"] == "proposed"])

    pd.DataFrame(rows).to_csv(CSV_DIR / "distribution_study_{}.csv".format(timestamp()), index=False)


if __name__ == "__main__":
    main()
