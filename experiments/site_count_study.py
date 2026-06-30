"""Site-count study for the proposed algorithm."""

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


def build_parser():
    parser = argparse.ArgumentParser(description="Run a site-count study.")
    parser.add_argument("--arch", type=str, default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--grid", type=int, default=512, dest="grid_size")
    parser.add_argument("--site-counts", type=int, nargs="+", default=[10, 50, 100, 500, 1000])
    parser.add_argument("--mode", type=str, default="random")
    parser.add_argument("--cutoff", type=int, default=80, dest="cutoff_radius")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--save-diff", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    ensure_dirs()
    rows = []

    for num_sites in tqdm(args.site_counts, desc="Site count study"):
        result = run_single_experiment(
            grid_size=args.grid_size,
            num_sites=num_sites,
            mode=args.mode,
            cutoff_radius=args.cutoff_radius,
            arch=args.arch,
            algorithm="proposed",
            seed=args.seed,
            output_prefix="site_count_{}_{}_{}_cutoff{}".format(
                args.mode, args.grid_size, num_sites, args.cutoff_radius
            ),
            compare_bruteforce=True,
            save_images=args.save_images,
            save_diff=args.save_diff,
        )
        rows.extend([row for row in result["rows"] if row["algorithm"] == "proposed"])

    pd.DataFrame(rows).to_csv(CSV_DIR / "site_count_study_{}.csv".format(timestamp()), index=False)


if __name__ == "__main__":
    main()
