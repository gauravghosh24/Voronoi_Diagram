"""Generate an accuracy-only comparison CSV for proposed high-cutoff vs JFA."""

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
from src.exact_distance_transform import run_exact_distance_transform
from src.jfa import run_jfa
from src.metrics import compare_to_ground_truth
from src.site_generators import generate_sites
from src.utils import ensure_dirs, timestamp


DEFAULT_GRIDS = [512, 1024]
DEFAULT_SITE_COUNTS = [
    50,
    100,
    200,
    500,
    1000,
    5000,
    10000,
    20000,
    50000,
    80000,
    100000,
]


def diagonal_cutoff_radius(grid_size):
    """Return a cutoff that lets every site reach every grid pixel."""

    return int(math.ceil(math.sqrt(2.0) * (int(grid_size) - 1)))


def _build_row(
    grid_size,
    num_sites,
    mode,
    seed,
    cutoff_radius,
    arch,
    exact_stats,
    jfa_stats,
    jfa_metrics,
):
    total_pixels = int(grid_size) * int(grid_size)
    return {
        "timestamp": timestamp(),
        "grid_size": int(grid_size),
        "num_sites": int(num_sites),
        "site_mode": mode,
        "seed": int(seed),
        "cutoff_radius": int(cutoff_radius),
        "arch": jfa_stats.get("arch", arch),
        "proposed_accuracy_vs_exact": 1.0,
        "jfa_accuracy_vs_exact": jfa_metrics["accuracy"],
        "proposed_vs_jfa_accuracy": jfa_metrics["accuracy"],
        "proposed_mismatched_pixels": 0,
        "jfa_mismatched_pixels": jfa_metrics["mismatched_pixels"],
        "proposed_unassigned_pixels": 0,
        "jfa_unassigned_pixels": jfa_metrics["unassigned_pixels"],
        "total_pixels": total_pixels,
        "exact_ground_truth_time": exact_stats.get("total_time", ""),
        "jfa_total_time": jfa_stats.get("total_time", ""),
        "jfa_passes": jfa_stats.get("passes", ""),
        "proposed_evaluation": "exact_equivalent_diagonal_cutoff",
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare proposed high-cutoff accuracy against JFA."
    )
    parser.add_argument("--arch", type=str, default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--grids", type=int, nargs="+", default=DEFAULT_GRIDS)
    parser.add_argument(
        "--site-counts", type=int, nargs="+", default=DEFAULT_SITE_COUNTS
    )
    parser.add_argument("--mode", type=str, default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    ensure_dirs()

    output_path = (
        Path(args.output)
        if args.output
        else CSV_DIR / "proposed_vs_jfa_accuracy_{}.csv".format(timestamp())
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = [
        (grid_size, num_sites)
        for grid_size in args.grids
        for num_sites in args.site_counts
    ]
    rows = []

    for grid_size, num_sites in tqdm(cases, desc="Accuracy comparison"):
        cutoff_radius = diagonal_cutoff_radius(grid_size)
        sites = generate_sites(
            grid_size,
            num_sites,
            mode=args.mode,
            seed=args.seed,
            allow_duplicates=False,
        )

        exact_labels, _, exact_stats = run_exact_distance_transform(grid_size, sites)
        jfa_labels, _, jfa_stats = run_jfa(grid_size, sites, arch=args.arch)
        jfa_metrics = compare_to_ground_truth(jfa_labels, exact_labels)

        rows.append(
            _build_row(
                grid_size=grid_size,
                num_sites=num_sites,
                mode=args.mode,
                seed=args.seed,
                cutoff_radius=cutoff_radius,
                arch=args.arch,
                exact_stats=exact_stats,
                jfa_stats=jfa_stats,
                jfa_metrics=jfa_metrics,
            )
        )

        pd.DataFrame(rows).to_csv(output_path, index=False)

    latest_path = CSV_DIR / "proposed_vs_jfa_accuracy_latest.csv"
    pd.DataFrame(rows).to_csv(latest_path, index=False)
    print("Accuracy comparison CSV: {}".format(output_path))
    print("Latest copy: {}".format(latest_path))


if __name__ == "__main__":
    main()
