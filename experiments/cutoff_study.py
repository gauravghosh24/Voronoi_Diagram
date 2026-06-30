"""Cutoff radius study for the proposed circle-growing algorithm."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CSV_DIR, PLOTS_DIR
from run_experiment import run_single_experiment
from src.utils import ensure_dirs, timestamp
from src.visualize import plot_cutoff_results


def _plot_metric(df, x_column, y_column, ylabel, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy()
    plot_df[x_column] = pd.to_numeric(plot_df[x_column], errors="coerce")
    plot_df[y_column] = pd.to_numeric(plot_df[y_column], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_column, y_column]).sort_values(x_column)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(plot_df[x_column], plot_df[y_column], marker="o")
    ax.set_xlabel("Cutoff radius")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description="Run a cutoff-radius study.")
    parser.add_argument("--arch", type=str, default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--grid", type=int, default=512, dest="grid_size")
    parser.add_argument("--sites", type=int, default=100, dest="num_sites")
    parser.add_argument("--mode", type=str, default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[20, 40, 60, 80, 100, 120, 160])
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--save-diff", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    ensure_dirs()
    rows = []

    for cutoff in tqdm(args.cutoffs, desc="Cutoff study"):
        result = run_single_experiment(
            grid_size=args.grid_size,
            num_sites=args.num_sites,
            mode=args.mode,
            cutoff_radius=cutoff,
            arch=args.arch,
            algorithm="proposed",
            seed=args.seed,
            output_prefix="cutoff_{}_{}_{}_cutoff{}".format(args.mode, args.grid_size, args.num_sites, cutoff),
            compare_bruteforce=True,
            save_images=args.save_images,
            save_diff=args.save_diff,
        )
        rows.extend([row for row in result["rows"] if row["algorithm"] == "proposed"])

    df = pd.DataFrame(rows)
    study_csv = CSV_DIR / "cutoff_study_{}.csv".format(timestamp())
    df.to_csv(study_csv, index=False)

    _plot_metric(df, "cutoff_radius", "total_time", "Runtime (s)", PLOTS_DIR / "cutoff_vs_runtime.png")
    _plot_metric(
        df,
        "cutoff_radius",
        "accuracy_vs_bruteforce",
        "Accuracy vs brute force",
        PLOTS_DIR / "cutoff_vs_accuracy.png",
    )
    _plot_metric(
        df,
        "cutoff_radius",
        "unassigned_pixels",
        "Unassigned pixels",
        PLOTS_DIR / "cutoff_vs_unassigned.png",
    )
    plot_cutoff_results(study_csv, PLOTS_DIR / "cutoff_study_summary.png")


if __name__ == "__main__":
    main()
