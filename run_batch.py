"""Run the default collection of research experiments."""

import argparse

from tqdm import tqdm

from run_experiment import run_single_experiment


def build_batch_plan(seed):
    plan = []

    for cutoff in [20, 40, 60, 80, 100, 120, 160]:
        plan.append(
            {
                "study": "cutoff",
                "grid_size": 512,
                "num_sites": 100,
                "mode": "random",
                "cutoff_radius": cutoff,
                "seed": seed,
            }
        )

    for mode in [
        "random",
        "clustered",
        "boundary",
        "close",
        "collinear_horizontal",
        "collinear_diagonal",
        "grid",
    ]:
        plan.append(
            {
                "study": "distribution",
                "grid_size": 512,
                "num_sites": 100,
                "mode": mode,
                "cutoff_radius": 80,
                "seed": seed,
            }
        )

    for site_count in [10, 50, 100, 500, 1000]:
        plan.append(
            {
                "study": "site_count",
                "grid_size": 512,
                "num_sites": site_count,
                "mode": "random",
                "cutoff_radius": 80,
                "seed": seed,
            }
        )

    for grid_size in [128, 256, 512]:
        plan.append(
            {
                "study": "grid_size",
                "grid_size": grid_size,
                "num_sites": 100,
                "mode": "random",
                "cutoff_radius": 80,
                "seed": seed,
            }
        )

    return plan


def build_parser():
    parser = argparse.ArgumentParser(description="Run the default Voronoi research batch.")
    parser.add_argument("--arch", type=str, default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--save-diff", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    plan = build_batch_plan(args.seed)

    for item in tqdm(plan, desc="Batch experiments"):
        prefix = "{study}_{mode}_{grid_size}_{num_sites}_cutoff{cutoff_radius}".format(**item)
        run_single_experiment(
            grid_size=item["grid_size"],
            num_sites=item["num_sites"],
            mode=item["mode"],
            cutoff_radius=item["cutoff_radius"],
            arch=args.arch,
            algorithm="proposed",
            seed=item["seed"],
            output_prefix=prefix,
            compare_bruteforce=True,
            save_images=args.save_images,
            save_diff=args.save_diff,
        )


if __name__ == "__main__":
    main()
