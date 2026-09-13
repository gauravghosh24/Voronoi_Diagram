"""Command-line runner for single Voronoi experiments."""

import argparse
import math
import time
from pathlib import Path

from config import CSV_DIR, DIFF_DIR, IMAGES_DIR, LOGS_DIR, MODELS_DIR
from src.metrics import compare_to_ground_truth
from src.site_generators import generate_sites, save_sites_csv
from src.utils import append_metrics_csv, ensure_dirs, save_json, timestamp
from src.visualize import save_difference_image, save_label_image


FILE_SUFFIX = {
    "proposed": "proposed",
    "brute_force": "bruteforce",
    "jfa": "jfa",
}
ML_MODEL_NOT_FOUND = "ML cutoff model not found. Run experiments/train_cutoff_model.py first."


def _normalize_algorithm(algorithm):
    algorithm = str(algorithm).lower()
    if algorithm == "bruteforce":
        return "brute_force"
    return algorithm


def _default_prefix(mode, grid_size, num_sites, cutoff_radius):
    if str(cutoff_radius).strip().lower() == "incremental":
        return "{}_{}_{}_cutoff_incremental".format(mode, grid_size, num_sites)
    return "{}_{}_{}_cutoff{}".format(mode, grid_size, num_sites, cutoff_radius)


def _diagonal_cutoff(grid_size):
    return int(math.ceil(math.sqrt(2.0) * (int(grid_size) - 1)))


def _resolve_cutoff_radius(cutoff_radius, grid_size, sites, mode, needs_proposed=True):
    cutoff_text = str(cutoff_radius).strip().lower()
    if cutoff_text == "incremental":
        return "incremental"

    if cutoff_text != "auto_ml":
        return int(cutoff_radius)

    if not needs_proposed:
        return 0

    model_path = MODELS_DIR / "cutoff_predictor.joblib"
    if not model_path.exists():
        raise FileNotFoundError(ML_MODEL_NOT_FOUND)

    from src.cutoff_features import extract_cutoff_features
    from src.cutoff_model import load_cutoff_model, predict_cutoff

    features = extract_cutoff_features(grid_size, sites)
    features["mode"] = mode
    model = load_cutoff_model(model_path)
    predicted_cutoff = predict_cutoff(
        model,
        features,
        safety_factor=1.10,
        safety_add=5,
        max_cutoff=_diagonal_cutoff(grid_size),
    )
    print("Predicted ML cutoff radius: {}".format(predicted_cutoff))
    return int(predicted_cutoff)


def _run_brute_force(grid_size, sites):
    from src.brute_force import run_brute_force

    print("Running brute force...")
    start = time.perf_counter()
    labels, dist_map = run_brute_force(grid_size, sites)
    elapsed = time.perf_counter() - start
    stats = {
        "algorithm": "brute_force",
        "grid_size": int(grid_size),
        "num_sites": int(len(sites)),
        "arch": "cpu",
        "total_time": elapsed,
        "fill_frames_time": "",
        "generate_result_time": "",
        "estimated_memory_mb": "",
        "unassigned_pixels": int((labels == -1).sum()),
    }
    return {"labels": labels, "dist_map": dist_map, "stats": stats}


def _run_proposed(grid_size, sites, cutoff_radius, arch, circle_backend="mdcs", step_radius=15):
    from src.proposed_circle_growing import run_proposed, run_proposed_incremental

    if str(cutoff_radius).strip().lower() == "incremental":
        print("Running proposed algorithm (Incremental Radius mode, backend={}, step={})...".format(
            circle_backend, step_radius
        ))
        labels, radius_map, unassigned_mask, stats = run_proposed_incremental(
            grid_size,
            sites,
            initial_radius=step_radius,
            step_radius=step_radius,
            circle_backend=circle_backend,
            arch=arch,
            use_parallel_reduction=True,
        )
        print("Incremental mode finished in {} iterations; final radius = {}; unassigned = {}".format(
            stats["iterations"], stats["final_radius"], stats["unassigned_pixels"]
        ))
    else:
        print("Running proposed algorithm (backend={})...".format(circle_backend))
        labels, radius_map, unassigned_mask, stats = run_proposed(
            grid_size,
            sites,
            int(cutoff_radius),
            circle_backend=circle_backend,
            arch=arch,
            use_parallel_reduction=True,
        )
    return {
        "labels": labels,
        "radius_map": radius_map,
        "unassigned_mask": unassigned_mask,
        "stats": stats,
    }


def _run_jfa(grid_size, sites, arch):
    from src.jfa import run_jfa

    print("Running JFA...")
    labels, dist_map, stats = run_jfa(grid_size, sites, arch=arch)
    return {"labels": labels, "dist_map": dist_map, "stats": stats}


def _build_csv_row(stats, metrics, grid_size, num_sites, mode, cutoff_radius, arch, seed):
    cutoff_val = stats.get("final_radius", cutoff_radius)
    try:
        cutoff_val = int(cutoff_val)
    except (ValueError, TypeError):
        cutoff_val = str(cutoff_val)

    row = {
        "timestamp": timestamp(),
        "algorithm": stats.get("algorithm", ""),
        "grid_size": int(grid_size),
        "num_sites": int(num_sites),
        "site_mode": mode,
        "cutoff_radius": cutoff_val,
        "arch": stats.get("arch", arch),
        "seed": int(seed),
        "total_time": stats.get("total_time", ""),
        "fill_frames_time": stats.get("fill_frames_time", ""),
        "generate_result_time": stats.get("generate_result_time", ""),
        "accuracy_vs_bruteforce": "",
        "mismatched_pixels": "",
        "unassigned_pixels": stats.get("unassigned_pixels", ""),
        "unassigned_percent": "",
        "estimated_memory_mb": stats.get("estimated_memory_mb", ""),
    }
    if metrics:
        row["accuracy_vs_bruteforce"] = metrics.get("accuracy", "")
        row["mismatched_pixels"] = metrics.get("mismatched_pixels", "")
        row["unassigned_pixels"] = metrics.get("unassigned_pixels", "")
        row["unassigned_percent"] = metrics.get("unassigned_percent", "")
    return row


def _save_images(results, sites, prefix):
    print("Saving images...")
    output_paths = {}
    for algorithm, result in results.items():
        suffix = FILE_SUFFIX[algorithm]
        path = IMAGES_DIR / "{}_{}.png".format(prefix, suffix)
        save_label_image(result["labels"], path, sites=sites)
        output_paths["{}_image".format(algorithm)] = str(path)
    return output_paths


def _save_diffs(results, gt_labels, prefix):
    output_paths = {}
    for algorithm in ("proposed", "jfa"):
        if algorithm in results:
            path = DIFF_DIR / "{}_{}_vs_bruteforce.png".format(prefix, FILE_SUFFIX[algorithm])
            save_difference_image(results[algorithm]["labels"], gt_labels, path)
            output_paths["{}_diff".format(algorithm)] = str(path)
    return output_paths


def run_single_experiment(
    grid_size=512,
    num_sites=100,
    mode="random",
    cutoff_radius=80,
    circle_backend="mdcs",
    step_radius=15,
    arch="gpu",
    algorithm="proposed",
    seed=42,
    output_prefix=None,
    compare_bruteforce=False,
    save_images=False,
    save_diff=False,
):
    """Run one experiment and persist images, metrics, and logs."""

    ensure_dirs()
    algorithm = _normalize_algorithm(algorithm)
    if algorithm not in ("proposed", "brute_force", "jfa", "all"):
        raise ValueError("Unsupported algorithm: {}".format(algorithm))

    grid_size = int(grid_size)
    num_sites = int(num_sites)
    seed = int(seed)
    requested_cutoff_radius = cutoff_radius
    needs_proposed = algorithm in ("proposed", "all")

    print("Grid size: {} x {}".format(grid_size, grid_size))
    print("Sites: {}".format(num_sites))
    print("Mode: {}".format(mode))
    print("Requested cutoff radius: {}".format(requested_cutoff_radius))
    print("Architecture: {}".format(arch))
    if needs_proposed:
        print("Circle backend: {}".format(circle_backend))

    sites = generate_sites(grid_size, num_sites, mode=mode, seed=seed)
    cutoff_radius = _resolve_cutoff_radius(
        requested_cutoff_radius,
        grid_size,
        sites,
        mode,
        needs_proposed=needs_proposed,
    )
    print("Cutoff radius: {}".format(cutoff_radius))

    prefix = output_prefix or _default_prefix(mode, grid_size, num_sites, cutoff_radius)
    sites_path = CSV_DIR / "{}_sites.csv".format(prefix)
    save_sites_csv(sites, sites_path)

    results = {}
    needs_brute_force = compare_bruteforce or algorithm in ("brute_force", "all")
    if needs_brute_force:
        results["brute_force"] = _run_brute_force(grid_size, sites)

    if algorithm in ("proposed", "all"):
        results["proposed"] = _run_proposed(
            grid_size,
            sites,
            cutoff_radius,
            arch,
            circle_backend=circle_backend,
            step_radius=step_radius,
        )

    if algorithm in ("jfa", "all"):
        results["jfa"] = _run_jfa(grid_size, sites, arch)

    if algorithm == "brute_force" and "brute_force" not in results:
        results["brute_force"] = _run_brute_force(grid_size, sites)

    gt_labels = results["brute_force"]["labels"] if "brute_force" in results else None
    result_metrics = {}
    for name, result in results.items():
        if gt_labels is not None:
            result_metrics[name] = compare_to_ground_truth(result["labels"], gt_labels)
        else:
            result_metrics[name] = None

    output_paths = {"sites_csv": str(sites_path)}
    if save_images:
        output_paths.update(_save_images(results, sites, prefix))
    if save_diff and gt_labels is not None:
        output_paths.update(_save_diffs(results, gt_labels, prefix))

    rows = []
    for name, result in results.items():
        row = _build_csv_row(
            result["stats"],
            result_metrics.get(name),
            grid_size,
            num_sites,
            mode,
            cutoff_radius,
            arch,
            seed,
        )
        rows.append(row)
        append_metrics_csv(row, CSV_DIR / "results.csv")

    if "proposed" in result_metrics and result_metrics["proposed"] is not None:
        proposed_metrics = result_metrics["proposed"]
        print("Accuracy proposed vs brute force: {:.6f}".format(proposed_metrics["accuracy"]))
        print("Unassigned pixels proposed: {}".format(proposed_metrics["unassigned_pixels"]))
        print("Total runtime proposed: {:.6f}s".format(results["proposed"]["stats"]["total_time"]))
    if "jfa" in result_metrics and result_metrics["jfa"] is not None:
        print("Accuracy JFA vs brute force: {:.6f}".format(result_metrics["jfa"]["accuracy"]))

    log_data = {
        "config": {
            "grid_size": grid_size,
            "num_sites": num_sites,
            "mode": mode,
            "cutoff_radius": cutoff_radius,
            "requested_cutoff_radius": requested_cutoff_radius,
            "circle_backend": circle_backend,
            "step_radius": step_radius,
            "arch": arch,
            "seed": seed,
            "algorithm": algorithm,
            "compare_bruteforce": bool(compare_bruteforce),
        },
        "sites": sites.tolist(),
        "outputs": output_paths,
        "results": {
            name: {
                "stats": result["stats"],
                "metrics_vs_bruteforce": result_metrics.get(name),
            }
            for name, result in results.items()
        },
    }
    log_path = LOGS_DIR / "{}_stats.json".format(prefix)
    save_json(log_data, log_path)
    output_paths["stats_json"] = str(log_path)

    print("Results saved to outputs/")
    return {
        "prefix": prefix,
        "sites": sites,
        "results": results,
        "metrics": result_metrics,
        "rows": rows,
        "outputs": output_paths,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Run a Voronoi construction experiment.")
    parser.add_argument("--grid", type=int, default=512, dest="grid_size")
    parser.add_argument("--sites", type=int, default=100, dest="num_sites")
    parser.add_argument("--mode", type=str, default="random")
    parser.add_argument(
        "--cutoff",
        type=str,
        default="80",
        dest="cutoff_radius",
        help="Cutoff radius integer, 'auto_ml', or 'incremental'",
    )
    parser.add_argument(
        "--circle-backend",
        type=str,
        default="mdcs",
        choices=["mdcs", "radius_band"],
        help="Circle generator: 'mdcs' (Algorithm 2) or 'radius_band'",
    )
    parser.add_argument(
        "--step-radius",
        type=int,
        default=15,
        help="Step increment for incremental radius mode (Algorithm 4, default: 15)",
    )
    parser.add_argument("--arch", type=str, default="gpu", choices=["gpu", "cpu"])
    parser.add_argument(
        "--algorithm",
        type=str,
        default="proposed",
        choices=["proposed", "brute_force", "bruteforce", "jfa", "all"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--compare-bruteforce", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--save-diff", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    run_single_experiment(
        grid_size=args.grid_size,
        num_sites=args.num_sites,
        mode=args.mode,
        cutoff_radius=args.cutoff_radius,
        circle_backend=args.circle_backend,
        step_radius=args.step_radius,
        arch=args.arch,
        algorithm=args.algorithm,
        seed=args.seed,
        output_prefix=args.output_prefix,
        compare_bruteforce=args.compare_bruteforce,
        save_images=args.save_images,
        save_diff=args.save_diff,
    )


if __name__ == "__main__":
    main()
