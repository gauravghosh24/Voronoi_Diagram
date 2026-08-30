"""Project-wide paths and constants."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
IMAGES_DIR = OUTPUT_DIR / "images"
DIFF_DIR = OUTPUT_DIR / "diff"
CSV_DIR = OUTPUT_DIR / "csv"
PLOTS_DIR = OUTPUT_DIR / "plots"
LOGS_DIR = OUTPUT_DIR / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_INF = 1_000_000
MEMORY_WARNING_MB = 1500.0

SUPPORTED_ALGORITHMS = ("proposed", "brute_force", "jfa", "all")

RESULT_COLUMNS = [
    "timestamp",
    "algorithm",
    "grid_size",
    "num_sites",
    "site_mode",
    "cutoff_radius",
    "arch",
    "seed",
    "total_time",
    "fill_frames_time",
    "generate_result_time",
    "accuracy_vs_bruteforce",
    "mismatched_pixels",
    "unassigned_pixels",
    "unassigned_percent",
    "estimated_memory_mb",
]
