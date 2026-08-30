"""Shared filesystem and serialization helpers."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import CSV_DIR, DIFF_DIR, IMAGES_DIR, LOGS_DIR, MODELS_DIR, OUTPUT_DIR, PLOTS_DIR, RESULT_COLUMNS


def ensure_dirs():
    """Create all standard output directories."""

    for path in (OUTPUT_DIR, IMAGES_DIR, DIFF_DIR, CSV_DIR, PLOTS_DIR, LOGS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Object of type {} is not JSON serializable".format(type(value).__name__))


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=_json_default)


def append_metrics_csv(metrics, path):
    """Append one metrics dictionary to a CSV with stable column ordering."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {column: metrics.get(column, "") for column in RESULT_COLUMNS}
    for key, value in metrics.items():
        if key not in row:
            row[key] = value

    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
