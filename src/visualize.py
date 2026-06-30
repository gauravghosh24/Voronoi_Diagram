"""Image and plot helpers for Voronoi outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .metrics import difference_mask


def labels_to_color_image(labels, sites=None, seed=42):
    """Convert a label matrix to an RGB ``uint8`` image array."""

    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array")

    height, width = labels.shape
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    assigned = labels >= 0

    if np.any(assigned):
        max_label = int(labels[assigned].max())
        rng = np.random.default_rng(seed)
        colors = rng.integers(35, 235, size=(max_label + 1, 3), dtype=np.uint8)
        image[assigned] = colors[labels[assigned]]

    if sites is not None:
        pil_image = Image.fromarray(image, mode="RGB")
        draw = ImageDraw.Draw(pil_image)
        radius = max(1, min(height, width) // 128)
        for x, y in np.asarray(sites, dtype=np.int32):
            draw.ellipse(
                (int(x) - radius, int(y) - radius, int(x) + radius, int(y) + radius),
                fill=(0, 0, 0),
            )
        image = np.asarray(pil_image)

    return image


def save_label_image(labels, path, sites=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(labels_to_color_image(labels, sites=sites), mode="RGB").save(path)


def save_difference_image(pred_labels, gt_labels, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = difference_mask(pred_labels, gt_labels)
    image = np.zeros(mask.shape + (3,), dtype=np.uint8)
    image[mask == 1] = (255, 0, 0)
    image[mask == 2] = (255, 255, 255)
    Image.fromarray(image, mode="RGB").save(path)


def plot_cutoff_results(csv_path, output_path):
    """Create a three-panel cutoff study plot from a results CSV."""

    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "algorithm" in df.columns:
        df = df[df["algorithm"] == "proposed"]
    df = df.copy()
    df["cutoff_radius"] = pd.to_numeric(df["cutoff_radius"], errors="coerce")
    df["total_time"] = pd.to_numeric(df["total_time"], errors="coerce")
    df["accuracy_vs_bruteforce"] = pd.to_numeric(df["accuracy_vs_bruteforce"], errors="coerce")
    df["unassigned_pixels"] = pd.to_numeric(df["unassigned_pixels"], errors="coerce")
    df = df.dropna(subset=["cutoff_radius"]).sort_values("cutoff_radius")

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(df["cutoff_radius"], df["total_time"], marker="o")
    axes[0].set_ylabel("Runtime (s)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["cutoff_radius"], df["accuracy_vs_bruteforce"], marker="o")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["cutoff_radius"], df["unassigned_pixels"], marker="o")
    axes[2].set_xlabel("Cutoff radius")
    axes[2].set_ylabel("Unassigned pixels")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
