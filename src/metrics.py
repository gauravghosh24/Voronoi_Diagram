"""Correctness, difference, and memory metrics."""

import numpy as np


def compare_to_ground_truth(pred_labels, gt_labels):
    """Compare a predicted label map to brute-force ground truth."""

    pred = np.asarray(pred_labels)
    gt = np.asarray(gt_labels)
    if pred.shape != gt.shape:
        raise ValueError("pred_labels and gt_labels must have the same shape")

    total_pixels = int(pred.size)
    same = pred == gt
    correct_pixels = int(np.count_nonzero(same))
    mismatched_pixels = int(total_pixels - correct_pixels)
    unassigned_pixels = int(np.count_nonzero(pred == -1))

    accuracy = float(correct_pixels / total_pixels) if total_pixels else 0.0
    unassigned_percent = float(100.0 * unassigned_pixels / total_pixels) if total_pixels else 0.0

    return {
        "total_pixels": total_pixels,
        "correct_pixels": correct_pixels,
        "mismatched_pixels": mismatched_pixels,
        "accuracy": accuracy,
        "unassigned_pixels": unassigned_pixels,
        "unassigned_percent": unassigned_percent,
    }


def difference_mask(pred_labels, gt_labels):
    """Return 0 for same, 1 for mismatch, and 2 for unassigned."""

    pred = np.asarray(pred_labels)
    gt = np.asarray(gt_labels)
    if pred.shape != gt.shape:
        raise ValueError("pred_labels and gt_labels must have the same shape")

    mask = np.zeros(pred.shape, dtype=np.uint8)
    mask[pred != gt] = 1
    mask[pred == -1] = 2
    return mask


def estimate_memory_mb(num_sites, grid_size, dtype_bytes=4):
    return float(num_sites) * float(grid_size) * float(grid_size) * float(dtype_bytes) / (1024.0 * 1024.0)
