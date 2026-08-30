"""Feature extraction for ML cutoff-radius prediction."""

import numpy as np


def _zero_stats(prefix):
    return {
        "{}_min".format(prefix): 0.0,
        "{}_mean".format(prefix): 0.0,
        "{}_median".format(prefix): 0.0,
        "{}_max".format(prefix): 0.0,
        "{}_std".format(prefix): 0.0,
        "{}_p75".format(prefix): 0.0,
        "{}_p90".format(prefix): 0.0,
        "{}_p95".format(prefix): 0.0,
    }


def _nearest_neighbor_stats(sites):
    num_sites = int(sites.shape[0])
    if num_sites <= 1:
        return _zero_stats("nn")

    coords = sites.astype(np.float64, copy=False)
    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, np.inf)
    nearest = np.sqrt(np.min(dist2, axis=1))

    return {
        "nn_min": float(np.min(nearest)),
        "nn_mean": float(np.mean(nearest)),
        "nn_median": float(np.median(nearest)),
        "nn_max": float(np.max(nearest)),
        "nn_std": float(np.std(nearest)),
        "nn_p75": float(np.percentile(nearest, 75)),
        "nn_p90": float(np.percentile(nearest, 90)),
        "nn_p95": float(np.percentile(nearest, 95)),
    }


def _boundary_stats(grid_size, sites):
    if sites.shape[0] == 0:
        return {
            "boundary_min": 0.0,
            "boundary_mean": 0.0,
            "boundary_max": 0.0,
            "boundary_std": 0.0,
        }

    x = sites[:, 0].astype(np.float64, copy=False)
    y = sites[:, 1].astype(np.float64, copy=False)
    edge = float(grid_size - 1)
    distances = np.minimum.reduce([x, y, edge - x, edge - y])

    return {
        "boundary_min": float(np.min(distances)),
        "boundary_mean": float(np.mean(distances)),
        "boundary_max": float(np.max(distances)),
        "boundary_std": float(np.std(distances)),
    }


def _block_occupancy_stats(grid_size, sites, block_grid):
    block_grid = int(block_grid)
    if block_grid <= 0:
        raise ValueError("block_grid must be positive")

    counts = np.zeros((block_grid, block_grid), dtype=np.int64)
    if sites.shape[0] > 0:
        bx = np.minimum((sites[:, 0].astype(np.int64) * block_grid) // grid_size, block_grid - 1)
        by = np.minimum((sites[:, 1].astype(np.int64) * block_grid) // grid_size, block_grid - 1)
        np.add.at(counts, (by, bx), 1)

    flat = counts.ravel()
    return {
        "block_min": float(np.min(flat)),
        "block_max": float(np.max(flat)),
        "block_mean": float(np.mean(flat)),
        "block_std": float(np.std(flat)),
        "empty_block_count": int(np.count_nonzero(flat == 0)),
    }


def extract_cutoff_features(grid_size, sites, block_grid=4):
    """Return numeric features for global cutoff-radius prediction."""

    grid_size = int(grid_size)
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    sites = np.asarray(sites, dtype=np.int32)
    if sites.ndim != 2 or sites.shape[1] != 2:
        raise ValueError("sites must have shape (S, 2)")

    num_sites = int(sites.shape[0])
    features = {
        "grid_size": grid_size,
        "num_sites": num_sites,
        "site_density": float(num_sites / float(grid_size * grid_size)),
    }

    features.update(_nearest_neighbor_stats(sites))
    features.update(_boundary_stats(grid_size, sites))
    features.update(_block_occupancy_stats(grid_size, sites, block_grid))
    return features
