"""Deterministic site distributions for Voronoi experiments."""

from pathlib import Path

import numpy as np
import pandas as pd


def _check_capacity(grid_size, num_sites, allow_duplicates):
    if not allow_duplicates and num_sites > grid_size * grid_size:
        raise ValueError("Cannot generate {} unique sites on a {}x{} grid".format(num_sites, grid_size, grid_size))


def _clip_points(points, grid_size):
    arr = np.asarray(points, dtype=np.int64).reshape(-1, 2)
    if arr.size == 0:
        return arr.astype(np.int32)
    arr[:, 0] = np.clip(arr[:, 0], 0, grid_size - 1)
    arr[:, 1] = np.clip(arr[:, 1], 0, grid_size - 1)
    return arr.astype(np.int32)


def _random_unique(grid_size, num_sites, rng):
    choices = rng.choice(grid_size * grid_size, size=num_sites, replace=False)
    xs = choices % grid_size
    ys = choices // grid_size
    return np.stack([xs, ys], axis=1).astype(np.int32)


def _deduplicate_and_fill(points, grid_size, num_sites, rng, allow_duplicates=False):
    points = _clip_points(points, grid_size)
    if num_sites == 0:
        return np.empty((0, 2), dtype=np.int32)

    if allow_duplicates:
        if len(points) < num_sites:
            extra = rng.integers(0, grid_size, size=(num_sites - len(points), 2), endpoint=False)
            points = np.vstack([points, extra.astype(np.int32)])
        return points[:num_sites].astype(np.int32)

    selected = []
    seen = set()
    for x, y in points:
        key = (int(x), int(y))
        if key not in seen:
            selected.append(key)
            seen.add(key)
            if len(selected) == num_sites:
                return np.asarray(selected, dtype=np.int32)

    while len(selected) < num_sites:
        x = int(rng.integers(0, grid_size))
        y = int(rng.integers(0, grid_size))
        key = (x, y)
        if key not in seen:
            selected.append(key)
            seen.add(key)

    return np.asarray(selected, dtype=np.int32)


def _clustered_points(grid_size, num_sites, rng):
    cluster_count = int(round(num_sites * 0.7))
    random_count = num_sites - cluster_count
    margin = max(1, grid_size // 8)
    low = margin if grid_size > 2 * margin else 0
    high = grid_size - margin if grid_size > 2 * margin else grid_size
    centers = rng.integers(low, max(low + 1, high), size=(2, 2), endpoint=False)
    sigma = max(1.0, grid_size * 0.06)

    clustered = []
    for _ in range(max(cluster_count * 4, cluster_count)):
        center = centers[int(rng.integers(0, len(centers)))]
        point = np.rint(rng.normal(loc=center, scale=sigma, size=2)).astype(np.int64)
        clustered.append(point)

    random_points = rng.integers(0, grid_size, size=(max(random_count * 3, random_count), 2), endpoint=False)
    return np.vstack([np.asarray(clustered, dtype=np.int64), random_points])


def _boundary_points(grid_size, num_sites, rng):
    margin = max(1, grid_size // 24)
    points = [(0, 0), (grid_size - 1, 0), (0, grid_size - 1), (grid_size - 1, grid_size - 1)]

    for _ in range(max(num_sites * 5, num_sites)):
        edge = int(rng.integers(0, 4))
        offset = int(rng.integers(0, margin + 1))
        if edge == 0:
            points.append((int(rng.integers(0, grid_size)), offset))
        elif edge == 1:
            points.append((int(rng.integers(0, grid_size)), grid_size - 1 - offset))
        elif edge == 2:
            points.append((offset, int(rng.integers(0, grid_size))))
        else:
            points.append((grid_size - 1 - offset, int(rng.integers(0, grid_size))))
    return np.asarray(points, dtype=np.int64)


def _close_points(grid_size, num_sites, rng):
    center = rng.integers(0, grid_size, size=2, endpoint=False)
    jitter = max(1, grid_size // 80)
    points = []
    for _ in range(max(num_sites * 8, num_sites)):
        delta = rng.integers(-jitter, jitter + 1, size=2)
        points.append(center + delta)
    return np.asarray(points, dtype=np.int64)


def _collinear_horizontal(grid_size, num_sites):
    count = min(num_sites, grid_size)
    xs = np.linspace(0, grid_size - 1, count, dtype=np.int64)
    ys = np.full(count, grid_size // 2, dtype=np.int64)
    return np.stack([xs, ys], axis=1)


def _collinear_diagonal(grid_size, num_sites):
    count = min(num_sites, grid_size)
    coords = np.linspace(0, grid_size - 1, count, dtype=np.int64)
    return np.stack([coords, coords], axis=1)


def _grid_points(grid_size, num_sites):
    side = int(np.ceil(np.sqrt(max(1, num_sites))))
    coords = np.linspace(0, grid_size - 1, side, dtype=np.int64)
    xx, yy = np.meshgrid(coords, coords)
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def _symmetric_tie_points(grid_size, num_sites):
    center = (grid_size - 1) // 2
    distances = [max(1, grid_size // 4), max(1, grid_size // 6), max(1, grid_size // 3)]
    points = [(center, center)]
    for d in distances:
        points.extend(
            [
                (center - d, center),
                (center + d, center),
                (center, center - d),
                (center, center + d),
                (center - d, center - d),
                (center + d, center + d),
                (center - d, center + d),
                (center + d, center - d),
            ]
        )
    return np.asarray(points[: max(num_sites * 2, len(points))], dtype=np.int64)


def generate_sites(grid_size, num_sites, mode="random", seed=42, allow_duplicates=False):
    """Generate ``num_sites`` points as ``(x, y)`` coordinates."""

    grid_size = int(grid_size)
    num_sites = int(num_sites)
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if num_sites < 0:
        raise ValueError("num_sites must be non-negative")

    _check_capacity(grid_size, num_sites, allow_duplicates)
    rng = np.random.default_rng(seed)
    mode = str(mode).lower()

    if num_sites == 0:
        return np.empty((0, 2), dtype=np.int32)

    if mode == "random":
        if allow_duplicates:
            return rng.integers(0, grid_size, size=(num_sites, 2), endpoint=False).astype(np.int32)
        return _random_unique(grid_size, num_sites, rng)
    if mode == "clustered":
        candidates = _clustered_points(grid_size, num_sites, rng)
    elif mode == "boundary":
        candidates = _boundary_points(grid_size, num_sites, rng)
    elif mode == "close":
        candidates = _close_points(grid_size, num_sites, rng)
    elif mode == "collinear_horizontal":
        candidates = _collinear_horizontal(grid_size, num_sites)
    elif mode == "collinear_diagonal":
        candidates = _collinear_diagonal(grid_size, num_sites)
    elif mode == "grid":
        candidates = _grid_points(grid_size, num_sites)
    elif mode == "symmetric_tie":
        candidates = _symmetric_tie_points(grid_size, num_sites)
    else:
        raise ValueError("Unsupported site mode: {}".format(mode))

    return _deduplicate_and_fill(candidates, grid_size, num_sites, rng, allow_duplicates=allow_duplicates)


def save_sites_csv(sites, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sites = np.asarray(sites, dtype=np.int32)
    df = pd.DataFrame(
        {
            "site_index": np.arange(len(sites), dtype=np.int32),
            "x": sites[:, 0] if len(sites) else [],
            "y": sites[:, 1] if len(sites) else [],
        }
    )
    df.to_csv(path, index=False)
