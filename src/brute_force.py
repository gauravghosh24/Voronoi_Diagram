"""Ground-truth brute force Voronoi construction."""

import numpy as np


def run_brute_force(grid_size, sites):
    """Compute the exact nearest site for every pixel.

    Parameters
    ----------
    grid_size:
        Width and height of the square grid.
    sites:
        Array of shape ``(S, 2)`` using ``(x, y)`` coordinates.

    Returns
    -------
    labels:
        ``labels[y, x]`` is the nearest site index, or ``-1`` when no sites exist.
    dist_map:
        Squared Euclidean distance to the nearest site.
    """

    grid_size = int(grid_size)
    sites = np.asarray(sites, dtype=np.int64)
    labels = np.full((grid_size, grid_size), -1, dtype=np.int32)
    dist_map = np.full((grid_size, grid_size), np.iinfo(np.int64).max, dtype=np.int64)

    if sites.size == 0:
        return labels, dist_map

    num_sites = sites.shape[0]
    max_chunk_bytes = 64 * 1024 * 1024
    bytes_per_row = max(1, grid_size * num_sites * np.dtype(np.int64).itemsize)
    chunk_rows = max(1, min(grid_size, max_chunk_bytes // bytes_per_row))

    site_x = sites[:, 0][None, None, :]
    site_y = sites[:, 1][None, None, :]
    xs = np.arange(grid_size, dtype=np.int64)[None, :, None]

    for y0 in range(0, grid_size, chunk_rows):
        y1 = min(grid_size, y0 + chunk_rows)
        ys = np.arange(y0, y1, dtype=np.int64)[:, None, None]
        dist = (xs - site_x) ** 2 + (ys - site_y) ** 2

        # NumPy argmin returns the first minimum, giving the required
        # smaller-site-index tie break.
        nearest = np.argmin(dist, axis=2).astype(np.int32)
        nearest_dist = np.take_along_axis(dist, nearest[:, :, None], axis=2)[:, :, 0]

        labels[y0:y1, :] = nearest
        dist_map[y0:y1, :] = nearest_dist

    return labels, dist_map
