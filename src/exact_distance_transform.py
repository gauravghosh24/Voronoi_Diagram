"""Exact Euclidean Voronoi labels via separable distance transform.

The transform minimizes ``distance_squared * scale + site_index`` so that
nearest-distance ties are resolved the same way as the rest of the project:
the lower site index wins.
"""

import time

import numpy as np


INF_COST = np.iinfo(np.int64).max // 4


def _edt_1d_scaled(values, scale, inf_cost=INF_COST):
    """Return the 1D squared distance transform for scaled integer costs."""

    values = np.asarray(values, dtype=np.int64)
    n = int(values.shape[0])
    output = np.empty(n, dtype=np.int64)
    finite_positions = np.nonzero(values < inf_cost)[0]

    if finite_positions.size == 0:
        output.fill(inf_cost)
        return output

    v = np.empty(n, dtype=np.int64)
    z = np.empty(n + 1, dtype=np.float64)

    k = 0
    v[0] = int(finite_positions[0])
    z[0] = -np.inf
    z[1] = np.inf

    scaled = float(scale)
    for q_value in finite_positions[1:]:
        q = int(q_value)

        while True:
            p = int(v[k])
            numerator = (
                float(values[q])
                + scaled * q * q
                - float(values[p])
                - scaled * p * p
            )
            denominator = 2.0 * scaled * (q - p)
            intersection = numerator / denominator

            if intersection <= z[k]:
                k -= 1
                if k < 0:
                    k = 0
                    v[0] = q
                    z[0] = -np.inf
                    z[1] = np.inf
                    break
            else:
                k += 1
                v[k] = q
                z[k] = intersection
                z[k + 1] = np.inf
                break

    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1

        p = int(v[k])
        output[q] = int(scale) * (q - p) * (q - p) + values[p]

    return output


def run_exact_distance_transform(grid_size, sites):
    """Compute exact Voronoi labels and squared distances for integer sites."""

    start = time.perf_counter()
    grid_size = int(grid_size)
    sites = np.asarray(sites, dtype=np.int32)

    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if sites.ndim != 2 or sites.shape[1] != 2:
        raise ValueError("sites must have shape (S, 2)")

    num_sites = int(sites.shape[0])
    labels = np.full((grid_size, grid_size), -1, dtype=np.int32)
    dist_map = np.full((grid_size, grid_size), np.iinfo(np.int64).max, dtype=np.int64)

    if num_sites == 0:
        return labels, dist_map, {
            "algorithm": "exact_distance_transform",
            "grid_size": grid_size,
            "num_sites": 0,
            "total_time": 0.0,
        }

    scale = num_sites + 1
    site_costs = np.full((grid_size, grid_size), INF_COST, dtype=np.int64)

    for site_index, (x, y) in enumerate(sites):
        x = int(x)
        y = int(y)
        if 0 <= x < grid_size and 0 <= y < grid_size:
            site_costs[y, x] = min(site_costs[y, x], int(site_index))

    row_costs = np.empty_like(site_costs)
    for y in range(grid_size):
        row_costs[y, :] = _edt_1d_scaled(site_costs[y, :], scale)

    final_costs = np.empty_like(site_costs)
    for x in range(grid_size):
        final_costs[:, x] = _edt_1d_scaled(row_costs[:, x], scale)

    reached = final_costs < INF_COST
    labels[reached] = (final_costs[reached] % scale).astype(np.int32)
    dist_map[reached] = final_costs[reached] // scale

    elapsed = time.perf_counter() - start
    return labels, dist_map, {
        "algorithm": "exact_distance_transform",
        "grid_size": grid_size,
        "num_sites": num_sites,
        "scale": scale,
        "total_time": elapsed,
    }
