"""Basic Jump Flood Algorithm implementation for comparison."""

import time

import numpy as np

from .taichi_init import init_taichi


def _highest_power_of_two_less_than(n):
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length() - 1)


def run_jfa(grid_size, sites, arch="gpu"):
    """Run a basic 8-neighbor Jump Flood Algorithm."""

    grid_size = int(grid_size)
    sites = np.asarray(sites, dtype=np.int32)
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if sites.ndim != 2 or sites.shape[1] != 2:
        raise ValueError("sites must have shape (S, 2)")

    num_sites = int(sites.shape[0])
    if num_sites == 0:
        labels = np.full((grid_size, grid_size), -1, dtype=np.int32)
        dist_map = np.full((grid_size, grid_size), np.iinfo(np.int32).max, dtype=np.int32)
        return labels, dist_map, {
            "algorithm": "jfa",
            "grid_size": grid_size,
            "num_sites": 0,
            "arch": arch,
            "total_time": 0.0,
            "passes": 0,
        }

    total_start = time.perf_counter()
    ti, active_arch = init_taichi(arch)

    inf_dist = min(np.iinfo(np.int32).max // 4, max(1_000_000, 2 * grid_size * grid_size + 1))

    site_x = ti.field(dtype=ti.i32, shape=num_sites)
    site_y = ti.field(dtype=ti.i32, shape=num_sites)
    labels_a = ti.field(dtype=ti.i32, shape=(grid_size, grid_size))
    labels_b = ti.field(dtype=ti.i32, shape=(grid_size, grid_size))
    dist_a = ti.field(dtype=ti.i32, shape=(grid_size, grid_size))
    dist_b = ti.field(dtype=ti.i32, shape=(grid_size, grid_size))

    site_x.from_numpy(sites[:, 0].astype(np.int32, copy=False))
    site_y.from_numpy(sites[:, 1].astype(np.int32, copy=False))

    @ti.kernel
    def initialize(
        labels_arg: ti.template(),
        dist_arg: ti.template(),
        scratch_labels_arg: ti.template(),
        scratch_dist_arg: ti.template(),
        site_x_arg: ti.template(),
        site_y_arg: ti.template(),
        num_sites_arg: ti.i32,
        grid_size_arg: ti.i32,
        inf_arg: ti.i32,
    ):
        for y, x in labels_arg:
            labels_arg[y, x] = -1
            dist_arg[y, x] = inf_arg
            scratch_labels_arg[y, x] = -1
            scratch_dist_arg[y, x] = inf_arg

        for s in range(num_sites_arg):
            x = site_x_arg[s]
            y = site_y_arg[s]
            if x >= 0 and x < grid_size_arg and y >= 0 and y < grid_size_arg:
                labels_arg[y, x] = s
                dist_arg[y, x] = 0

    @ti.kernel
    def jump_pass(
        src_labels: ti.template(),
        src_dist: ti.template(),
        dst_labels: ti.template(),
        dst_dist: ti.template(),
        site_x_arg: ti.template(),
        site_y_arg: ti.template(),
        grid_size_arg: ti.i32,
        jump_arg: ti.i32,
    ):
        for y, x in src_labels:
            best_label = src_labels[y, x]
            best_dist = src_dist[y, x]

            for oy in ti.static(range(-1, 2)):
                for ox in ti.static(range(-1, 2)):
                    nx = x + ox * jump_arg
                    ny = y + oy * jump_arg
                    if nx >= 0 and nx < grid_size_arg and ny >= 0 and ny < grid_size_arg:
                        candidate = src_labels[ny, nx]
                        if candidate >= 0:
                            dx = x - site_x_arg[candidate]
                            dy = y - site_y_arg[candidate]
                            candidate_dist = dx * dx + dy * dy
                            if (
                                best_label < 0
                                or candidate_dist < best_dist
                                or (candidate_dist == best_dist and candidate < best_label)
                            ):
                                best_label = candidate
                                best_dist = candidate_dist

            dst_labels[y, x] = best_label
            dst_dist[y, x] = best_dist

    initialize(labels_a, dist_a, labels_b, dist_b, site_x, site_y, num_sites, grid_size, inf_dist)
    ti.sync()

    current_labels = labels_a
    current_dist = dist_a
    scratch_labels = labels_b
    scratch_dist = dist_b

    passes = []
    jump = _highest_power_of_two_less_than(grid_size)
    while jump >= 1:
        passes.append(jump)
        jump_pass(current_labels, current_dist, scratch_labels, scratch_dist, site_x, site_y, grid_size, jump)
        ti.sync()
        current_labels, scratch_labels = scratch_labels, current_labels
        current_dist, scratch_dist = scratch_dist, current_dist
        jump //= 2

    # One extra local correction pass helps repair common JFA artifacts without
    # turning the method into brute force.
    jump_pass(current_labels, current_dist, scratch_labels, scratch_dist, site_x, site_y, grid_size, 1)
    ti.sync()
    current_labels, scratch_labels = scratch_labels, current_labels
    current_dist, scratch_dist = scratch_dist, current_dist
    passes.append(1)

    labels = current_labels.to_numpy()
    dist_map = current_dist.to_numpy()
    total_time = time.perf_counter() - total_start

    stats = {
        "algorithm": "jfa",
        "grid_size": grid_size,
        "num_sites": num_sites,
        "arch": active_arch,
        "requested_arch": arch,
        "total_time": total_time,
        "passes": len(passes),
        "jump_schedule": passes,
    }
    return labels, dist_map, stats
