"""Paper-inspired digital circle growing Voronoi construction.

This implementation is an MDCS-inspired radius-band version.

It does not implement the exact square-number MDCS procedure from the paper.
Instead, it assigns every pixel inside the cutoff disk to its nearest integer
radius band using rounded Euclidean distance.

This prevents absentee pixels/holes and gives a correctness-first baseline.

Important:
    If cutoff_radius is large enough so that every pixel is reached by its
    true nearest site, then the final exact squared-distance tie-breaking
    makes this output match brute force very closely, usually exactly except
    for intentional tie-breaking behavior.
"""

import time
import warnings
from pathlib import Path

import numpy as np

from config import DEFAULT_INF, MEMORY_WARNING_MB
from .metrics import estimate_memory_mb
from .taichi_init import init_taichi


def _validate_inputs(grid_size, sites, cutoff_radius):
    """Validate and normalize inputs."""
    grid_size = int(grid_size)
    cutoff_radius = int(cutoff_radius)

    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    if cutoff_radius < 0:
        raise ValueError("cutoff_radius must be non-negative")

    sites = np.asarray(sites, dtype=np.int32)

    if sites.ndim != 2 or sites.shape[1] != 2:
        raise ValueError("sites must have shape (S, 2), where each site is (x, y)")

    return grid_size, sites, cutoff_radius


def run_proposed(
    grid_size,
    sites,
    cutoff_radius,
    arch="gpu",
    use_parallel_reduction=True,
    save_debug_frames=False,
):
    """Run the proposed digital-circle-growing Voronoi baseline.

    Parameters
    ----------
    grid_size : int
        Width and height of the square grid.

    sites : np.ndarray
        Array of shape (S, 2). Each site is represented as (x, y).

    cutoff_radius : int
        Maximum radius up to which each site grows.

    arch : str
        "gpu" or "cpu".

    use_parallel_reduction : bool
        Kept for API compatibility. Current implementation uses a simple
        per-pixel site loop. It can be replaced later with tree reduction.

    save_debug_frames : bool
        If True, saves the full frames[S, H, W] array to outputs/logs.

    Returns
    -------
    labels : np.ndarray
        Shape (grid_size, grid_size). labels[y, x] = nearest site index.
        If unassigned, label is -1.

    radius_map : np.ndarray
        Shape (grid_size, grid_size). Minimum rounded radius value.

    unassigned_mask : np.ndarray
        Boolean mask where True means pixel was not reached by any site.

    stats : dict
        Runtime and experiment information.
    """

    grid_size, sites, cutoff_radius = _validate_inputs(
        grid_size, sites, cutoff_radius
    )

    num_sites = int(sites.shape[0])

    # Empty input case
    if num_sites == 0:
        labels = np.full((grid_size, grid_size), -1, dtype=np.int32)
        radius_map = np.full((grid_size, grid_size), DEFAULT_INF, dtype=np.int32)
        unassigned_mask = np.ones((grid_size, grid_size), dtype=bool)

        stats = {
            "algorithm": "proposed",
            "grid_size": grid_size,
            "num_sites": 0,
            "cutoff_radius": cutoff_radius,
            "arch": arch,
            "requested_arch": arch,
            "taichi_init_time": 0.0,
            "fill_frames_time": 0.0,
            "generate_result_time": 0.0,
            "algorithm_only_time": 0.0,
            "total_time": 0.0,
            "unassigned_pixels": int(unassigned_mask.sum()),
            "estimated_memory_mb": 0.0,
            "reduction_mode": "none",
            "use_parallel_reduction_requested": bool(use_parallel_reduction),
            "circle_mode": "radius_band_fill",
            "tie_breaking": "exact_squared_distance_then_lower_site_index",
        }

        return labels, radius_map, unassigned_mask, stats

    total_start = time.perf_counter()

    estimated_mb = estimate_memory_mb(num_sites, grid_size)

    if estimated_mb > MEMORY_WARNING_MB:
        message = (
            "Estimated proposed frames memory is {:.1f} MB for S={}, grid={}. "
            "This may exceed GPU memory."
        ).format(estimated_mb, num_sites, grid_size)

        warnings.warn(message)
        print("WARNING: {}".format(message))

    # Initialize Taichi
    init_start = time.perf_counter()
    ti, active_arch = init_taichi(arch)
    init_time = time.perf_counter() - init_start

    # Taichi fields
    sites_field = ti.field(dtype=ti.i32, shape=(num_sites, 2))

    # frames[s, y, x] stores rounded radius at which site s reaches pixel (x, y)
    frames = ti.field(dtype=ti.i32, shape=(num_sites, grid_size, grid_size))

    labels_field = ti.field(dtype=ti.i32, shape=(grid_size, grid_size))
    radius_field = ti.field(dtype=ti.i32, shape=(grid_size, grid_size))
    unassigned_field = ti.field(dtype=ti.i32, shape=(grid_size, grid_size))

    sites_field.from_numpy(sites.astype(np.int32, copy=False))

    diameter = 2 * cutoff_radius + 1

    @ti.kernel
    def initialize_frames(
        frames_arg: ti.template(),
        inf_arg: ti.i32,
    ):
        """Set all frame values to INF."""
        for s, y, x in frames_arg:
            frames_arg[s, y, x] = inf_arg

    @ti.kernel
    def fill_radius_frames(
        frames_arg: ti.template(),
        sites_arg: ti.template(),
        cutoff_arg: ti.i32,
        grid_size_arg: ti.i32,
    ):
        """Fill each site's frame using radius-band digital circle growing.

        For each site, we scan a square box of side length:

            2 * cutoff_radius + 1

        For every offset (dx, dy), compute its Euclidean distance from the
        site center and round it to the nearest integer radius:

            radius = floor(sqrt(dx^2 + dy^2) + 0.5)

        This assigns every pixel inside the cutoff disk to one integer radius
        band, avoiding absentee pixels.
        """

        for s, oy, ox in ti.ndrange(num_sites, diameter, diameter):
            dx = ox - cutoff_arg
            dy = oy - cutoff_arg

            dist2 = dx * dx + dy * dy

            # Rounded Euclidean radius.
            # Equivalent to assigning the pixel to the nearest integer radius band.
            radius = ti.cast(ti.sqrt(ti.cast(dist2, ti.f32)) + 0.5, ti.i32)

            if radius <= cutoff_arg:
                x = sites_arg[s, 0] + dx
                y = sites_arg[s, 1] + dy

                if x >= 0 and x < grid_size_arg and y >= 0 and y < grid_size_arg:
                    frames_arg[s, y, x] = radius

    @ti.kernel
    def generate_result(
        frames_arg: ti.template(),
        sites_arg: ti.template(),
        labels_arg: ti.template(),
        radius_arg: ti.template(),
        unassigned_arg: ti.template(),
        num_sites_arg: ti.i32,
        inf_arg: ti.i32,
    ):
        """Generate final Voronoi labels.

        First priority:
            smaller rounded radius wins.

        Tie case:
            if two sites have the same rounded radius, compare exact squared
            Euclidean distance.

        Final tie:
            if exact squared distance is also same, choose smaller site index.
        """

        for y, x in labels_arg:
            best_site = -1
            best_radius = inf_arg
            best_dist2 = inf_arg

            for s in range(num_sites_arg):
                candidate_radius = frames_arg[s, y, x]

                if candidate_radius < inf_arg:
                    sx = sites_arg[s, 0]
                    sy = sites_arg[s, 1]

                    dx = x - sx
                    dy = y - sy
                    candidate_dist2 = dx * dx + dy * dy

                    should_update = False

                    if candidate_radius < best_radius:
                        should_update = True

                    elif candidate_radius == best_radius:
                        if candidate_dist2 < best_dist2:
                            should_update = True

                        elif candidate_dist2 == best_dist2:
                            # Tie-breaking rule: smaller site index wins.
                            if best_site == -1 or s < best_site:
                                should_update = True

                    if should_update:
                        best_radius = candidate_radius
                        best_dist2 = candidate_dist2
                        best_site = s

            labels_arg[y, x] = best_site

            if best_site >= 0:
                radius_arg[y, x] = best_radius
                unassigned_arg[y, x] = 0
            else:
                radius_arg[y, x] = inf_arg
                unassigned_arg[y, x] = 1

    # Fill frames
    fill_start = time.perf_counter()

    initialize_frames(frames, DEFAULT_INF)
    fill_radius_frames(frames, sites_field, cutoff_radius, grid_size)

    ti.sync()
    fill_time = time.perf_counter() - fill_start

    # Generate final result
    result_start = time.perf_counter()

    generate_result(
        frames,
        sites_field,
        labels_field,
        radius_field,
        unassigned_field,
        num_sites,
        DEFAULT_INF,
    )

    ti.sync()
    generate_time = time.perf_counter() - result_start

    # Transfer results back to NumPy
    labels = labels_field.to_numpy()
    radius_map = radius_field.to_numpy()
    unassigned_mask = unassigned_field.to_numpy().astype(bool)

    algorithm_only_time = fill_time + generate_time
    total_time = time.perf_counter() - total_start

    stats = {
        "algorithm": "proposed",
        "grid_size": grid_size,
        "num_sites": num_sites,
        "cutoff_radius": cutoff_radius,
        "arch": active_arch,
        "requested_arch": arch,
        "taichi_init_time": init_time,
        "fill_frames_time": fill_time,
        "generate_result_time": generate_time,
        "algorithm_only_time": algorithm_only_time,
        "total_time": total_time,
        "unassigned_pixels": int(unassigned_mask.sum()),
        "estimated_memory_mb": float(estimated_mb),
        "reduction_mode": "site_loop",
        "use_parallel_reduction_requested": bool(use_parallel_reduction),
        "circle_mode": "radius_band_fill",
        "tie_breaking": "exact_squared_distance_then_lower_site_index",
    }

    if save_debug_frames:
        debug_path = (
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "logs"
            / "debug_frames_last.npy"
        )
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(debug_path), frames.to_numpy())
        stats["debug_frames_path"] = str(debug_path)

    return labels, radius_map, unassigned_mask, stats