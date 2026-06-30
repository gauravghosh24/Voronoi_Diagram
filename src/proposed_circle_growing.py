"""Paper-inspired digital circle growing Voronoi construction."""

import time
import warnings
from pathlib import Path

import numpy as np

from config import DEFAULT_INF, MEMORY_WARNING_MB
from .metrics import estimate_memory_mb
from .taichi_init import init_taichi


def _validate_inputs(grid_size, sites, cutoff_radius):
    grid_size = int(grid_size)
    cutoff_radius = int(cutoff_radius)
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if cutoff_radius < 0:
        raise ValueError("cutoff_radius must be non-negative")

    sites = np.asarray(sites, dtype=np.int32)
    if sites.ndim != 2 or sites.shape[1] != 2:
        raise ValueError("sites must have shape (S, 2)")
    return grid_size, sites, cutoff_radius


def run_proposed(
    grid_size,
    sites,
    cutoff_radius,
    arch="gpu",
    use_parallel_reduction=True,
    save_debug_frames=False,
):
    """Run the digital-circle-growing Voronoi baseline.

    The implementation keeps the requested ``frames[S, H, W]`` structure. For
    correctness and hole prevention, each site/pixel pair is assigned to the
    nearest integer radius band when the pixel is inside the cutoff.
    """

    grid_size, sites, cutoff_radius = _validate_inputs(grid_size, sites, cutoff_radius)
    num_sites = int(sites.shape[0])

    if num_sites == 0:
        labels = np.full((grid_size, grid_size), -1, dtype=np.int32)
        radius_map = np.full((grid_size, grid_size), DEFAULT_INF, dtype=np.int32)
        unassigned_mask = np.ones((grid_size, grid_size), dtype=bool)
        return labels, radius_map, unassigned_mask, {
            "algorithm": "proposed",
            "grid_size": grid_size,
            "num_sites": 0,
            "cutoff_radius": cutoff_radius,
            "arch": arch,
            "fill_frames_time": 0.0,
            "generate_result_time": 0.0,
            "total_time": 0.0,
            "unassigned_pixels": int(unassigned_mask.sum()),
            "estimated_memory_mb": 0.0,
            "reduction_mode": "none",
        }

    estimated_mb = estimate_memory_mb(num_sites, grid_size)
    if estimated_mb > MEMORY_WARNING_MB:
        message = (
            "Estimated proposed frames memory is {:.1f} MB for S={}, grid={}. "
            "This may exceed GPU memory."
        ).format(estimated_mb, num_sites, grid_size)
        warnings.warn(message)
        print("WARNING: {}".format(message))

    total_start = time.perf_counter()
    ti, active_arch = init_taichi(arch)

    sites_field = ti.field(dtype=ti.i32, shape=(num_sites, 2))
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
        for s, y, x in frames_arg:
            frames_arg[s, y, x] = inf_arg

    @ti.kernel
    def fill_radius_frames(
        frames_arg: ti.template(),
        sites_arg: ti.template(),
        cutoff_arg: ti.i32,
        grid_size_arg: ti.i32,
    ):
        for s, oy, ox in ti.ndrange(num_sites, diameter, diameter):
            dx = ox - cutoff_arg
            dy = oy - cutoff_arg
            dist2 = dx * dx + dy * dy

            # This is the radius-band fill:
            # (r - 0.5)^2 <= dist2 < (r + 0.5)^2.
            radius = ti.cast(ti.sqrt(ti.cast(dist2, ti.f32)) + 0.5, ti.i32)
            if radius <= cutoff_arg:
                x = sites_arg[s, 0] + dx
                y = sites_arg[s, 1] + dy
                if x >= 0 and x < grid_size_arg and y >= 0 and y < grid_size_arg:
                    frames_arg[s, y, x] = radius

    @ti.kernel
    def generate_result(
        frames_arg: ti.template(),
        labels_arg: ti.template(),
        radius_arg: ti.template(),
        unassigned_arg: ti.template(),
        num_sites_arg: ti.i32,
        inf_arg: ti.i32,
    ):
        for y, x in labels_arg:
            best_site = -1
            best_radius = inf_arg

            # Simple site loop. The parameter is kept in the public API so this
            # can later be replaced with a tree/parallel reduction kernel.
            for s in range(num_sites_arg):
                candidate_radius = frames_arg[s, y, x]
                if candidate_radius < best_radius:
                    best_radius = candidate_radius
                    best_site = s

            labels_arg[y, x] = best_site
            if best_site >= 0:
                radius_arg[y, x] = best_radius
                unassigned_arg[y, x] = 0
            else:
                radius_arg[y, x] = inf_arg
                unassigned_arg[y, x] = 1

    fill_start = time.perf_counter()
    initialize_frames(frames, DEFAULT_INF)
    fill_radius_frames(frames, sites_field, cutoff_radius, grid_size)
    ti.sync()
    fill_time = time.perf_counter() - fill_start

    result_start = time.perf_counter()
    generate_result(frames, labels_field, radius_field, unassigned_field, num_sites, DEFAULT_INF)
    ti.sync()
    generate_time = time.perf_counter() - result_start

    labels = labels_field.to_numpy()
    radius_map = radius_field.to_numpy()
    unassigned_mask = unassigned_field.to_numpy().astype(bool)

    total_time = time.perf_counter() - total_start
    stats = {
        "algorithm": "proposed",
        "grid_size": grid_size,
        "num_sites": num_sites,
        "cutoff_radius": cutoff_radius,
        "arch": active_arch,
        "requested_arch": arch,
        "fill_frames_time": fill_time,
        "generate_result_time": generate_time,
        "total_time": total_time,
        "unassigned_pixels": int(unassigned_mask.sum()),
        "estimated_memory_mb": float(estimated_mb),
        "reduction_mode": "site_loop",
        "use_parallel_reduction_requested": bool(use_parallel_reduction),
    }

    if save_debug_frames:
        debug_path = Path(__file__).resolve().parents[1] / "outputs" / "logs" / "debug_frames_last.npy"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(debug_path), frames.to_numpy())
        stats["debug_frames_path"] = str(debug_path)

    return labels, radius_map, unassigned_mask, stats
