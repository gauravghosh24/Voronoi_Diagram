"""Digital circle growing Voronoi construction based on Dhar et al. (2024).

Implements:
1. Algorithm 2: MDCS (Modified Digital Circle using Square numbers)
   - Number-theoretic integer arithmetic with lookahead state to eliminate absentee pixels.
2. Algorithm 4: Incremental Radius Approach
   - Dynamically grows circles by step increments (default: 15) until no vacant pixels remain.
3. Radius-band baseline (comparative backend using rounded Euclidean distance).
"""

import math
import time
import warnings
from pathlib import Path

import numpy as np

from config import DEFAULT_INF, MEMORY_WARNING_MB
from .metrics import estimate_memory_mb
from .taichi_init import init_taichi


def generate_dcs_circle(xc, yc, r):
    """Generate points of a digital circle using DCS algorithm (Figure 3 in Dhar et al. 2024).

    Parameters
    ----------
    xc, yc : int
        Center coordinates.
    r : int
        Circle radius.

    Returns
    -------
    set of (int, int)
        Coordinates of all points on the digital circle.
    """
    points = set()
    if r == 0:
        return {(xc, yc)}

    i = 0
    j = r
    s_sq = 0
    w = r - 1
    l = w << 1

    while i <= j:
        while True:
            for dx, dy in [(i, j), (i, -j), (-i, j), (-i, -j),
                           (j, i), (j, -i), (-j, i), (-j, -i)]:
                points.add((xc + dx, yc + dy))
            s_sq += i
            i += 1
            s_sq += i
            if s_sq > w:
                break
        w += l
        l -= 2
        j -= 1

    return points


def generate_mdcs_circle(xc, yc, r):
    """Generate points of a digital circle using MDCS algorithm (Algorithm 2 / Figure 5).

    Parameters
    ----------
    xc, yc : int
        Center coordinates.
    r : int
        Circle radius.

    Returns
    -------
    set of (int, int)
        Coordinates of all points on the digital circle without absentee gaps.
    """
    points = set()
    if r == 0:
        return {(xc, yc)}

    if r == 1:
        # Include 8-neighborhood for r=1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx != 0 or dy != 0:
                    points.add((xc + dx, yc + dy))
        return points

    i = 0
    j = r
    s_sq = 0
    w = r - 1
    l = w << 1
    g = l

    while i <= j:
        while True:
            for dx, dy in [(i, j), (i, -j), (-i, j), (-i, -j),
                           (j, i), (j, -i), (-j, i), (-j, -i)]:
                points.add((xc + dx, yc + dy))
            s_sq += i
            i += 1
            s_sq += i
            if s_sq > w:
                break

        # MDCS Lookahead Check (captures absentee pixels)
        if s_sq > w and s_sq <= (w + g) and i <= j:
            for dx, dy in [(i, j), (i, -j), (-i, j), (-i, -j),
                           (j, i), (j, -i), (-j, i), (-j, -i)]:
                points.add((xc + dx, yc + dy))

        w += l
        l -= 2
        j -= 1
        g += 2

    return points


def generate_dcs_disc(xc, yc, r_max):
    """Generate a digital disc by concentric DCS circles up to r_max."""
    grid_size = 2 * r_max + 5
    disc = np.zeros((grid_size, grid_size), dtype=np.int32)
    disc[yc, xc] = 1
    for r in range(1, r_max + 1):
        for px, py in generate_dcs_circle(xc, yc, r):
            if 0 <= px < grid_size and 0 <= py < grid_size:
                disc[py, px] = r
    return disc


def generate_mdcs_disc(xc, yc, r_max):
    """Generate a digital disc by concentric MDCS circles up to r_max."""
    grid_size = 2 * r_max + 5
    disc = np.zeros((grid_size, grid_size), dtype=np.int32)
    disc[yc, xc] = 1
    for r in range(1, r_max + 1):
        for px, py in generate_mdcs_circle(xc, yc, r):
            if 0 <= px < grid_size and 0 <= py < grid_size:
                disc[py, px] = r
    return disc


def find_absentee_pixels(disc, xc, yc, r_max):
    """Find pixels inside Euclidean circle of radius r_max that are not covered in disc."""
    holes = []
    r_bound2 = (r_max - 0.5) ** 2
    for y in range(disc.shape[0]):
        for x in range(disc.shape[1]):
            d2 = (x - xc) ** 2 + (y - yc) ** 2
            if d2 <= r_bound2 and disc[y, x] == 0:
                holes.append((x - xc, y - yc))
    return holes


def _validate_inputs(grid_size, sites, cutoff_radius=0):
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
    circle_backend="mdcs",
    arch="gpu",
    use_parallel_reduction=True,
    save_debug_frames=False,
):
    """Run the proposed digital-circle-growing Voronoi algorithm.

    Parameters
    ----------
    grid_size : int
        Width and height of the square grid.
    sites : np.ndarray
        Array of shape (S, 2). Each site is represented as (x, y).
    cutoff_radius : int
        Maximum radius up to which each site grows.
    circle_backend : str
        "mdcs" for true integer Algorithm 2 MDCS generator,
        or "radius_band" for float sqrt baseline.
    arch : str
        "gpu" or "cpu".
    use_parallel_reduction : bool
        Kept for API compatibility.
    save_debug_frames : bool
        If True, saves frames array to outputs/logs.

    Returns
    -------
    labels : np.ndarray
        Shape (grid_size, grid_size). labels[y, x] = nearest site index.
    radius_map : np.ndarray
        Shape (grid_size, grid_size). Minimum radius value.
    unassigned_mask : np.ndarray
        Boolean mask where True means unassigned pixel.
    stats : dict
        Runtime and execution statistics.
    """
    grid_size, sites, cutoff_radius = _validate_inputs(
        grid_size, sites, cutoff_radius
    )
    num_sites = int(sites.shape[0])

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
            "circle_backend": circle_backend,
            "circle_mode": circle_backend,
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

    sites_field = ti.field(dtype=ti.i32, shape=(num_sites, 2))
    frames = ti.field(dtype=ti.i32, shape=(num_sites, grid_size, grid_size))
    labels_field = ti.field(dtype=ti.i32, shape=(grid_size, grid_size))
    radius_field = ti.field(dtype=ti.i32, shape=(grid_size, grid_size))
    unassigned_field = ti.field(dtype=ti.i32, shape=(grid_size, grid_size))

    sites_field.from_numpy(sites.astype(np.int32, copy=False))

    diameter = 2 * cutoff_radius + 1

    @ti.func
    def include_sym_points(
        frames_arg: ti.template(),
        site_idx: ti.i32,
        xc: ti.i32,
        yc: ti.i32,
        i: ti.i32,
        j: ti.i32,
        r: ti.i32,
        grid_size_arg: ti.i32,
    ):
        points = ti.Matrix([
            [xc + i, yc + j], [xc + i, yc - j], [xc - i, yc + j], [xc - i, yc - j],
            [xc + j, yc + i], [xc + j, yc - i], [xc - j, yc + i], [xc - j, yc - i]
        ])
        for k in ti.static(range(8)):
            px = points[k, 0]
            py = points[k, 1]
            if 0 <= px < grid_size_arg and 0 <= py < grid_size_arg:
                ti.atomic_min(frames_arg[site_idx, py, px], r)

    @ti.func
    def include_diag_points(
        frames_arg: ti.template(),
        site_idx: ti.i32,
        xc: ti.i32,
        yc: ti.i32,
        r: ti.i32,
        grid_size_arg: ti.i32,
    ):
        diag_points = ti.Matrix([
            [xc + 1, yc + 1], [xc + 1, yc - 1], [xc - 1, yc + 1], [xc - 1, yc - 1]
        ])
        for k in ti.static(range(4)):
            px = diag_points[k, 0]
            py = diag_points[k, 1]
            if 0 <= px < grid_size_arg and 0 <= py < grid_size_arg:
                ti.atomic_min(frames_arg[site_idx, py, px], r)

    @ti.kernel
    def initialize_frames(
        frames_arg: ti.template(),
        inf_arg: ti.i32,
    ):
        for s, y, x in frames_arg:
            frames_arg[s, y, x] = inf_arg

    @ti.kernel
    def fill_radius_frames_mdcs(
        frames_arg: ti.template(),
        sites_arg: ti.template(),
        start_radius_arg: ti.i32,
        end_radius_arg: ti.i32,
        grid_size_arg: ti.i32,
        num_sites_arg: ti.i32,
    ):
        """Implements Algorithm 2 MDCS with lookahead and integer arithmetic."""
        for s in range(num_sites_arg):
            xc = sites_arg[s, 0]
            yc = sites_arg[s, 1]

            if start_radius_arg == 0:
                if 0 <= xc < grid_size_arg and 0 <= yc < grid_size_arg:
                    ti.atomic_min(frames_arg[s, yc, xc], 0)

            actual_start = ti.max(1, start_radius_arg)
            for r in range(actual_start, end_radius_arg + 1):
                if r == 1:
                    include_diag_points(frames_arg, s, xc, yc, 1, grid_size_arg)

                # MDCS Algorithm 2
                i = 0
                j = r
                s_sq = 0
                w = r - 1
                l = w << 1
                g = l

                while i <= j:
                    while True:
                        include_sym_points(frames_arg, s, xc, yc, i, j, r, grid_size_arg)
                        s_sq += i
                        i += 1
                        s_sq += i
                        if s_sq > w:
                            break

                    # MDCS Lookahead Check
                    if s_sq > w and s_sq <= (w + g) and i <= j:
                        include_sym_points(frames_arg, s, xc, yc, i, j, r, grid_size_arg)

                    w += l
                    l -= 2
                    j -= 1
                    g += 2

    @ti.kernel
    def fill_radius_frames_band(
        frames_arg: ti.template(),
        sites_arg: ti.template(),
        cutoff_arg: ti.i32,
        grid_size_arg: ti.i32,
    ):
        for s, oy, ox in ti.ndrange(num_sites, diameter, diameter):
            dx = ox - cutoff_arg
            dy = oy - cutoff_arg
            dist2 = dx * dx + dy * dy
            radius = ti.cast(ti.sqrt(ti.cast(dist2, ti.f32)) + 0.5, ti.i32)
            if radius <= cutoff_arg:
                x = sites_arg[s, 0] + dx
                y = sites_arg[s, 1] + dy
                if 0 <= x < grid_size_arg and 0 <= y < grid_size_arg:
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

    if circle_backend == "mdcs":
        fill_radius_frames_mdcs(frames, sites_field, 0, cutoff_radius, grid_size, num_sites)
    else:
        fill_radius_frames_band(frames, sites_field, cutoff_radius, grid_size)

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
        "circle_backend": circle_backend,
        "circle_mode": circle_backend,
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


def run_proposed_incremental(
    grid_size,
    sites,
    initial_radius=15,
    step_radius=15,
    max_radius=None,
    circle_backend="mdcs",
    arch="gpu",
    use_parallel_reduction=True,
    save_debug_frames=False,
):
    """Run Algorithm 4 (Incremental Radius Approach) from Dhar et al. (2024).

    Grows concentric circles incrementally by step_radius until there are
    no vacant (unassigned) pixels remaining.

    Parameters
    ----------
    grid_size : int
        Width and height of the square grid.
    sites : np.ndarray
        Array of shape (S, 2).
    initial_radius : int
        Starting cutoff radius (default: 15).
    step_radius : int
        Radius increment per iteration (default: 15 as in Algorithm 4).
    max_radius : int, optional
        Maximum allowable radius limit. If None, uses grid diagonal.
    circle_backend : str
        "mdcs" (default) or "radius_band".
    arch : str
        "gpu" or "cpu".
    use_parallel_reduction : bool
        Kept for API compatibility.
    save_debug_frames : bool
        If True, saves frames array.

    Returns
    -------
    labels : np.ndarray
    radius_map : np.ndarray
    unassigned_mask : np.ndarray
    stats : dict
        Contains iterations count, final cutoff reached, and vacant pixel history.
    """
    grid_size = int(grid_size)
    initial_radius = int(initial_radius)
    step_radius = int(step_radius)

    if max_radius is None:
        max_radius = int(math.ceil(math.sqrt(2.0) * (grid_size - 1)))
    else:
        max_radius = int(max_radius)

    grid_size, sites, _ = _validate_inputs(grid_size, sites, 0)
    num_sites = int(sites.shape[0])

    if num_sites == 0:
        labels = np.full((grid_size, grid_size), -1, dtype=np.int32)
        radius_map = np.full((grid_size, grid_size), DEFAULT_INF, dtype=np.int32)
        unassigned_mask = np.ones((grid_size, grid_size), dtype=bool)

        stats = {
            "algorithm": "proposed_incremental",
            "grid_size": grid_size,
            "num_sites": 0,
            "cutoff_radius": 0,
            "final_radius": 0,
            "iterations": 0,
            "history": [],
            "arch": arch,
            "requested_arch": arch,
            "total_time": 0.0,
            "unassigned_pixels": int(unassigned_mask.sum()),
            "circle_backend": circle_backend,
        }
        return labels, radius_map, unassigned_mask, stats

    total_start = time.perf_counter()
    estimated_mb = estimate_memory_mb(num_sites, grid_size)

    init_start = time.perf_counter()
    ti, active_arch = init_taichi(arch)
    init_time = time.perf_counter() - init_start

    sites_field = ti.field(dtype=ti.i32, shape=(num_sites, 2))
    frames = ti.field(dtype=ti.i32, shape=(num_sites, grid_size, grid_size))
    labels_field = ti.field(dtype=ti.i32, shape=(grid_size, grid_size))
    radius_field = ti.field(dtype=ti.i32, shape=(grid_size, grid_size))
    unassigned_field = ti.field(dtype=ti.i32, shape=(grid_size, grid_size))

    sites_field.from_numpy(sites.astype(np.int32, copy=False))

    @ti.func
    def include_sym_points(
        frames_arg: ti.template(),
        site_idx: ti.i32,
        xc: ti.i32,
        yc: ti.i32,
        i: ti.i32,
        j: ti.i32,
        r: ti.i32,
        grid_size_arg: ti.i32,
    ):
        points = ti.Matrix([
            [xc + i, yc + j], [xc + i, yc - j], [xc - i, yc + j], [xc - i, yc - j],
            [xc + j, yc + i], [xc + j, yc - i], [xc - j, yc + i], [xc - j, yc - i]
        ])
        for k in ti.static(range(8)):
            px = points[k, 0]
            py = points[k, 1]
            if 0 <= px < grid_size_arg and 0 <= py < grid_size_arg:
                ti.atomic_min(frames_arg[site_idx, py, px], r)

    @ti.func
    def include_diag_points(
        frames_arg: ti.template(),
        site_idx: ti.i32,
        xc: ti.i32,
        yc: ti.i32,
        r: ti.i32,
        grid_size_arg: ti.i32,
    ):
        diag_points = ti.Matrix([
            [xc + 1, yc + 1], [xc + 1, yc - 1], [xc - 1, yc + 1], [xc - 1, yc - 1]
        ])
        for k in ti.static(range(4)):
            px = diag_points[k, 0]
            py = diag_points[k, 1]
            if 0 <= px < grid_size_arg and 0 <= py < grid_size_arg:
                ti.atomic_min(frames_arg[site_idx, py, px], r)

    @ti.kernel
    def initialize_frames(
        frames_arg: ti.template(),
        inf_arg: ti.i32,
    ):
        for s, y, x in frames_arg:
            frames_arg[s, y, x] = inf_arg

    @ti.kernel
    def fill_radius_frames_mdcs(
        frames_arg: ti.template(),
        sites_arg: ti.template(),
        start_radius_arg: ti.i32,
        end_radius_arg: ti.i32,
        grid_size_arg: ti.i32,
        num_sites_arg: ti.i32,
    ):
        for s in range(num_sites_arg):
            xc = sites_arg[s, 0]
            yc = sites_arg[s, 1]

            if start_radius_arg == 0:
                if 0 <= xc < grid_size_arg and 0 <= yc < grid_size_arg:
                    ti.atomic_min(frames_arg[s, yc, xc], 0)

            actual_start = ti.max(1, start_radius_arg)
            for r in range(actual_start, end_radius_arg + 1):
                if r == 1:
                    include_diag_points(frames_arg, s, xc, yc, 1, grid_size_arg)

                i = 0
                j = r
                s_sq = 0
                w = r - 1
                l = w << 1
                g = l

                while i <= j:
                    while True:
                        include_sym_points(frames_arg, s, xc, yc, i, j, r, grid_size_arg)
                        s_sq += i
                        i += 1
                        s_sq += i
                        if s_sq > w:
                            break

                    if s_sq > w and s_sq <= (w + g) and i <= j:
                        include_sym_points(frames_arg, s, xc, yc, i, j, r, grid_size_arg)

                    w += l
                    l -= 2
                    j -= 1
                    g += 2

    @ti.kernel
    def fill_radius_frames_band(
        frames_arg: ti.template(),
        sites_arg: ti.template(),
        cutoff_arg: ti.i32,
        grid_size_arg: ti.i32,
    ):
        diameter_val = 2 * cutoff_arg + 1
        for s, oy, ox in ti.ndrange(num_sites, diameter_val, diameter_val):
            dx = ox - cutoff_arg
            dy = oy - cutoff_arg
            dist2 = dx * dx + dy * dy
            radius = ti.cast(ti.sqrt(ti.cast(dist2, ti.f32)) + 0.5, ti.i32)
            if radius <= cutoff_arg:
                x = sites_arg[s, 0] + dx
                y = sites_arg[s, 1] + dy
                if 0 <= x < grid_size_arg and 0 <= y < grid_size_arg:
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

    @ti.kernel
    def count_unassigned(unassigned_arg: ti.template()) -> ti.i32:
        total = 0
        for y, x in unassigned_arg:
            total += unassigned_arg[y, x]
        return total

    # Initialize buffer to infinity
    initialize_frames(frames, DEFAULT_INF)

    current_radius = initial_radius
    prev_radius = 0
    iteration = 0
    history = []
    total_fill_time = 0.0
    total_gen_time = 0.0

    while True:
        iteration += 1
        curr_r = min(max_radius, current_radius)

        fill_t0 = time.perf_counter()
        if circle_backend == "mdcs":
            start_r = 0 if iteration == 1 else (prev_radius + 1)
            fill_radius_frames_mdcs(frames, sites_field, start_r, curr_r, grid_size, num_sites)
        else:
            fill_radius_frames_band(frames, sites_field, curr_r, grid_size)
        ti.sync()
        total_fill_time += time.perf_counter() - fill_t0

        gen_t0 = time.perf_counter()
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
        total_gen_time += time.perf_counter() - gen_t0

        vacant_pixels = int(count_unassigned(unassigned_field))
        history.append({
            "iteration": iteration,
            "radius": curr_r,
            "vacant_pixels": vacant_pixels,
        })

        if vacant_pixels == 0 or curr_r >= max_radius:
            break

        prev_radius = curr_r
        current_radius = curr_r + step_radius

    labels = labels_field.to_numpy()
    radius_map = radius_field.to_numpy()
    unassigned_mask = unassigned_field.to_numpy().astype(bool)

    total_time = time.perf_counter() - total_start

    stats = {
        "algorithm": "proposed_incremental",
        "grid_size": grid_size,
        "num_sites": num_sites,
        "cutoff_radius": curr_r,
        "final_radius": curr_r,
        "initial_radius": initial_radius,
        "step_radius": step_radius,
        "iterations": iteration,
        "history": history,
        "arch": active_arch,
        "requested_arch": arch,
        "taichi_init_time": init_time,
        "fill_frames_time": total_fill_time,
        "generate_result_time": total_gen_time,
        "algorithm_only_time": total_fill_time + total_gen_time,
        "total_time": total_time,
        "unassigned_pixels": int(unassigned_mask.sum()),
        "estimated_memory_mb": float(estimated_mb),
        "circle_backend": circle_backend,
        "circle_mode": circle_backend,
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