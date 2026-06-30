import time
import warnings
from pathlib import Path
import numpy as np
import taichi as ti

# Adjust these imports based on your exact project structure if needed
from config import DEFAULT_INF, MEMORY_WARNING_MB
from utils import _validate_inputs, estimate_memory_mb
from taichi_init import init_taichi

@ti.func
def include_sym_points(frames_arg: ti.template(), s: ti.i32, xc: ti.i32, yc: ti.i32, i: ti.i32, j: ti.i32, r: ti.i32, grid_size: ti.i32):
    """
    Plots the 8 symmetrical octant points for a given circle radius.
    """
    points = ti.Matrix([
        [xc + i, yc + j], [xc + i, yc - j], [xc - i, yc + j], [xc - i, yc - j],
        [xc + j, yc + i], [xc + j, yc - i], [xc - j, yc + i], [xc - j, yc - i]
    ])
    
    # ti.static unrolls this small loop for maximum GPU speed
    for k in ti.static(range(8)):
        px = points[k, 0]
        py = points[k, 1]
        if 0 <= px < grid_size and 0 <= py < grid_size:
            frames_arg[s, py, px] = r


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
    cutoff_arg: ti.i32,
    grid_size_arg: ti.i32,
    num_sites_arg: ti.i32
):
    """
    Implements MDCS (Modified Digital Circle using Square Numbers) - Algorithm 2.
    Uses pure integer math and a lookahead buffer to prevent absentee pixels.
    """
    for site_idx in range(num_sites_arg):
        xc = sites_arg[site_idx, 0]
        yc = sites_arg[site_idx, 1]
        
        # Paint the center site itself (radius 0)
        if 0 <= xc < grid_size_arg and 0 <= yc < grid_size_arg:
            frames_arg[site_idx, yc, xc] = 0
            
        # Grow the digital circle radius step by step up to the cutoff limit
        for r in range(1, cutoff_arg + 1):
            
            # Initializations exactly mirroring Dhar Algorithm 2
            i = 0
            j = r
            s = 0 
            w = r - 1
            l = w << 1  # Bitwise shift instead of multiply
            g = l
            
            while i <= j:
                # do-while loop translation from Algorithm 2
                while True:
                    include_sym_points(frames_arg, site_idx, xc, yc, i, j, r, grid_size_arg)
                    s += i
                    i += 1
                    s += i
                    if s > w:
                        break
                
                # The MDCS Lookahead Check: Captures the 'absentee pixels' (holes)
                if (s > w) and (s <= (w + g)) and (i <= j):
                    include_sym_points(frames_arg, site_idx, xc, yc, i, j, r, grid_size_arg)
                    
                w += l
                l -= 2
                j -= 1
                g += 2


@ti.kernel
def generate_result(
    frames_arg: ti.template(),
    labels_arg: ti.template(),
    radius_arg: ti.template(),
    unassigned_arg: ti.template(),
    num_sites_arg: ti.i32,
    inf_arg: ti.i32,
):
    """
    Resolves the Z-Buffer stack via Argmin Projection.
    Currently uses O(S) linear sweep. Next upgrade: O(log S) Parallel Reduction.
    """
    for y, x in labels_arg:
        best_site = -1
        best_radius = inf_arg

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


def run_proposed(
    grid_size,
    sites,
    cutoff_radius,
    arch="gpu",
    use_parallel_reduction=True,
    save_debug_frames=False,
):
    """Run the MDCS digital-circle-growing Voronoi baseline."""

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

    # 1. Initialize buffers to Infinity
    fill_start = time.perf_counter()
    initialize_frames(frames, DEFAULT_INF)
    
    # 2. Run the MDCS Algorithm
    fill_radius_frames_mdcs(frames, sites_field, cutoff_radius, grid_size, num_sites)
    ti.sync()
    fill_time = time.perf_counter() - fill_start

    # 3. Z-Buffer Argmin Projection
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