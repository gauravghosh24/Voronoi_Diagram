"""Unit tests for Algorithm 2 (MDCS) and Algorithm 4 (Incremental Radius Approach).

Validates:
1. Figure 4B: Zero absentee pixels (holes) in the digital disc of radius r=20 using MDCS.
2. Figure 4A vs 4B: Classical DCS generates 128 absentee pixels at r=20, whereas MDCS eliminates them.
3. Multiple radii (5, 10, 15, 20, 25, 30) have 0 absentee pixels with MDCS.
4. Proposed Voronoi algorithm runs properly with circle_backend="mdcs".
5. Incremental radius approach iteratively reduces vacant pixels to 0.
"""

import numpy as np
import pytest

from src.proposed_circle_growing import (
    find_absentee_pixels,
    generate_dcs_circle,
    generate_dcs_disc,
    generate_mdcs_circle,
    generate_mdcs_disc,
    run_proposed,
    run_proposed_incremental,
)


def test_mdcs_no_absentee_pixels_r20():
    """Verify Figure 4B: MDCS produces zero absentee pixels for radius r=20."""
    r_max = 20
    xc, yc = 25, 25
    disc = generate_mdcs_disc(xc, yc, r_max)

    holes = find_absentee_pixels(disc, xc, yc, r_max)
    assert len(holes) == 0, f"Expected 0 absentee pixels for MDCS at r=20, got {len(holes)}: {holes}"


def test_dcs_vs_mdcs_absentee_reduction():
    """Verify Figure 4A vs 4B: DCS has holes while MDCS eliminates all holes at r=20."""
    r_max = 20
    xc, yc = 25, 25
    dcs_disc = generate_dcs_disc(xc, yc, r_max)
    mdcs_disc = generate_mdcs_disc(xc, yc, r_max)

    dcs_holes = find_absentee_pixels(dcs_disc, xc, yc, r_max)
    mdcs_holes = find_absentee_pixels(mdcs_disc, xc, yc, r_max)

    # DCS suffers from 128 absentee pixels inside the r=20 disc (Figure 4A)
    assert len(dcs_holes) == 128, f"Expected DCS to have 128 holes at r=20, got {len(dcs_holes)}"
    # MDCS completely eliminates all absentee pixels (Figure 4B)
    assert len(mdcs_holes) == 0, f"Expected MDCS to have 0 holes at r=20, got {len(mdcs_holes)}"


@pytest.mark.parametrize("r_max", [5, 10, 15, 20, 25, 30])
def test_mdcs_various_radii_zero_holes(r_max):
    """Verify that MDCS disc has 0 absentee pixels across various radii."""
    xc = r_max + 3
    yc = r_max + 3
    disc = generate_mdcs_disc(xc, yc, r_max)
    holes = find_absentee_pixels(disc, xc, yc, r_max)
    assert len(holes) == 0, f"MDCS at r={r_max} produced {len(holes)} holes"


def test_mdcs_circle_symmetry():
    """Verify 8-fold symmetry of generated MDCS circle points."""
    xc, yc = 30, 30
    points = generate_mdcs_circle(xc, yc, 12)
    assert len(points) > 0

    for x, y in points:
        dx = x - xc
        dy = y - yc
        # All 8 reflections/rotations should be in the set
        for sx, sy in [(dx, dy), (dx, -dy), (-dx, dy), (-dx, -dy),
                       (dy, dx), (dy, -dx), (-dy, dx), (-dy, -dx)]:
            assert (xc + sx, yc + sy) in points


def test_run_proposed_mdcs_execution():
    """Verify run_proposed works correctly with MDCS backend."""
    grid_size = 64
    sites = np.array([[12, 12], [48, 48], [15, 45]], dtype=np.int32)
    cutoff_radius = 35

    labels, radius_map, unassigned_mask, stats = run_proposed(
        grid_size=grid_size,
        sites=sites,
        cutoff_radius=cutoff_radius,
        circle_backend="mdcs",
        arch="cpu",
    )

    assert labels.shape == (grid_size, grid_size)
    assert radius_map.shape == (grid_size, grid_size)
    assert unassigned_mask.shape == (grid_size, grid_size)
    assert stats["circle_backend"] == "mdcs"
    # Sites should be assigned to themselves with radius 0
    for s_idx, (sx, sy) in enumerate(sites):
        assert labels[sy, sx] == s_idx
        assert radius_map[sy, sx] == 0


def test_incremental_radius_eliminates_unassigned():
    """Verify Algorithm 4 (Incremental Radius): expands radius until 0 unassigned pixels remain."""
    grid_size = 64
    sites = np.array([[10, 10], [50, 50], [50, 10], [10, 50]], dtype=np.int32)

    labels, radius_map, unassigned_mask, stats = run_proposed_incremental(
        grid_size=grid_size,
        sites=sites,
        initial_radius=10,
        step_radius=10,
        circle_backend="mdcs",
        arch="cpu",
    )

    assert stats["unassigned_pixels"] == 0
    assert not unassigned_mask.any()
    assert (labels >= 0).all()
    assert stats["iterations"] >= 1
    assert stats["final_radius"] >= 10

    # Vacant pixels count must be strictly non-increasing across iterations
    vacant_history = [h["vacant_pixels"] for h in stats["history"]]
    for i in range(len(vacant_history) - 1):
        assert vacant_history[i] >= vacant_history[i + 1]
    assert vacant_history[-1] == 0


def test_circle_backend_comparison():
    """Compare MDCS vs radius_band backend results."""
    grid_size = 64
    sites = np.array([[20, 20], [44, 44]], dtype=np.int32)
    cutoff = 40

    labels_mdcs, _, _, _ = run_proposed(
        grid_size, sites, cutoff, circle_backend="mdcs", arch="cpu"
    )
    labels_band, _, _, _ = run_proposed(
        grid_size, sites, cutoff, circle_backend="radius_band", arch="cpu"
    )

    # Both backends should have high agreement (> 98%)
    assigned = (labels_mdcs >= 0) & (labels_band >= 0)
    match = (labels_mdcs[assigned] == labels_band[assigned]).sum()
    agreement = match / assigned.sum()
    assert agreement > 0.98, f"Expected > 98% agreement between backends, got {agreement:.4f}"
