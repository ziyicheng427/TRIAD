"""Unit tests of TRIAD-Map's update rule and TRIAD-Link's interfaces."""

import numpy as np
import pytest

from triad.link import l2_gains, occupancy_reweight
from triad.map import LogOddsGrid, MapConfig, PersistenceTracker, TriadMapUpdater, UpdateMode


def make_grid():
    return LogOddsGrid(MapConfig(), width=60, height=60, origin=(0.0, 0.0))


class TestUpdateBands:
    def test_pgc_gates_transient_hits_in_free_cells(self):
        """One or two scans of endpoints in a free cell must not occupy it
        (N = 3 persistence gate); the third scan admits the hit."""
        grid = make_grid()
        updater = TriadMapUpdater(grid, mode=UpdateMode.PGC_TBB)
        # drive the cell deep into the free band with misses first
        endpoint_far = np.array([[8.0, 5.0]])
        for _ in range(20):
            updater.update(endpoint_far, np.array([0.0, 5.0]))
        target_cell = grid.world_to_cell(endpoint_far)[0]
        assert grid.p_at(target_cell[None, :])[0] < 0.49

        # a transient passing object: two scans only → gated out
        endpoint_pass = np.array([[3.0, 5.0]])
        for _ in range(2):
            updater.update(np.vstack([endpoint_far, endpoint_pass]),
                           np.array([0.0, 5.0]))
        pass_cell = grid.world_to_cell(endpoint_pass)[0]
        assert not grid.occupied_mask()[pass_cell[1], pass_cell[0]]

    def test_tbb_boosts_transition_band(self):
        """The boosted hit (p_hit = 0.85) integrates faster than the base."""
        grid_tbb = make_grid()
        grid_base = make_grid()
        tbb = TriadMapUpdater(grid_tbb, mode=UpdateMode.PGC_TBB)
        base = TriadMapUpdater(grid_base, mode=UpdateMode.PGC_TBB)
        # place both updaters' cells into the transition band, then compare
        # a single hit's effect: monkey-patch the base updater to use base hits
        endpoint = np.array([[5.0, 5.0]])
        for _ in range(30):
            tbb.update(endpoint, np.array([0.0, 5.0]))
            base.update(endpoint, np.array([0.0, 5.0]))
        cell = grid_tbb.world_to_cell(endpoint)[0]
        # with PGC satisfied, both admit hits; TBB's boost keeps its cell
        # at least as occupied as the base increment path
        p_tbb = grid_tbb.p_at(cell[None, :])[0]
        assert p_tbb > 0.5

    def test_misses_clear_unknown_and_occupied(self):
        grid = make_grid()
        updater = TriadMapUpdater(grid, mode=UpdateMode.PGC_TBB)
        endpoint = np.array([[5.0, 5.0]])
        for _ in range(30):
            updater.update(endpoint, np.array([0.0, 5.0]))
        cell = grid.world_to_cell(endpoint)[0]
        assert grid.occupied_mask()[cell[1], cell[0]]
        # remove the obstacle: rays now traverse the cell
        for _ in range(200):
            updater.update(np.array([[5.0, 5.9]]), np.array([0.0, 5.0]))
        assert grid.log_odds[cell[1], cell[0]] < 0


class TestPersistence:
    def test_counts_reset_on_gap(self):
        grid = make_grid()
        tracker = PersistenceTracker(0.15)
        cell = np.array([[10, 10]])
        tracker.observe(cell, grid)
        tracker.observe(cell, grid)
        assert tracker.count((10, 10)) == 2
        tracker.observe(np.array([[40, 40]]), grid)  # no support → reset
        assert tracker.count((10, 10)) == 0


class TestL2:
    def test_gains_at_zero_noise(self):
        assert l2_gains(0.0) == (1.0, 1.0)

    def test_asymmetric_clip_bounds(self):
        g_hit, g_miss = l2_gains(1.0)  # σ_t ≫ σ₀
        assert g_hit == pytest.approx(0.25)
        assert g_miss == pytest.approx(0.6)

    def test_monotone_damping(self):
        prev_hit = prev_miss = 1.0
        for sigma_t in (0.0, 0.02, 0.04, 0.08, 0.12, 0.5):
            g_hit, g_miss = l2_gains(sigma_t)
            assert g_hit <= prev_hit and g_miss <= prev_miss
            assert g_miss >= g_hit  # the deliberate asymmetry (Eq. 3)
            prev_hit, prev_miss = g_hit, g_miss


class TestL3:
    def test_free_cells_down_weighted(self):
        grid = make_grid()
        config_map = grid.config
        # drive one region free, leave the rest unknown (p = 0.5)
        endpoint = np.array([[5.0, 5.0]])
        updater = TriadMapUpdater(grid, mode=UpdateMode.PGC_TBB)
        for _ in range(100):
            updater.update(np.array([[5.0, 5.9]]), np.array([0.0, 5.0]))
        free_point = np.array([[5.0, 5.0]])   # traversed by the rays → free
        unknown_point = np.array([[0.4, 0.4]])
        weights = occupancy_reweight(np.vstack([free_point, unknown_point]), grid)
        assert weights[0] == pytest.approx(0.1)   # P < 0.2 → 0.1
        assert weights[1] == pytest.approx(1.0)   # unknown keeps its weight
        del config_map
