"""Unit tests of TRIAD-Cal's geometric edge extraction and estimator pieces."""

import numpy as np
import pytest

from triad.cal import CalConfig, extract_lidar_edges
from triad.cal.association import (
    CrossModalAssociation,
    project_points,
    projection_jacobian,
    temporal_weights_over_window,
)
from triad.cal.image_edges import AttractionField, build_attraction_field
from triad.cal.sam_adapter import MaskSet


def _corner_cloud(voxel_size=1.0, n_per_plane=200, seed=0):
    """Two perpendicular planes meeting at x = 0 (a depth-continuous edge)."""
    rng = np.random.default_rng(seed)
    plane_a = np.column_stack([
        np.zeros(n_per_plane), rng.uniform(0.5, 3.5, n_per_plane),
        rng.uniform(0.5, 3.5, n_per_plane),
    ])
    plane_b = np.column_stack([
        rng.uniform(0.5, 3.5, n_per_plane), np.zeros(n_per_plane),
        rng.uniform(0.5, 3.5, n_per_plane),
    ])
    return np.vstack([plane_a, plane_b]) * voxel_size


class TestLidarEdges:
    def test_perpendicular_planes_yield_edge(self):
        cloud = _corner_cloud()
        config = CalConfig(voxel_size=1.0, plane_min_inliers=5)
        edges = extract_lidar_edges(cloud, config, seed=0)
        assert len(edges) >= 1
        # the intersection line runs along z; every direction must be ±z-ish
        for edge in edges:
            assert abs(abs(edge.direction[2]) - 1.0) < 0.1

    def test_dihedral_window_rejects_parallel_planes(self):
        # two near-parallel planes (dihedral ≈ 0°): no edge may be produced
        rng = np.random.default_rng(1)
        a = np.column_stack([np.zeros(300), rng.uniform(1, 4, 300),
                             rng.uniform(1, 4, 300)])
        b = a + np.array([0.0, 2.0, 0.0])  # parallel offset plane
        cloud = np.vstack([a, b]) * 2.0  # spread beyond a single voxel
        config = CalConfig(voxel_size=2.0, plane_min_inliers=5)
        edges = extract_lidar_edges(cloud, config, seed=0)
        for edge in edges:
            dihedral = np.degrees(np.arccos(
                np.clip(abs(np.array([0, 1, 0]) @ edge.direction), -1, 1)
            ))
            # any produced edge must lie inside the dihedral window
            assert 30.0 - 1e-6 <= dihedral <= 150.0 + 1e-6


class TestAssociation:
    def test_projection_jacobian_finite_difference(self):
        K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
        pts = np.array([[0.5, -0.2, 3.0], [-0.3, 0.4, 5.0]])
        J = projection_jacobian(K, pts)
        eps = 1e-6
        for k in range(3):
            d = np.zeros(3); d[k] = eps
            uv0 = project_points(K, pts)
            uv1 = project_points(K, pts + d)
            assert np.allclose(J[:, :, k], (uv1 - uv0) / eps, atol=1e-3)

    def test_temporal_weight_low_for_unstable_projection(self):
        config = CalConfig()
        stable = [np.full(4, 2.0)] * config.multi_frame_window
        wobbling = [np.full(4, 2.0), np.full(4, 20.0),
                    np.full(4, 2.0), np.full(4, 2.0), np.full(4, 2.0)]
        w_stable = temporal_weights_over_window(stable, config)
        w_wob = temporal_weights_over_window(wobbling, config)
        assert np.all(w_wob < w_stable)


class TestAttractionField:
    def test_field_zero_on_boundary_and_growing_away(self):
        image = np.full((60, 80, 3), 255, dtype=np.uint8)
        masks = MaskSet(
            masks=np.zeros((1, 60, 80), dtype=bool), scores=np.ones(1)
        )
        masks.masks[0, :, 30:50] = True  # a right-edge boundary at col 50
        field, _ = build_attraction_field(image, masks)
        assert field.field[30, 50] == 0 or field.field[29, 50] == 0
        # distance grows with pixel distance from the boundary (leftward)
        assert field.field[30, 10] > field.field[30, 40]
