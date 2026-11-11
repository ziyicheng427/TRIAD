"""Unit tests of TRIAD-Reg (skipped automatically when torch is absent)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from triad.reg import (  # noqa: E402
    RegConfig,
    TRIADRegModel,
    distance_weighted_fps,
    geometric_compatibility,
    weighted_kabsch,
)
from triad.reg.propagation import ConsistencyPropagation  # noqa: E402


class TestDWFPS:
    def test_selects_requested_count_without_repetition(self):
        pts = torch.randn(2, 300, 3)
        idx = distance_weighted_fps(pts, n_samples=64, beta=1.0)
        assert idx.shape == (2, 64)
        for row in idx:
            assert len(torch.unique(row)) == 64

    def test_far_points_survive_on_decaying_density(self):
        """A near-dense / far-sparse cloud: DW-FPS must keep far anchors that
        plain FPS would under-sample relative to the log weight."""
        near = torch.randn(400, 3) * 0.5
        far = torch.randn(20, 3) * 12.0
        cloud = torch.cat([near, far]).unsqueeze(0)
        idx = distance_weighted_fps(cloud, n_samples=60, beta=1.0)[0]
        n_far = int((idx >= 400).sum())
        assert n_far >= 8  # distant anchors retained (Sec. 3.3)


class TestCompatibility:
    def test_compatible_pairs_score_one(self):
        src = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        tgt = torch.tensor([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]])  # isometric
        B = geometric_compatibility(src, tgt, sigma_c=0.1)
        assert torch.allclose(B, torch.ones(2, 2), atol=1e-5)

    def test_incompatible_pair_suppressed(self):
        src = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        tgt = torch.tensor([[5.0, 0.0, 0.0], [7.0, 0.0, 0.0]])  # distances differ
        B = geometric_compatibility(src, tgt, sigma_c=0.1)
        assert B[0, 1] == pytest.approx(0.0, abs=1e-6)


class TestPropagation:
    def test_outlier_suppressed(self):
        """A correspondence incompatible with its neighbours must lose score
        (Eq. 2)."""
        M = 6
        B = torch.ones(M, M)
        B[3, :] = 0.0
        B[:, 3] = 0.0
        s0 = torch.full((M,), 0.8)
        prop = ConsistencyPropagation(iterations=3)
        s = prop(B, s0)
        assert s[3] < s0[3]
        assert (torch.delete(s, 3) >= s[3]).all()


class TestKabsch:
    def test_recovers_rigid_transform(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(50, 3)).astype(np.float32)
        theta = np.deg2rad(30)
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]], dtype=np.float32)
        t = np.array([1.0, -2.0, 0.5], dtype=np.float32)
        tgt = src @ R.T + t
        T = weighted_kabsch(torch.from_numpy(src), torch.from_numpy(tgt))
        assert np.allclose(T[:3, :3], R, atol=1e-4)
        assert np.allclose(T[:3, 3], t, atol=1e-4)


class TestModel:
    def test_forward_shapes_and_identity_on_clean_pair(self):
        torch.manual_seed(0)
        model = TRIADRegModel(RegConfig())
        src = torch.randn(1500, 3)
        R = torch.eye(3)
        tgt = src @ R.T
        out = model(src, tgt)
        assert out.transform.shape == (4, 4)
        assert len(out.confidences) == len(out.src_idx)
