"""Unit tests of the common layer (no torch, no datasets)."""

import numpy as np
import pytest

from triad.common import (
    C2COperator,
    IsotonicConfidenceMap,
    expected_calibration_error,
    exp_se3,
    exp_so3,
    fit_isotonic,
    hat3,
    huber_weight,
    identity_map,
    log_se3,
    log_so3,
    point_jacobian_right,
    transform_points,
)


class TestSE3:
    def test_hat_skew(self):
        v = np.array([1.0, 2.0, 3.0])
        S = hat3(v)
        assert np.allclose(S + S.T, 0.0)
        assert np.allclose(S @ v, 0.0)

    def test_so3_roundtrip(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            w = rng.normal(size=3) * 1.5
            R = exp_so3(w)
            assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
            assert np.allclose(log_so3(R), w, atol=1e-6)

    def test_se3_roundtrip(self):
        rng = np.random.default_rng(1)
        xi = np.concatenate([rng.normal(size=3), rng.normal(size=3)])
        T = exp_se3(xi)
        assert np.allclose(log_se3(T), xi, atol=1e-6)

    def test_point_jacobian_matches_finite_difference(self):
        rng = np.random.default_rng(2)
        xi = rng.normal(size=6) * 0.3
        T = exp_se3(xi)
        pts = rng.normal(size=(5, 3))
        J = point_jacobian_right(T, pts)
        eps = 1e-6
        for k in range(6):
            d = np.zeros(6)
            d[k] = eps
            numeric = (transform_points(T @ exp_se3(d), pts)
                       - transform_points(T, pts)) / eps
            assert np.allclose(J[:, :, k], numeric, atol=1e-4)


class TestC2C:
    def test_huber_weight_bounds(self):
        r = np.array([0.5, 1.345, 5.0])
        w = huber_weight(r, delta=1.345)
        assert w[0] == 1.0
        assert w[1] == 1.0
        assert w[2] == pytest.approx(1.345 / 5.0)

    def test_weights_decrease_with_disagreement(self):
        op = C2COperator(identity_map)
        from triad.common.c2c import ConsistencyProbe

        probe_hi = ConsistencyProbe(scores=np.array([0.9, 0.9]))
        probe_lo = ConsistencyProbe(scores=np.array([0.2, 0.2]))
        residuals = np.array([1.0, 1.0])
        assert np.all(
            op.irls_weights(residuals, probe_hi)
            >= op.irls_weights(residuals, probe_lo)
        )


class TestIsotonic:
    def test_pav_is_monotone(self):
        from triad.common.confidence_calibration import _pool_adjacent_violators

        y = np.array([0.9, 0.1, 0.2, 0.8, 0.05, 0.7])
        fitted = _pool_adjacent_violators(y)
        assert np.all(np.diff(fitted) >= -1e-12)

    def test_isotonic_map_orders_scores(self):
        rng = np.random.default_rng(3)
        scores = rng.uniform(0, 1, 6000)
        labels = (rng.uniform(0, 1, 6000) < scores).astype(float)
        phi = fit_isotonic(scores, labels)
        lo, hi = phi(np.array([0.1])), phi(np.array([0.9]))
        assert lo[0] < hi[0]

    def test_ece_perfect_calibration(self):
        p = np.array([0.25, 0.25, 0.75, 0.75])
        y = np.array([0.0, 1.0, 0.0, 1.0])
        assert expected_calibration_error(p, y) == pytest.approx(0.5, abs=0.01)


def test_isotonic_confidence_map_step_interpolation():
    phi = IsotonicConfidenceMap(
        knots=np.array([0.0, 0.5, 1.0]), values=np.array([0.2, 0.2, 0.9])
    )
    out = phi(np.array([-0.5, 0.2, 0.7, 2.0]))
    assert np.allclose(out, [0.2, 0.2, 0.9, 0.9])
