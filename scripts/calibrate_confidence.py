#!/usr/bin/env python3
"""Confidence-map calibration CLI — Sec. 5.4, Table 4.

Fits one module's isotonic confidence map from labelled probe observations
(≥ 5000 per module) and reports ECE before and after calibration.

    python scripts/calibrate_confidence.py --module cal \
        --scores cal_scores.npy --labels cal_labels.npy --out outputs/phi_cal.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from triad.common import (
    expected_calibration_error,
    fit_isotonic,
    identity_map,
)

LABEL_MEANING = {
    "cal": "valid edge matches against reference extrinsics",
    "reg": "inliers against ground-truth transformations",
    "map": "reliable cell updates against a reference map",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, choices=list(LABEL_MEANING))
    parser.add_argument("--scores", required=True, help=".npy raw probe scores")
    parser.add_argument("--labels", required=True, help=".npy binary labels")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    scores = np.load(args.scores).astype(float)
    labels = np.load(args.labels).astype(float)
    print(f"module={args.module}; labels: {LABEL_MEANING[args.module]}")
    print(f"observations: {len(scores)}")

    raw = identity_map(scores)
    ece_raw = expected_calibration_error(raw, labels)

    phi = fit_isotonic(scores, labels)
    calibrated = phi(scores)
    ece_cal = expected_calibration_error(calibrated, labels)

    print(f"ECE raw        = {ece_raw:.3f}")
    print(f"ECE calibrated = {ece_cal:.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez(args.out, knots=phi.knots, values=phi.values,
                 ece_raw=ece_raw, ece_calibrated=ece_cal)
        print(f"confidence map saved to {args.out}")


if __name__ == "__main__":
    main()
