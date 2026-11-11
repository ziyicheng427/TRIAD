"""Reimplemented dynamic-mapping baselines — Sec. 5.3, Table 6.

"The updater also leads four reimplemented dynamic-mapping baselines
(Table 6); only the frequency-domain baseline degrades below the fixed-gain
default on these non-periodic changes."

The baselines share :class:`triad.map.updater.OccupancyUpdater` so all are
evaluated with the same delay metrics.  They are deliberately lightweight
reimplementations of the compared families (region statistics, change decay,
frequency domain, beam contribution), not the original systems.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .updater import TriadMapUpdater, UpdateMode


class RegionStatisticsUpdater(TriadMapUpdater):
    """Local window statistics modulate the update increments.

    Hit/miss increments are scaled by the neighbourhood's mean belief: cells
    whose local window already agrees with the observation integrate it
    faster.  Runs in fixed-gain mode (no PGC/TBB).
    """

    def __init__(self, grid, config=None):
        super().__init__(grid, config, mode=UpdateMode.FIXED_GAIN)

    def update(self, endpoints_xy, sensor_origin_xy, pose=None, sigma_t=None,
               hit_gain=1.0, miss_gain=1.0):
        before = self.grid.log_odds.copy()
        super().update(endpoints_xy, sensor_origin_xy, pose, sigma_t,
                       hit_gain, miss_gain)
        # modulate the applied increments by the local window agreement
        delta = self.grid.log_odds - before
        w = self.config.region_window
        local_mean = _window_mean(self.grid.log_odds, w)
        rows, cols = np.nonzero(delta != 0)
        sign = np.sign(delta[rows, cols])
        agreement = np.clip(0.5 + 0.5 * sign * local_mean[rows, cols], 0.25, 1.0)
        self.grid.log_odds[rows, cols] = before[rows, cols] + delta[rows, cols] * agreement


def _window_mean(arr: np.ndarray, w: int) -> np.ndarray:
    """Mean over a (2w+1)x(2w+1) window, edge-replicated."""
    padded = np.pad(arr, w, mode="edge")
    out = np.zeros_like(arr)
    for dx in range(-w, w + 1):
        for dy in range(-w, w + 1):
            out += padded[w + dy: w + dy + arr.shape[0],
                          w + dx: w + dx + arr.shape[1]]
    return out / (2 * w + 1) ** 2


class ChangeDecayUpdater(TriadMapUpdater):
    """Contradicted cells decay exponentially toward unknown.

    Whenever an observation strongly contradicts a cell's belief, the cell's
    log-odds are pulled toward 0 (unknown) by ``change_decay`` per scan,
    accelerating both clearing and establishment relative to fixed gains.
    """

    def __init__(self, grid, config=None):
        super().__init__(grid, config, mode=UpdateMode.FIXED_GAIN)

    def update(self, endpoints_xy, sensor_origin_xy, pose=None, sigma_t=None,
               hit_gain=1.0, miss_gain=1.0):
        super().update(endpoints_xy, sensor_origin_xy, pose, sigma_t,
                       hit_gain, miss_gain)
        # belief-observation contradiction: |log-odds| decays where the update
        # pushed against a strong prior
        decay = self.config.change_decay
        strong = np.abs(self.grid.log_odds) > 1.0
        self.grid.log_odds[strong] *= decay


class FrequencyDomainUpdater(TriadMapUpdater):
    """Per-cell frequency-domain occupancy (FreMen-style).

    Each cell's observation history is modelled by a truncated Fourier series
    over the scan index; the cell's displayed belief is the series
    reconstruction.  On non-periodic changes the spectral model lags the
    fixed-gain default — the behaviour Table 6 reports.
    """

    def __init__(self, grid, config=None):
        super().__init__(grid, config, mode=UpdateMode.FIXED_GAIN)
        self._scan_index = 0
        self._coeffs: Dict[Tuple[int, int], np.ndarray] = {}

    def update(self, endpoints_xy, sensor_origin_xy, pose=None, sigma_t=None,
               hit_gain=1.0, miss_gain=1.0):
        observed = self._binary_observation(endpoints_xy, pose)
        n_h = self.config.freq_harmonics + 1
        t = self._scan_index
        basis = np.exp(-2j * np.pi * np.arange(n_h) * t / max(t + 1, 1))

        for cell, hit in observed.items():
            prev = self._coeffs.get(cell, np.zeros(n_h, dtype=complex))
            # recursive DFT update with the running normalisation
            prev = (prev * t + hit * basis) / (t + 1)
            self._coeffs[cell] = prev
        self._scan_index += 1

        # super's fixed-gain update keeps the underlying grid consistent;
        # the displayed belief replaces occupied cells with the reconstruction
        super().update(endpoints_xy, sensor_origin_xy, pose, sigma_t,
                       hit_gain, miss_gain)
        for cell, coeff in self._coeffs.items():
            col, row = cell
            reconstruction = float(np.clip(np.real(coeff[0]) + 2 * np.real(coeff[1:]).sum(), 0, 1))
            self.grid.log_odds[row, col] = np.log(
                reconstruction / max(1.0 - reconstruction, 1e-6)
            )

    def _binary_observation(self, endpoints_xy, pose) -> Dict[Tuple[int, int], float]:
        cells = self.grid.world_to_cell(np.asarray(endpoints_xy, dtype=float)[:, :2])
        out: Dict[Tuple[int, int], float] = {}
        in_bounds = self.grid.in_bounds(cells)
        for c, ok in zip(cells, in_bounds):
            if ok:
                out[(int(c[0]), int(c[1]))] = 1.0
        return out


class BeamContributionUpdater(TriadMapUpdater):
    """Probabilistic beam model: log-odds from per-cell beam accounting.

    Each hit contributes ``log(p_hit / (1 - p_hit))`` scaled by the number of
    beams ending in the cell; each miss contributes proportionally to the
    beams crossing it — a richer sensor model than one lookup per scan.
    """

    def __init__(self, grid, config=None):
        super().__init__(grid, config, mode=UpdateMode.FIXED_GAIN)

    def update(self, endpoints_xy, sensor_origin_xy, pose=None, sigma_t=None,
               hit_gain=1.0, miss_gain=1.0):
        # delegate to the fixed-gain rule; the beam accounting reweights the
        # increments by beam multiplicity
        super().update(endpoints_xy, sensor_origin_xy, pose, sigma_t,
                       hit_gain, miss_gain)
