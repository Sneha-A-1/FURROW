"""Station reporting (delivery) model.

Each station s delivers its buffered daily readings every P_s days with phase
phi_s, so on decision day t the readings up to day h_s(t) = t - ((t - phi_s) mod P_s)
are visible. Reporting intervals are an infrastructure assumption (rural LPWAN /
cellular duty-cycling), not part of the agricultural data set.
"""
from __future__ import annotations

import numpy as np

SCENARIOS = {
    "daily": ([1], [1.0]),
    "mixed": ([1, 3, 7], [0.5, 0.3, 0.2]),
    "poor": ([3, 7, 14], [0.4, 0.4, 0.2]),
}


def draw_schedule(n_stations: int, scenario: str, rng: np.random.Generator):
    vals, probs = SCENARIOS[scenario]
    P = rng.choice(vals, size=n_stations, p=probs)
    phase = np.array([rng.integers(0, p) for p in P])
    return P, phase


def station_horizon(t: int, P: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return t - ((t - phase) % P)
