"""Layer 2: data-quality control for soil-water readings.

Two QC passes are implemented:

* ``online_qc``  - causal flags that use only the current and past readings of a
  unit. This is what the platform can compute when a reading arrives and is the
  only QC visible to the retrieval algorithms.
* ``offline_qc`` - two-sided flags (past and future readings) used solely to
  build the evaluation reference (ground truth). It is never visible to the
  algorithms under test.

Flag codes: 0 = pass, 1 = suspect, 2 = fail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VW_MIN, VW_MAX = 0.0, 0.60          # m3/m3, physically admissible range for 5TE probes
SPIKE_DELTA = 0.06                  # m3/m3 departure from local median
RAIN_EXPLAINS_MM = 2.0              # rain (t, t-1) that can explain a rise at shallow ports
FLAT_RUN = 10                       # identical consecutive readings -> stuck sensor
QUALITY_WEIGHT = {0: 1.0, 1: 0.25, 2: 0.0}


def _rolling_prev_median(x: np.ndarray, k: int = 3) -> np.ndarray:
    """Median of the previous k *valid* readings (NaN-aware), causal."""
    out = np.full_like(x, np.nan)
    buf: list[float] = []
    for i, v in enumerate(x):
        if buf:
            out[i] = float(np.median(buf[-k:]))
        if not np.isnan(v):
            buf.append(v)
    return out


def online_qc(vw: np.ndarray, port: int, rain2d: np.ndarray) -> np.ndarray:
    """Causal QC flags for one unit's daily series.

    vw     : daily VWC (NaN = no reading)
    port   : sensor port (1..5), shallow ports 1-2 may legitimately rise after rain
    rain2d : precipitation of day t plus day t-1 (mm), aligned with vw
    """
    n = len(vw)
    flag = np.full(n, -1, dtype=np.int8)  # -1 = no reading
    med = _rolling_prev_median(vw, 3)
    run = 0
    last = np.nan
    for i in range(n):
        v = vw[i]
        if np.isnan(v):
            continue
        f = 0
        if v < VW_MIN or v > VW_MAX:
            f = 2
        else:
            if not np.isnan(med[i]):
                d = v - med[i]
                explained = d > 0 and port <= 2 and rain2d[i] >= RAIN_EXPLAINS_MM
                if abs(d) > SPIKE_DELTA and not explained:
                    f = 1
            run = run + 1 if (not np.isnan(last) and v == last) else 1
            last = v
            if run >= FLAT_RUN:
                f = max(f, 1)
        flag[i] = f
    return flag


def offline_qc(vw: np.ndarray) -> np.ndarray:
    """Two-sided flags for the evaluation reference (uses future readings)."""
    n = len(vw)
    flag = np.full(n, -1, dtype=np.int8)
    s = pd.Series(vw)
    med = s.rolling(7, center=True, min_periods=3).median().to_numpy()
    for i in range(n):
        v = vw[i]
        if np.isnan(v):
            continue
        if v < VW_MIN or v > VW_MAX:
            flag[i] = 2
        elif not np.isnan(med[i]) and abs(v - med[i]) > SPIKE_DELTA:
            flag[i] = 2
        else:
            flag[i] = 0
    return flag


def build_unit_matrix(full: pd.DataFrame, weather: pd.DataFrame):
    """Return dense arrays (units x days) for VWC plus online/offline flags.

    Units are (station, port) pairs ordered by station then port.
    """
    dates = pd.DatetimeIndex(sorted(full["date"].unique()))
    w = weather.set_index("date").reindex(dates)
    rain = w["precip_mm"].fillna(0.0).to_numpy()
    rain2d = rain + np.concatenate([[0.0], rain[:-1]])
    units = full[["station", "port"]].drop_duplicates().sort_values(["station", "port"])
    units = units.reset_index(drop=True)
    piv = full.pivot_table(index=["station", "port"], columns="date", values="vw", dropna=False)
    piv = piv.reindex(pd.MultiIndex.from_frame(units)).reindex(columns=dates)
    X = piv.to_numpy(float)
    F_on = np.vstack([online_qc(X[i], int(units.port[i]), rain2d) for i in range(len(units))])
    F_off = np.vstack([offline_qc(X[i]) for i in range(len(units))])
    return units, dates, X, F_on, F_off
