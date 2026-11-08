"""Rejected design variants (kept for transparency).

Pilot experiments (scripts/pilot_rejected_variants.py) tested two nowcasting
components for stale soil-water readings:

1. a weather-driven bucket drift model
       dx = -beta * ET0 * W + gamma * P_eff * (1 - W) - delta * W^2,
   with Hargreaves-Samani ET0, fitted per depth on the calibration year;
2. a correlation-indexed surrogate map x_u ~ a + b x_v learned from same-depth
   neighbours in the calibration year and fused by inverse-variance weighting.

On the 2012 evaluation days both increased the mean absolute error of the
depletion estimate relative to simple persistence of the last delivered reading
(the surrogate map failed because crop rotation changed inter-station relations
between years; the drift model was biased during spring recharge). They are
therefore not part of FURROW-R.
"""
from __future__ import annotations

import numpy as np

from ..data.metadata import hargreaves_et0, weather_drivers  # noqa: F401

MIN_OVERLAP = 90
N_SURROGATES = 4


def _fit_drift(Xc, ports, fc, ll, et0, p_eff):
    out = {}
    for p in np.unique(ports):
        rows_f, rows_y = [], []
        for u in np.where(ports == p)[0]:
            x = Xc[u]
            ok = ~np.isnan(x[1:]) & ~np.isnan(x[:-1])
            if ok.sum() < 10:
                continue
            xp = x[:-1][ok]
            W = np.clip((xp - ll[u]) / (fc[u] - ll[u]), 0, 1)
            e = et0[1:][ok]
            r = p_eff[1:][ok]
            rows_f.append(np.column_stack([-e * W, r * (1 - W), -W ** 2]))
            rows_y.append((x[1:] - x[:-1])[ok])
        A = np.vstack(rows_f)
        y = np.concatenate(rows_y)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        coef = np.clip(coef, 0, None)       # physical sign constraints
        res = y - A @ coef
        r2 = 1 - res.var() / y.var()
        out[int(p)] = dict(beta=float(coef[0]), gamma=float(coef[1]), delta=float(coef[2]),
                           sigma=float(np.sqrt(np.mean(res ** 2))), r2=float(r2), n=int(len(y)))
    return out


def _surrogates(Xc, ports, stations, m=N_SURROGATES):
    U = Xc.shape[0]
    idx = np.full((U, m), -1, int)
    A = np.zeros((U, m))
    B = np.zeros((U, m))
    S = np.full((U, m), np.inf)
    for u in range(U):
        cands = []
        for v in np.where((ports == ports[u]) & (stations != stations[u]))[0]:
            ok = ~np.isnan(Xc[u]) & ~np.isnan(Xc[v])
            if ok.sum() < MIN_OVERLAP:
                continue
            xv, xu = Xc[v][ok], Xc[u][ok]
            if np.std(xv) < 1e-6:
                continue
            b, a = np.polyfit(xv, xu, 1)
            s = float(np.std(xu - (a + b * xv), ddof=2))
            cands.append((s, v, a, b))
        cands.sort()
        for j, (s, v, a, b) in enumerate(cands[:m]):
            idx[u, j], A[u, j], B[u, j], S[u, j] = v, a, b, s
    return idx, A, B, S




def propagate(x0, start, t, ports, ll, fc, drift, et0, p_eff):
    """Propagate states from day ``start`` (exclusive) to day t with the drift model."""
    x = x0.copy()
    ok = ~np.isnan(x) & (start >= 0)
    if not ok.any():
        return x
    beta = np.array([drift[int(p)]["beta"] for p in ports])
    gamma = np.array([drift[int(p)]["gamma"] for p in ports])
    delta = np.array([drift[int(p)]["delta"] for p in ports])
    rng = fc - ll
    for d in range(int(start[ok].min()) + 1, t + 1):
        act = ok & (start < d)
        W = np.clip((x - ll) / rng, 0, 1)
        dx = -beta * et0[d] * W + gamma * p_eff[d] * (1 - W) - delta * W ** 2
        x = np.where(act, np.minimum(x + dx, fc + 0.02), x)
    return x
