"""Pilot that documents two rejected nowcasting variants (see
src/furrow/retrieval/rejected_variants.py): mean absolute error of the depletion
estimate for root-zone units on 2012 days (mixed connectivity, 3 schedule seeds).

  persistence : last delivered reading (used by FURROW-R)
  drift       : last reading propagated with the weather-driven bucket model
  surrogate   : inverse-variance fusion of the own (drifted) estimate and the
                2011-calibrated linear maps from same-depth neighbours

Output: results/processed/pilot_rejected_variants.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from furrow.data.metadata import weather_drivers  # noqa: E402
from furrow.data.pipeline import load  # noqa: E402
from furrow.retrieval import rejected_variants as rv  # noqa: E402
from furrow.simulator.reporting import draw_schedule, station_horizon  # noqa: E402

if __name__ == "__main__":
    b = load(ROOT / "data/processed")
    ctx = b.ctx
    ports = b.ports
    cal = b.dates.year == 2011
    Xc = np.where((b.F_on == 0) & cal[None, :], b.X, np.nan)
    et0, p_eff, _ = weather_drivers(b.dates, b.weather)
    drift = rv._fit_drift(Xc, ports, ctx.fc, ctx.ll, et0, p_eff)
    sidx, sa, sb, ss = rv._surrogates(Xc, ports, b.units.station.to_numpy())
    root = b.root_units
    sd_drift = np.array([drift[int(p)]["sigma"] for p in ports])
    rows = []
    for seed in range(3):
        P, ph = draw_schedule(42, "mixed", np.random.default_rng(seed))
        for t in np.where(b.dates.year == 2012)[0]:
            hz = station_horizon(t, P, ph)[ctx.unit_station]
            truth = np.where(b.F_off[:, t] == 0, b.X[:, t], np.nan)
            li, lv = ctx.store.last(hz, t)
            xd = rv.propagate(lv, li, t, ports, ctx.ll, ctx.fc, drift, et0, p_eff)
            age = np.where(li >= 0, t - li, 61)
            var_own = 0.003 ** 2 + age * sd_drift ** 2
            num = np.where(np.isfinite(xd), xd / var_own, 0.0)
            den = np.where(np.isfinite(xd), 1.0 / var_own, 0.0)
            for j in range(sidx.shape[1]):
                v = sidx[:, j]
                ok = v >= 0
                vv = np.where(ok, v, 0)
                xs = sa[:, j] + sb[:, j] * xd[vv]
                vs = 2.0 * (ss[:, j] ** 2 + sb[:, j] ** 2 * var_own[vv])
                good = ok & np.isfinite(xs) & np.isfinite(vs)
                num += np.where(good, xs / np.where(good, vs, 1), 0)
                den += np.where(good, 1 / np.where(good, vs, 1), 0)
            xf = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
            rngu = ctx.fc - ctx.ll
            for u in root:
                if np.isnan(truth[u]) or np.isnan(lv[u]):
                    continue
                d0 = (ctx.fc[u] - truth[u]) / rngu[u]
                rows.append(dict(seed=seed, t=t, age=int(age[u]),
                                 persistence=abs((ctx.fc[u] - lv[u]) / rngu[u] - d0),
                                 drift=abs((ctx.fc[u] - xd[u]) / rngu[u] - d0),
                                 surrogate=abs((ctx.fc[u] - xf[u]) / rngu[u] - d0)))
    df = pd.DataFrame(rows)
    out = df.groupby(pd.cut(df.age, [-1, 0, 3, 7, 61]))[["persistence", "drift", "surrogate"]].mean()
    out.loc["all"] = df[["persistence", "drift", "surrogate"]].mean()
    out.to_csv(ROOT / "results/processed/pilot_rejected_variants.csv")
    print(out.round(4))
