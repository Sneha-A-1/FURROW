"""Layer 4 (strategic path): analytical query templates at two retrieval fidelities.

Every template can be answered from
  * the warm tier  (fidelity 'full'   : daily partitions, exact answer), or
  * the summary tier (fidelity 'summary': weekly aggregates, approximate answer).

FURROW-R chooses the fidelity per query at dispatch time from the scheduler's slack
(see furrow.scheduling.furrow). This module executes both variants on the real data,
measures their CPU time (Parquet read + decode + compute), bytes and partition
counts, and reports the answer error of the summary variant against the exact one.

Templates
---------
S1 water_status   mean VWC per depth over the window for a station subset
S2 deficit_days   per-station count of days with D >= 0.5 at 0.3 m
S3 spatial_cv     coefficient of variation across stations of mean VWC per depth
S4 drying_rate    per-station OLS slope of 0.3 m VWC (m3 m-3 day-1)
S5 analogues      top-5 most similar historical 14-day windows (week-aligned) for a
                  target unit, Euclidean distance; error = 1 - recall@5 vs exact
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

TEMPLATES = ["S1_water_status", "S2_deficit_days", "S3_spatial_cv", "S4_drying_rate",
             "S5_analogues"]
SLOPE_SCALE = 5e-4      # m3 m-3 day-1, floor for relative slope error
K_ANALOGUE = 5


@dataclass
class QueryInstance:
    template: str
    t_end: pd.Timestamp
    window: int
    stations: list
    target: tuple | None = None   # (station, port) for S5 analogues


@dataclass
class Execution:
    answer: object
    cpu_s: float
    bytes: int
    n_parts: int


class StrategicEngine:
    def __init__(self, warm_cat: pd.DataFrame, summ_cat: pd.DataFrame, fc: dict, ll: dict):
        self.warm = warm_cat
        self.summ = summ_cat
        self.fc = fc            # (station, port) -> FC
        self.ll = ll
        self.weather_full = None

    # ---------------- data access ----------------
    def _read_warm(self, stations, start, end):
        months = {(p.year, p.month) for p in pd.period_range(start, end, freq="M")}
        sel = self.warm[self.warm["station"].isin(stations)]
        sel = sel[[(y, m) in months for y, m in zip(sel["year"], sel["month"])]]
        df = pd.concat([pd.read_parquet(p) for p in sel["path"]], ignore_index=True)
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        return df, int(sel["bytes"].sum()), len(sel)

    def _read_summary(self, start, end):
        years = range(start.year, end.year + 1)
        sel = self.summ[self.summ["year"].isin(list(years))]
        df = pd.concat([pd.read_parquet(p) for p in sel["path"]], ignore_index=True)
        df = df[(df["week"] >= start - pd.Timedelta(days=6)) & (df["week"] <= end)]
        return df, int(sel["bytes"].sum()), len(sel)

    # ---------------- templates: full ----------------
    def run(self, q: QueryInstance, fidelity: str, weather: pd.DataFrame) -> Execution:
        t0 = time.perf_counter()
        start = q.t_end - pd.Timedelta(days=q.window - 1)
        if q.template == "S5_analogues":
            ans, nb, npart = self._analogues(q, fidelity)
        elif fidelity == "full":
            df, nb, npart = self._read_warm(q.stations, start, q.t_end)
            ans = self._full_answer(q, df)
        else:
            sm, nb, npart = self._read_summary(start, q.t_end)
            ans = self._summary_answer(q, sm, start)
        return Execution(ans, time.perf_counter() - t0, nb, npart)

    def _full_answer(self, q, df):
        vw = {p: df.groupby("station")[f"vw_p{p}"] for p in range(1, 6)}
        if q.template == "S1_water_status":
            return np.array([df[f"vw_p{p}"].mean() for p in range(1, 6)])
        if q.template == "S2_deficit_days":
            out = {}
            for s, g in df.groupby("station"):
                D = (self.fc[(s, 1)] - g["vw_p1"]) / (self.fc[(s, 1)] - self.ll[(s, 1)])
                out[s] = int(np.sum(D >= 0.5))
            return out
        if q.template == "S3_spatial_cv":
            res = []
            for p in range(1, 6):
                m = vw[p].mean().dropna()
                res.append(m.std() / m.mean() if len(m) > 1 else np.nan)
            return np.array(res)
        if q.template == "S4_drying_rate":
            out = {}
            for s, g in df.groupby("station"):
                y = g["vw_p1"].to_numpy(float)
                x = (g["date"] - g["date"].min()).dt.days.to_numpy(float)
                ok = ~np.isnan(y)
                out[s] = float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 5 else np.nan
            return out
        raise ValueError(q.template)

    # ---------------- templates: summary ----------------
    def _summary_answer(self, q, sm, start):
        s = sm[sm["station"].isin(q.stations)]
        if q.template == "S1_water_status":
            return np.array([np.average(g["vw_mean"].fillna(0), weights=g["n"]) if g["n"].sum() > 0 else np.nan
                             for _, g in s.groupby("port")])
        if q.template == "S2_deficit_days":
            out = {}
            for st, g in s[s["port"] == 1].groupby("station"):
                D = (self.fc[(st, 1)] - g["vw_mean"]) / (self.fc[(st, 1)] - self.ll[(st, 1)])
                out[st] = int(np.sum(np.where(D >= 0.5, g["n"], 0)))
            return out
        if q.template == "S3_spatial_cv":
            res = []
            s = s.assign(wsum=s["vw_mean"].fillna(0) * s["n"])
            agg = s.groupby(["port", "station"])[["wsum", "n"]].sum()
            means = (agg["wsum"] / agg["n"].where(agg["n"] > 0)).dropna()
            for p in range(1, 6):
                m = means.xs(p, level="port") if p in means.index.get_level_values(0) else pd.Series(dtype=float)
                res.append(m.std() / m.mean() if len(m) > 1 else np.nan)
            return np.array(res)
        if q.template == "S4_drying_rate":
            out = {}
            for st, g in s[s["port"] == 1].groupby("station"):
                y = g["vw_mean"].to_numpy(float)
                x = ((g["week"] - g["week"].min()).dt.days + 3).to_numpy(float)
                ok = ~np.isnan(y)
                out[st] = float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 2 else np.nan
            return out
        raise ValueError(q.template)

    # ---------------- S5 analogue retrieval ----------------
    def _analogues(self, q, fidelity):
        s0, p0 = q.target
        end = q.t_end
        wk_end = (end - pd.Timedelta(days=end.dayofweek + 1)).normalize()   # last complete Sunday
        q_start = wk_end - pd.Timedelta(days=13)                            # Monday two weeks earlier
        hist_start = pd.Timestamp(f"{end.year - 1}-01-01")
        hist_start = hist_start + pd.Timedelta(days=(7 - hist_start.dayofweek) % 7)  # first Monday
        n_weeks = (q_start - hist_start).days // 7          # complete weeks before the query window
        if fidelity == "full":
            df, nb, npart = self._read_warm(q.stations, hist_start, wk_end)
            piv = df.pivot_table(index="date", columns="station", values=f"vw_p{p0}", dropna=False)
            piv = piv.reindex(pd.date_range(hist_start, wk_end))
            if s0 not in piv.columns:
                return [], nb, npart
            M = piv.to_numpy(float)                          # days x stations
            qv = piv[s0].to_numpy(float)[-14:]
            if np.isnan(qv).any() or n_weeks < 2:
                return [], nb, npart
            starts = np.arange(0, (n_weeks - 1) * 7, 7)      # window = weeks i, i+1 (ends before query)
            W = np.stack([M[i:i + 14] for i in starts])      # windows x 14 x stations
            d = np.sqrt(np.sum((W - qv[None, :, None]) ** 2, axis=1))   # windows x stations
            d = np.where(np.isnan(d), np.inf, d)
            cols = piv.columns.to_numpy()
        else:
            sm, nb, npart = self._read_summary(hist_start, wk_end)
            g = sm[(sm["port"] == p0) & sm["station"].isin(q.stations)]
            weeks = pd.date_range(hist_start, wk_end - pd.Timedelta(days=6), freq="7D")
            feats = {}
            for k in ("vw_mean", "vw_min", "vw_max", "n"):
                feats[k] = g.pivot_table(index="week", columns="station", values=k, dropna=False).reindex(weeks)
            cols = feats["vw_mean"].columns.to_numpy()
            if s0 not in feats["vw_mean"].columns or n_weeks < 2:
                return [], nb, npart
            F = np.stack([feats[k].to_numpy(float) for k in ("vw_mean", "vw_min", "vw_max")], axis=-1)  # weeks x st x 3
            N = feats["n"].to_numpy(float)
            F = np.where((N >= 7)[..., None], F, np.nan)
            j0 = list(cols).index(s0)
            qf = np.r_[F[-2, j0], F[-1, j0]]
            if np.isnan(qf).any():
                return [], nb, npart
            wts = np.array([7, 1, 1, 7, 1, 1], float)
            Wf = np.concatenate([F[:n_weeks - 1], F[1:n_weeks]], axis=-1)             # windows x st x 6
            d = np.sqrt(np.sum(wts * (Wf - qf) ** 2, axis=-1))
            d = np.where(np.isnan(d), np.inf, d)
        flat = np.argsort(d, axis=None, kind="stable")[:K_ANALOGUE]
        res = []
        for f in flat:
            i, j = np.unravel_index(f, d.shape)
            if np.isfinite(d[i, j]):
                res.append((cols[j], hist_start + pd.Timedelta(days=7 * int(i))))
        return res, nb, npart


def answer_error(template: str, exact, approx) -> float:
    """Normalised error of the summary answer (0 = identical, capped at 1)."""
    if template == "S5_analogues":
        if not exact:
            return np.nan
        hits = len(set(exact) & set(approx))
        return 1.0 - hits / len(exact)
    if isinstance(exact, dict):
        errs = []
        for k, v in exact.items():
            a = approx.get(k, np.nan)
            if v is None or np.isnan(v) or a is None or np.isnan(a):
                continue
            if template == "S2_deficit_days":
                errs.append(abs(a - v) / max(abs(v), 1.0))
            else:
                errs.append(abs(a - v) / max(abs(v), SLOPE_SCALE))
        return float(min(1.0, np.mean(errs))) if errs else np.nan
    e = np.asarray(exact, float)
    a = np.asarray(approx, float)
    ok = ~np.isnan(e) & ~np.isnan(a) & (np.abs(e) > 0)
    return float(min(1.0, np.mean(np.abs(a[ok] - e[ok]) / np.abs(e[ok])))) if ok.any() else np.nan
