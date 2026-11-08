"""Urgent verification task (U2): profile check and time-to-critical projection.

For an alerted unit (station s, depth p) the task reads the station's last
``HISTORY_DAYS`` days from the warm tier (all depths), fits an OLS trend to the
last 14 days of each depth, and projects the number of days until the unit
reaches the critical depletion D = 0.85. The output is the advisory payload.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

HISTORY_DAYS = 90
TREND_DAYS = 14
D_CRIT = 0.85


def verify(warm_cat: pd.DataFrame, station: str, port: int, t_end: pd.Timestamp, fc: float, ll: float):
    t0 = time.perf_counter()
    start = t_end - pd.Timedelta(days=HISTORY_DAYS - 1)
    months = {(p.year, p.month) for p in pd.period_range(start, t_end, freq="M")}
    sel = warm_cat[(warm_cat["station"] == station)]
    sel = sel[[(y, m) in months for y, m in zip(sel["year"], sel["month"])]]
    df = pd.concat([pd.read_parquet(p) for p in sel["path"]], ignore_index=True)
    df = df[(df["date"] >= start) & (df["date"] <= t_end)].sort_values("date")
    recent = df[df["date"] > t_end - pd.Timedelta(days=TREND_DAYS)]
    x = (recent["date"] - t_end).dt.days.to_numpy(float)
    out = {}
    for p in range(1, 6):
        y = recent[f"vw_p{p}"].to_numpy(float)
        ok = ~np.isnan(y)
        out[p] = float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 4 else np.nan
    y = recent[f"vw_p{port}"].to_numpy(float)
    ok = ~np.isnan(y)
    eta = np.nan
    if ok.sum() >= 4:
        b, a = np.polyfit(x[ok], y[ok], 1)
        x_crit = fc - D_CRIT * (fc - ll)
        if b < 0:
            eta = max(0.0, (x_crit - a) / b)
    payload = {"station": station, "port": port, "slopes": out, "days_to_critical": eta}
    return payload, time.perf_counter() - t0, int(sel["bytes"].sum()), len(sel)
