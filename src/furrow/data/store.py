"""Layer 3: tiered agricultural data store.

Tiers (physical layout used by the experiments)
-------------------------------------------------
* warm tier    : Parquet partition per (station, calendar month) with daily rows of
                 all depths and variables plus online QC flags (cloud object store);
* summary tier : one Parquet partition per (farm, ISO year) holding weekly
                 aggregates (mean, min, max, valid count) of VWC per unit and weekly
                 weather aggregates (cloud object store, replicated to fog nodes);
* hot tier     : last 7 delivered days per unit kept by the farm gateway (edge).

Partition sizes in bytes are measured from the files actually written.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TierCatalog:
    warm: pd.DataFrame        # station, year, month, path, bytes
    summary: pd.DataFrame     # year, path, bytes
    hot_bytes_per_unit_day: int = 16


def write_warm_tier(full: pd.DataFrame, flags_long: pd.DataFrame, root: Path) -> pd.DataFrame:
    root.mkdir(parents=True, exist_ok=True)
    df = full.merge(flags_long, on=["station", "port", "date"], how="left")
    wide = df.pivot_table(index=["station", "date"], columns="port",
                          values=["vw", "temp_c", "ec", "qc"], dropna=False)
    wide.columns = [f"{v}_p{p}" for v, p in wide.columns]
    wide = wide.reset_index()
    wide["year"] = wide["date"].dt.year
    wide["month"] = wide["date"].dt.month
    rows = []
    for (s, y, m), g in wide.groupby(["station", "year", "month"]):
        p = root / f"station={s}" / f"{y:04d}-{m:02d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        g.drop(columns=["year", "month"]).to_parquet(p, index=False, compression="snappy")
        rows.append(dict(station=s, year=y, month=m, path=str(p), bytes=p.stat().st_size))
    return pd.DataFrame(rows)


def weekly_summary(units: pd.DataFrame, dates: pd.DatetimeIndex, X: np.ndarray, F_on: np.ndarray,
                   weather: pd.DataFrame) -> pd.DataFrame:
    """Weekly aggregates per unit (weeks start on Monday)."""
    Xv = np.where(F_on >= 0, X, np.nan)
    wk = dates.to_period("W-SUN").start_time
    out = []
    for i, (s, p) in enumerate(units[["station", "port"]].itertuples(index=False)):
        ser = pd.Series(Xv[i], index=dates)
        g = ser.groupby(wk)
        out.append(pd.DataFrame({"station": s, "port": p, "week": g.mean().index,
                                 "vw_mean": g.mean().values, "vw_min": g.min().values,
                                 "vw_max": g.max().values, "n": g.count().values}))
    summ = pd.concat(out, ignore_index=True)
    w = weather.set_index("date").reindex(dates)
    w["week"] = wk
    ww = w.groupby("week").agg(tmax_mean=("tmax_c", "mean"), tmin_mean=("tmin_c", "mean"),
                               precip_sum=("precip_mm", "sum")).reset_index()
    return summ.merge(ww, on="week", how="left")


def write_summary_tier(summ: pd.DataFrame, root: Path) -> pd.DataFrame:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    summ = summ.copy()
    summ["year"] = summ["week"].dt.year
    for y, g in summ.groupby("year"):
        p = root / f"summary_{y}.parquet"
        g.drop(columns=["year"]).to_parquet(p, index=False, compression="snappy")
        rows.append(dict(year=y, path=str(p), bytes=p.stat().st_size))
    return pd.DataFrame(rows)


def warm_partitions(catalog: pd.DataFrame, stations, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    months = pd.period_range(start, end, freq="M")
    keys = {(p.year, p.month) for p in months}
    sel = catalog[catalog["station"].isin(stations)]
    return sel[[(y, m) in keys for y, m in zip(sel["year"], sel["month"])]]
