"""End-to-end data preparation (Layers 1-3) and loader for processed artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .ingest import CAF_SHA256, complete_grid, load_cookfarm, sha256, to_long
from .metadata import build_index
from .qc import build_unit_matrix
from .store import weekly_summary, write_summary_tier, write_warm_tier
from ..retrieval.triage import EvidenceStore, StalenessModel, TriageContext

CAL_YEAR, EVAL_YEAR = 2011, 2012


def prepare(rda: Path, out: Path, verify_hash: bool = True) -> dict:
    if verify_hash:
        h = sha256(rda)
        if h != CAF_SHA256:
            raise RuntimeError(f"unexpected checksum {h}")
    out.mkdir(parents=True, exist_ok=True)
    cf = load_cookfarm(rda)
    long, weather, stations = to_long(cf)
    full = complete_grid(long)
    units, dates, X, F_on, F_off = build_unit_matrix(full, weather)
    cal = dates.year == CAL_YEAR
    idx = build_index(units, dates, X, F_on, weather, stations, cal)
    stale = StalenessModel.fit(X, F_on, units["port"].to_numpy(), dates, cal, idx.fc, idx.ll)

    np.save(out / "X.npy", X)
    np.save(out / "F_on.npy", F_on)
    np.save(out / "F_off.npy", F_off)
    np.save(out / "stale_sigma.npy", stale.sigma)
    np.save(out / "stale_clim.npy", stale.clim)
    u = units.copy()
    u["fc"], u["ll"] = idx.fc, idx.ll
    u["easting"], u["northing"] = idx.coords[:, 0], idx.coords[:, 1]
    u.to_csv(out / "units.csv", index=False)
    pd.Series(dates).to_csv(out / "dates.csv", index=False, header=["date"])
    weather.to_csv(out / "weather.csv", index=False)
    stations.to_csv(out / "stations.csv", index=False)

    # Layer 3 tiers
    flags_long = []
    for i, (s, p) in enumerate(units[["station", "port"]].itertuples(index=False)):
        flags_long.append(pd.DataFrame({"station": s, "port": p, "date": dates, "qc": F_on[i]}))
    flags_long = pd.concat(flags_long, ignore_index=True)
    warm = write_warm_tier(full, flags_long, out / "warm")
    warm.to_csv(out / "warm_catalog.csv", index=False)
    summ = weekly_summary(units, dates, X, F_on, weather)
    scat = write_summary_tier(summ, out / "summary")
    scat.to_csv(out / "summary_catalog.csv", index=False)

    stats = {
        "n_units": int(len(units)), "n_stations": int(units.station.nunique()), "n_days": int(len(dates)),
        "first_day": str(dates[0].date()), "last_day": str(dates[-1].date()),
        "readings_present": int((F_on >= 0).sum()), "readings_missing": int((F_on < 0).sum()),
        "missing_fraction": float((F_on < 0).mean()),
        "online_suspect": int((F_on == 1).sum()), "offline_rejected": int((F_off == 2).sum()),
        "warm_partitions": int(len(warm)), "warm_bytes": int(warm["bytes"].sum()),
        "summary_partitions": int(len(scat)), "summary_bytes": int(scat["bytes"].sum()),
        "weather_days": int(len(weather)),
    }
    (out / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


@dataclass
class Bundle:
    units: pd.DataFrame
    dates: pd.DatetimeIndex
    X: np.ndarray
    F_on: np.ndarray
    F_off: np.ndarray
    weather: pd.DataFrame
    stations: pd.DataFrame
    warm: pd.DataFrame
    summary: pd.DataFrame
    ctx: TriageContext

    @property
    def ports(self):
        return self.units["port"].to_numpy()

    @property
    def root_units(self):
        return np.where(self.ports <= 2)[0]


def load(out: Path) -> Bundle:
    units = pd.read_csv(out / "units.csv")
    dates = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(out / "dates.csv")["date"]))
    X = np.load(out / "X.npy")
    F_on = np.load(out / "F_on.npy")
    F_off = np.load(out / "F_off.npy")
    weather = pd.read_csv(out / "weather.csv", parse_dates=["date"])
    stations = pd.read_csv(out / "stations.csv")
    warm = pd.read_csv(out / "warm_catalog.csv")
    summ = pd.read_csv(out / "summary_catalog.csv")
    stale = StalenessModel(np.load(out / "stale_sigma.npy"), np.zeros(1), np.load(out / "stale_clim.npy"))
    st_ids = {s: i for i, s in enumerate(sorted(units.station.unique()))}
    ust = units.station.map(st_ids).to_numpy()
    ctx = TriageContext(EvidenceStore(X, F_on), units.fc.to_numpy(), units.ll.to_numpy(),
                        units.port.to_numpy(), ust, units[["easting", "northing"]].to_numpy(float),
                        stale, dates.month.to_numpy(), X, F_on)
    return Bundle(units, dates, X, F_on, F_off, weather, stations, warm, summ, ctx)
