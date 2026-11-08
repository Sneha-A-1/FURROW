"""Layer 1-2: acquisition and parsing of the Cook Agronomy Farm (CAF) data set.

Source
------
The CAF sensor-network data set (Gasch et al., 2015, Spatial Statistics 14:70-90,
doi:10.1016/j.spasta.2015.04.001; Gasch et al., 2017, Water Resources Research
53:10878-10887, doi:10.1002/2017WR021307) is redistributed as the ``cookfarm``
object of the CRAN package ``landmap`` (GPL-3). The raw ``.rda`` file is read
from the read-only CRAN GitHub mirror.

The functions here only parse and reshape; no values are modified.
"""
from __future__ import annotations

import hashlib
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

CAF_URL = "https://raw.githubusercontent.com/cran/landmap/master/data/cookfarm.rda"
CAF_SHA256 = "59ac08cebff116aceb74a970529fdf066ef13608eb182a630010bfd87b768669"  # landmap 0.0.13 CRAN mirror
DEPTHS_M = {1: 0.3, 2: 0.6, 3: 0.9, 4: 1.2, 5: 1.5}
LATITUDE_DEG = 46.78  # R.J. Cook Agronomy Farm, Pullman WA (UTM 11N northing ~5.18e6 m)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def download(raw_dir: Path, url: str = CAF_URL) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / "cookfarm.rda"
    if not out.exists():
        urllib.request.urlretrieve(url, out)
    return out


def load_cookfarm(rda_path: Path) -> dict:
    """Parse the R list object into pandas data frames."""
    import rdata  # imported lazily: only needed for raw parsing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = rdata.read_rda(str(rda_path))["cookfarm"]
    return obj


def to_long(cf: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (readings_long, weather, stations).

    readings_long columns: station, date, port, depth_m, vw, temp_c, ec
    weather columns: date, precip_mm, tmax_c, tmin_c
    stations columns: station, easting, northing, soil_taxon, dem, twi, ndre_mean,
                      crop_2011, crop_2012
    """
    r = cf["readings"].copy()
    r["date"] = pd.to_datetime(r["Date"].astype(float), unit="D")
    r["station"] = r["SOURCEID"].astype(str)
    frames = []
    for p in range(1, 6):
        f = pd.DataFrame(
            {
                "station": r["station"].values,
                "date": r["date"].values,
                "port": p,
                "depth_m": DEPTHS_M[p],
                "vw": r[f"Port{p}VW"].astype(float).values,
                "temp_c": r[f"Port{p}C"].astype(float).values,
                "ec": r[f"Port{p}EC"].astype(float).values,
            }
        )
        frames.append(f)
    long = pd.concat(frames, ignore_index=True)
    long = long.sort_values(["station", "port", "date"]).reset_index(drop=True)

    w = cf["weather"].copy()
    weather = pd.DataFrame(
        {
            "date": pd.to_datetime(w["Date"].astype(float), unit="D"),
            "precip_mm": w["Precip_wrcc"].astype(float).values,
            "tmax_c": w["MaxT_wrcc"].astype(float).values,
            "tmin_c": w["MinT_wrcc"].astype(float).values,
        }
    ).sort_values("date").reset_index(drop=True)

    prof = cf["profiles"].copy()
    prof["station"] = prof["SOURCEID"].astype(str)
    first = prof.groupby("station", observed=True).first().reset_index()
    sensor_ids = sorted(long["station"].unique())
    st = first[first["station"].isin(sensor_ids)][["station", "Easting", "Northing", "TAXSUSDA"]]
    st = st.rename(columns={"Easting": "easting", "Northing": "northing", "TAXSUSDA": "soil_taxon"})
    st["soil_taxon"] = st["soil_taxon"].astype(str)

    # nearest 10 m grid cell for topographic / crop covariates
    g = cf["grids"].copy()
    gx = g["x"].to_numpy(float)
    gy = g["y"].to_numpy(float)
    rows = []
    for _, s in st.iterrows():
        k = int(np.argmin((gx - s.easting) ** 2 + (gy - s.northing) ** 2))
        rows.append(
            {
                "station": s.station,
                "dem": float(g["DEM"].iloc[k]),
                "twi": float(g["TWI"].iloc[k]),
                "ndre_mean": float(g["NDRE.M"].iloc[k]),
                "crop_2011": str(g["X2011"].iloc[k]),
                "crop_2012": str(g["X2012"].iloc[k]),
                "grid_dist_m": float(np.sqrt((gx[k] - s.easting) ** 2 + (gy[k] - s.northing) ** 2)),
            }
        )
    stations = st.merge(pd.DataFrame(rows), on="station").sort_values("station").reset_index(drop=True)
    return long, weather, stations


def complete_grid(long: pd.DataFrame) -> pd.DataFrame:
    """Expand to the full station x port x date grid; absent rows become NaN."""
    dates = pd.date_range(long["date"].min(), long["date"].max(), freq="D")
    idx = pd.MultiIndex.from_product(
        [sorted(long["station"].unique()), range(1, 6), dates], names=["station", "port", "date"]
    )
    full = long.set_index(["station", "port", "date"]).reindex(idx).reset_index()
    full["depth_m"] = full["port"].map(DEPTHS_M)
    return full
