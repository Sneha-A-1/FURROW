"""Layer 3: metadata index built from the calibration year only.

Contents of the index (all estimated from calibration-period data):

* unit thresholds  FC_u (upper reference, P95) and LL_u (lower reference, P2)
  of quality-controlled volumetric water content;
* station coordinates for spatial baselines (IDW).

Two further index components (a weather-driven drift model and a correlation-indexed
surrogate map) were evaluated in pilot experiments and rejected because they
increased nowcast error on the evaluation year; they are kept, with the pilot that
documents the negative result, in ``furrow.retrieval.rejected_variants``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .ingest import LATITUDE_DEG

FC_PCT, LL_PCT = 95.0, 2.0
SNOW_TMAX_C = 1.0       # precipitation on days with Tmax <= 1 C is treated as snow (no infiltration)
MIN_CAL_DAYS = 120      # minimum valid calibration days for unit-specific thresholds


def extraterrestrial_radiation(doy: np.ndarray, lat_deg: float = LATITUDE_DEG) -> np.ndarray:
    """FAO-56 Eq. 21, MJ m-2 day-1."""
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    gsc = 0.0820
    return (24 * 60 / np.pi) * gsc * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))


def hargreaves_et0(dates: pd.DatetimeIndex, tmax: np.ndarray, tmin: np.ndarray) -> np.ndarray:
    """Hargreaves-Samani reference ET (mm/day); Ra converted with 0.408 mm per MJ m-2."""
    ra = extraterrestrial_radiation(dates.dayofyear.to_numpy())
    tmean = (tmax + tmin) / 2
    trange = np.clip(tmax - tmin, 0, None)
    return np.clip(0.0023 * (tmean + 17.8) * np.sqrt(trange) * 0.408 * ra, 0, None)


@dataclass
class MetadataIndex:
    units: pd.DataFrame                  # station, port, unit_id
    fc: np.ndarray                       # (U,)
    ll: np.ndarray                       # (U,)
    coords: np.ndarray                   # (U, 2) easting, northing of the unit's station
    extras: dict = field(default_factory=dict)


def weather_drivers(dates: pd.DatetimeIndex, weather: pd.DataFrame):
    w = weather.set_index("date").reindex(dates)
    tmax = w["tmax_c"].interpolate().to_numpy()
    tmin = w["tmin_c"].interpolate().to_numpy()
    p = w["precip_mm"].fillna(0).to_numpy()
    et0 = hargreaves_et0(dates, tmax, tmin)
    p_eff = np.where(tmax > SNOW_TMAX_C, p, 0.0)
    return et0, p_eff, p


def _thresholds(Xc: np.ndarray, ports: np.ndarray):
    U = Xc.shape[0]
    fc = np.full(U, np.nan)
    ll = np.full(U, np.nan)
    pooled = {}
    for p in np.unique(ports):
        v = Xc[ports == p].ravel()
        v = v[~np.isnan(v)]
        pooled[p] = (np.percentile(v, FC_PCT), np.percentile(v, LL_PCT))
    for u in range(U):
        v = Xc[u][~np.isnan(Xc[u])]
        if len(v) >= MIN_CAL_DAYS:
            fc[u], ll[u] = np.percentile(v, FC_PCT), np.percentile(v, LL_PCT)
        else:
            fc[u], ll[u] = pooled[ports[u]]
        if fc[u] - ll[u] < 0.03:   # guard against degenerate ranges
            fc[u], ll[u] = pooled[ports[u]]
    return fc, ll


def build_index(units: pd.DataFrame, dates: pd.DatetimeIndex, X: np.ndarray, F_on: np.ndarray,
                weather: pd.DataFrame, stations: pd.DataFrame, cal_mask: np.ndarray) -> MetadataIndex:
    """Build the metadata index from calibration days ``cal_mask`` only."""
    Xc = np.where((F_on == 0) & cal_mask[None, :], X, np.nan)
    ports = units["port"].to_numpy()
    st = units["station"].to_numpy()
    fc, ll = _thresholds(Xc, ports)
    cmap = stations.set_index("station")[["easting", "northing"]]
    coords = cmap.reindex(st).to_numpy(float)
    return MetadataIndex(units=units, fc=fc, ll=ll, coords=coords)
