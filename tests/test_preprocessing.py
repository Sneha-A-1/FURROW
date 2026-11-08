import numpy as np
import pandas as pd

from furrow.data.qc import offline_qc, online_qc
from furrow.data.store import weekly_summary


def test_online_qc_range_spike_rain_flatline():
    vw = np.array([0.30, 0.30, 0.30, 0.40, 0.31, np.nan, 0.70])
    rain = np.zeros_like(vw)
    f = online_qc(vw, port=1, rain2d=rain)
    assert f[0] == 0 and f[3] == 1          # +0.10 without rain -> suspect
    assert f[5] == -1                       # missing
    assert f[6] == 2                        # out of physical range
    rain[3] = 10.0
    f2 = online_qc(vw, port=1, rain2d=rain)
    assert f2[3] == 0                       # rise explained by rain at a shallow port
    f3 = online_qc(vw, port=4, rain2d=rain)
    assert f3[3] == 1                       # deep port: rain does not explain a jump
    flat = np.full(12, 0.25)
    ff = online_qc(flat, port=1, rain2d=np.zeros(12))
    assert ff[8] == 0 and ff[9] == 1        # 10th identical reading -> stuck sensor


def test_offline_qc_two_sided():
    vw = np.array([0.30, 0.30, 0.30, 0.45, 0.30, 0.30, 0.30])
    f = offline_qc(vw)
    assert f[3] == 2 and (f[[0, 1, 2, 4, 5, 6]] == 0).all()


def test_weekly_summary_values():
    dates = pd.date_range("2012-01-02", periods=14, freq="D")   # two Monday-start weeks
    units = pd.DataFrame({"station": ["A"], "port": [1]})
    X = np.arange(14, dtype=float)[None, :] / 100
    F = np.zeros_like(X, dtype=np.int8)
    F[0, 3] = -1                                                # one missing reading
    weather = pd.DataFrame({"date": dates, "precip_mm": 1.0, "tmax_c": 10.0, "tmin_c": 0.0})
    s = weekly_summary(units, dates, X, F, weather)
    assert len(s) == 2
    w1 = s.iloc[0]
    assert w1["n"] == 6
    assert np.isclose(w1["vw_mean"], np.mean([0, 1, 2, 4, 5, 6]) / 100)
    assert np.isclose(s.iloc[1]["vw_max"], 0.13)
    assert np.isclose(w1["precip_sum"], 7.0)
