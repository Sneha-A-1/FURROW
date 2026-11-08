import numpy as np
import pandas as pd
import pytest

from furrow.retrieval.strategic import answer_error
from furrow.retrieval.triage import (AgeFirst, EvidenceStore, FurrowRetrieval, NoRefresh, PollAll,
                                     StalenessModel, TriageContext, alerts_from, grade)


def test_grade_thresholds():
    assert list(grade(np.array([0.49, 0.5, 0.69, 0.7, 0.849, 0.85, 1.2]))) == [0, 1, 1, 2, 2, 3, 3]


def test_evidence_store_last_and_lookback():
    X = np.array([[0.1, np.nan, 0.3, np.nan, np.nan]])
    F = np.array([[0, -1, 0, -1, -1]], dtype=np.int8)
    st = EvidenceStore(X, F)
    li, lv = st.last(np.array([4]), t=4)
    assert li[0] == 2 and lv[0] == pytest.approx(0.3)
    li, lv = st.last(np.array([1]), t=4)          # delivery horizon day 1
    assert li[0] == 0 and lv[0] == pytest.approx(0.1)
    li, lv = st.last(np.array([4]), t=4, lookback=1)
    assert li[0] == -1 and np.isnan(lv[0])


def _toy_context():
    """3 stations x 1 root unit each, 10 days. Truth on day 9 known locally.
    s0: stale 5 days, prior D close to the 0.7 threshold
    s1: stale 8 days, prior D far from all thresholds (wet)
    s2: fresh"""
    T = 10
    X = np.full((3, T), np.nan)
    F = np.full((3, T), -1, dtype=np.int8)
    fc, ll = np.array([0.40, 0.40, 0.40]), np.array([0.10, 0.10, 0.10])
    X[0, 4] = 0.40 - 0.69 * 0.30       # D = 0.69 at day 4
    X[1, 1] = 0.40 - 0.05 * 0.30       # D = 0.05 at day 1
    X[2, 9] = 0.40 - 0.95 * 0.30       # fresh, D = 0.95
    X[0, 9] = 0.40 - 0.72 * 0.30       # local truth for s0 on day 9 (D=0.72)
    X[1, 9] = 0.40 - 0.06 * 0.30
    F[~np.isnan(X)] = 0
    delivered = F.copy()
    delivered[0, 9] = -1
    delivered[1, 9] = -1                   # not yet delivered to the cloud
    sig = np.zeros((5, 12, 16))
    sig[:, :, 1:] = 0.01 * np.sqrt(np.arange(1, 16))[None, None, :]
    ctx = TriageContext(EvidenceStore(np.where(delivered >= 0, X, np.nan), delivered), fc, ll,
                        np.array([1, 1, 1]), np.array([0, 1, 2]), np.zeros((3, 2)),
                        StalenessModel(sig, np.zeros(1)), np.full(T, 8), X, F)
    return ctx


def test_voi_prefers_threshold_proximity_over_age():
    ctx = _toy_context()
    cand = np.array([0, 1, 2])
    hz = np.array([9, 9, 9])
    r_f = FurrowRetrieval()(cand, 9, hz, ctx, budget=1)
    r_a = AgeFirst()(cand, 9, hz, ctx, budget=1, rng=np.random.default_rng(0))
    assert list(r_f.polled_stations) == [0]       # near the alert threshold
    assert list(r_a.polled_stations) == [1]       # oldest data
    # after refresh s0 is above the alert threshold (D = 0.72) -> alerted
    assert 0 in set(alerts_from(r_f, 10)) and 0 not in set(alerts_from(r_a, 10))


def test_threshold_and_pollall():
    ctx = _toy_context()
    cand = np.array([0, 1, 2])
    hz = np.array([9, 9, 9])
    thr = NoRefresh()(cand, 9, hz, ctx)
    assert thr.d_hat[0] == pytest.approx(0.69)
    pa = PollAll()(cand, 9, hz, ctx)
    assert set(pa.polled_stations) == {0, 1}
    assert pa.d_hat[0] == pytest.approx(0.72)


def test_voi_min_skips_useless_polls():
    ctx = _toy_context()
    r = FurrowRetrieval(voi_min=0.5)(np.array([1, 2]), 9, np.array([9, 9, 9]), ctx, budget=5)
    assert len(r.polled_stations) == 0


def test_answer_error_variants():
    assert answer_error("S5_analogues", [1, 2, 3, 4, 5], [1, 2, 9, 8, 7]) == pytest.approx(0.6)
    assert answer_error("S1_water_status", np.array([0.2, 0.4]), np.array([0.22, 0.4])) == pytest.approx(0.05)
    assert answer_error("S2_deficit_days", {"a": 10}, {"a": 12}) == pytest.approx(0.2)
    assert answer_error("S4_drying_rate", {"a": 0.0}, {"a": 0.0005}) == pytest.approx(1.0)
