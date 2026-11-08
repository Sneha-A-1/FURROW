"""Layer 4 (urgent path): FURROW-R urgency triage with value-of-information refresh.

Problem
-------
At decision day t the platform returns, for a farm, a ranked list of root-zone
sensing units u = (station, depth) ordered by agronomic urgency, measured by the
soil-water depletion fraction

    D_u(t) = (FC_u - x_u(t)) / (FC_u - LL_u),

graded g = 0 (D < 0.5), 1 (0.5 <= D < 0.7), 2 (0.7 <= D < 0.85), 3 (D >= 0.85),
and raises a verification task for every unit with grade >= 2.

The cloud copy of each station is stale by a_u days because stations deliver
readings every P_s days. A station can be refreshed on demand (an edge poll),
which costs radio latency and sensor energy. Given a polling budget B (stations),
FURROW-R chooses which stations to refresh:

  prior     x_u(t) ~ N(x_last_u, sigma^2_{p,m}(a_u))   (persistence prior; sigma from
            the calibration year, per depth p, month m and age a)
  VoI_u   = sum_k  (2^k - 2^(k-1)) * Phi(-|D_hat_u - tau_k| / sigma_D,u)
            (gain-weighted probability that refreshing flips a grade boundary)
  VoI_s   = sum_{u in s} VoI_u ;   poll the B stations with largest VoI_s / c_s,
            skipping stations with VoI_s < v_min
  rank    by posterior expected gain  E[2^g - 1] = sum_k (2^k - 2^(k-1)) P(g >= k)

Ranking by expected gain maximises expected DCG for a fixed gain vector.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import ndtr

TAU = np.array([0.5, 0.7, 0.85])
GAIN_INC = np.array([1.0, 2.0, 4.0])      # (2^k - 2^(k-1)) for k = 1..3
LOOKBACK_DAYS = 60
MAX_AGE_BIN = 14
BYTES_PER_READING = 16                     # timestamp 8 + value 4 + flag 4
BYTES_PER_STATION_POLL = 5 * 3 * 4 + 64    # 5 depths x 3 variables x float32 + header


def grade(D: np.ndarray) -> np.ndarray:
    g = np.zeros(np.shape(D), dtype=np.int8)
    for t in TAU:
        g = g + (np.asarray(D) >= t)
    return g.astype(np.int8)


# ----------------------------------------------------------------------------------
# evidence store: forward-filled pointers to the latest delivered reading
# ----------------------------------------------------------------------------------
class EvidenceStore:
    def __init__(self, X: np.ndarray, F_on: np.ndarray):
        self.X = X
        self.F = F_on
        present = F_on >= 0                      # any delivered reading (QC pass or suspect)
        U, T = X.shape
        idx = np.where(present, np.arange(T)[None, :], -1)
        self.ffill = np.maximum.accumulate(idx, axis=1)

    def last(self, horizon: np.ndarray, t: int, lookback: int = LOOKBACK_DAYS):
        """Latest delivered reading per unit given delivery horizons (day indices)."""
        U = len(horizon)
        rows = np.arange(U)
        h = np.clip(horizon, 0, None)
        li = np.where(horizon >= 0, self.ffill[rows, h], -1)
        li = np.where((li >= 0) & (t - li <= lookback), li, -1)
        lv = np.where(li >= 0, self.X[rows, np.clip(li, 0, None)], np.nan)
        return li, lv


# ----------------------------------------------------------------------------------
# staleness uncertainty model (calibration year only)
# ----------------------------------------------------------------------------------
@dataclass
class StalenessModel:
    sigma: np.ndarray        # (5 ports, 12 months, MAX_AGE_BIN+2) sd of x(t)-x(t-a); last bin = >14 d
    n: np.ndarray
    clim: np.ndarray | None = None   # (5 ports, 12 months, 2) mean and sd of D (climatological prior)

    @classmethod
    def fit(cls, X: np.ndarray, F_on: np.ndarray, ports: np.ndarray, dates: pd.DatetimeIndex,
            cal_mask: np.ndarray, fc: np.ndarray | None = None, ll: np.ndarray | None = None) -> "StalenessModel":
        Xc = np.where((F_on >= 0), X, np.nan)
        months = dates.month.to_numpy() - 1
        A = MAX_AGE_BIN + 2
        sig = np.full((5, 12, A), np.nan)
        cnt = np.zeros((5, 12, A), int)
        for p in range(1, 6):
            Xp = Xc[ports == p]
            for a in range(1, A):
                lag = a if a <= MAX_AGE_BIN else 30
                d = Xp[:, lag:] - Xp[:, :-lag]
                tm = months[lag:]
                ok_t = cal_mask[lag:] & cal_mask[:-lag]
                for m in range(12):
                    sel = d[:, (tm == m) & ok_t].ravel()
                    sel = sel[~np.isnan(sel)]
                    cnt[p - 1, m, a] = len(sel)
                    if len(sel) >= 30:
                        sig[p - 1, m, a] = np.sqrt(np.mean(sel ** 2))   # RMS change (includes drift)
        # fill sparse cells: pool adjacent months, then enforce monotone growth in age
        for p in range(5):
            for a in range(1, A):
                col = sig[p, :, a]
                if np.isnan(col).any():
                    good = ~np.isnan(col)
                    if good.any():
                        mm = np.arange(12)
                        col[~good] = np.interp(mm[~good], mm[good], col[good], period=12)
                sig[p, :, a] = col
            sig[p, :, 1:] = np.fmax.accumulate(np.nan_to_num(sig[p, :, 1:], nan=0.0), axis=1)
        sig[:, :, 0] = 0.0
        sig = np.maximum(sig, 1e-4)
        sig[:, :, 0] = 0.0
        clim = None
        if fc is not None:
            D = (fc[:, None] - Xc) / (fc - ll)[:, None]
            clim = np.zeros((5, 12, 2))
            for p in range(1, 6):
                for m in range(12):
                    v = D[ports == p][:, (months == m) & cal_mask].ravel()
                    v = v[~np.isnan(v)]
                    clim[p - 1, m] = (np.mean(v), np.std(v)) if len(v) > 30 else (0.5, 0.5)
        return cls(sig, cnt, clim)

    def climatology(self, ports: np.ndarray, month: int):
        if self.clim is None:
            return np.full(len(ports), np.nan), np.full(len(ports), np.nan)
        c = self.clim[ports - 1, month - 1]
        return c[:, 0], c[:, 1]

    def sd(self, ports: np.ndarray, month: int, age: np.ndarray) -> np.ndarray:
        a = np.where(age > MAX_AGE_BIN, MAX_AGE_BIN + 1, np.clip(age, 0, None)).astype(int)
        return self.sigma[ports - 1, month - 1, a]


# ----------------------------------------------------------------------------------
# triage methods
# ----------------------------------------------------------------------------------
@dataclass
class TriageContext:
    store: EvidenceStore
    fc: np.ndarray
    ll: np.ndarray
    ports: np.ndarray
    unit_station: np.ndarray          # station index of every unit
    coords: np.ndarray                # (U, 2)
    stale_model: StalenessModel
    month_of_day: np.ndarray          # month (1..12) of every day index
    truth_X: np.ndarray               # readings the station holds locally (revealed by a poll)
    truth_F: np.ndarray


@dataclass
class TriageResult:
    unit_ids: np.ndarray
    score: np.ndarray
    d_hat: np.ndarray
    sd_D: np.ndarray
    p_warn: np.ndarray
    polled_stations: np.ndarray = field(default_factory=lambda: np.array([], int))
    bytes_read: int = 0
    prior_expected_alerts: float = 0.0


def _prior(cand, t, horizon_all, ctx: TriageContext, climatology: bool = False):
    """Persistence prior of the depletion fraction; with ``climatology`` a unit without
    any delivered reading in the look-back window receives the calibration-year
    climatological prior of its depth and month instead of being unrankable."""
    li, lv = ctx.store.last(horizon_all, t)
    age = np.where(li >= 0, t - li, LOOKBACK_DAYS + 1)
    rng = ctx.fc - ctx.ll
    D = (ctx.fc - lv) / rng
    sd = ctx.stale_model.sd(ctx.ports, int(ctx.month_of_day[t]), age) / rng
    if climatology:
        cm, cs = ctx.stale_model.climatology(ctx.ports, int(ctx.month_of_day[t]))
        none = np.isnan(D) & ~np.isnan(cm)
        D = np.where(none, cm, D)
        sd = np.where(none, cs, sd)
    return D[cand], sd[cand], age[cand], li


def _expected_gain(D, sd):
    z = (D[:, None] - TAU[None, :]) / np.maximum(sd[:, None], 1e-9)
    pk = np.where(sd[:, None] > 0, ndtr(z), (D[:, None] >= TAU[None, :]).astype(float))
    return pk @ GAIN_INC, pk


def _voi(D, sd):
    """Gain-weighted probability that a refresh changes at least one grade boundary."""
    z = np.abs(D[:, None] - TAU[None, :]) / np.maximum(sd[:, None], 1e-9)
    flip = np.where(sd[:, None] > 0, ndtr(-z), 0.0)
    return np.nan_to_num(flip @ GAIN_INC)


def _refresh(cand, t, D, sd, polled_st, ctx: TriageContext):
    """Apply polls: polled stations reveal their current local reading (if the sensor has one)."""
    D = D.copy()
    sd = sd.copy()
    if len(polled_st) == 0:
        return D, sd
    hit = np.isin(ctx.unit_station[cand], polled_st)
    xv = ctx.truth_X[cand, t]
    fv = ctx.truth_F[cand, t]
    ok = hit & (fv >= 0) & ~np.isnan(xv)
    D[ok] = (ctx.fc[cand][ok] - xv[ok]) / (ctx.fc[cand][ok] - ctx.ll[cand][ok])
    sd[ok] = 0.0
    return D, sd


class TriageMethod:
    name = "base"
    polls = True
    climatology = True       # refresh-family methods share the Bayesian prior

    def select(self, cand, t, D, sd, age, ctx, budget, rng) -> np.ndarray:
        return np.array([], int)

    def rank_score(self, D, sd):
        eg, pk = _expected_gain(np.nan_to_num(D, nan=-9.0), np.nan_to_num(sd, nan=0.0))
        return np.where(np.isnan(D), -np.inf, eg + 1e-6 * np.nan_to_num(D)), pk

    def __call__(self, cand, t, horizon_all, ctx: TriageContext, budget: int = 0,
                 rng: np.random.Generator | None = None) -> TriageResult:
        rng = rng or np.random.default_rng(0)
        D0, sd0, age, _ = _prior(cand, t, horizon_all, ctx, self.climatology)
        _, pk0 = _expected_gain(np.nan_to_num(D0, nan=-9.0), np.nan_to_num(sd0, nan=0.0))
        prior_alerts = float(np.sum(np.where(np.isnan(D0), 0.0, pk0[:, 1])))
        polled = self.select(cand, t, D0, sd0, age, ctx, budget, rng)
        D, sd = _refresh(cand, t, D0, sd0, polled, ctx)
        score, pk = self.rank_score(D, sd)
        p_warn = np.where(np.isnan(D), 0.0, pk[:, 1])
        nbytes = len(cand) * BYTES_PER_READING + len(polled) * BYTES_PER_STATION_POLL
        res = TriageResult(cand, score, D, sd, p_warn, np.asarray(polled, int), int(nbytes), prior_alerts)
        res.prior_D, res.prior_sd = D0, sd0
        return res


class NoRefresh(TriageMethod):
    """THR: conventional indexed threshold retrieval on the latest delivered reading
    (point ranking by depletion, no refresh)."""
    name = "THR"
    climatology = False

    def rank_score(self, D, sd):
        pk = (np.nan_to_num(D, nan=-9.0)[:, None] >= TAU[None, :]).astype(float)
        return np.where(np.isnan(D), -np.inf, D), pk


class FreshnessFiltered(NoRefresh):
    """FRESH: threshold retrieval restricted to readings no older than max_age days."""
    name = "FRESH"

    def __init__(self, max_age: int = 3):
        self.max_age = max_age

    def __call__(self, cand, t, horizon_all, ctx, budget=0, rng=None):
        res = super().__call__(cand, t, horizon_all, ctx, 0, rng)
        _, _, age, _ = _prior(cand, t, horizon_all, ctx)
        stale = age > self.max_age
        res.score = np.where(stale, -np.inf, res.score)
        res.p_warn = np.where(stale, 0.0, res.p_warn)
        res.d_hat = np.where(stale, np.nan, res.d_hat)
        return res


class IDWNowcast(NoRefresh):
    """IDW: units older than max_age are replaced by an inverse-distance-weighted
    estimate (power 2, k nearest) of fresh same-depth units' depletion."""
    name = "IDW"

    def __init__(self, max_age: int = 3, k: int = 5, power: float = 2.0):
        self.max_age, self.k, self.power = max_age, k, power

    def __call__(self, cand, t, horizon_all, ctx, budget=0, rng=None):
        D, _, age, _ = _prior(cand, t, horizon_all, ctx)
        fresh = (age <= self.max_age) & ~np.isnan(D)
        Dn = D.copy()
        for j in np.where(~fresh)[0]:
            nb = np.where(fresh & (ctx.ports[cand] == ctx.ports[cand[j]]))[0]
            if len(nb) == 0:
                continue
            dist = np.maximum(np.linalg.norm(ctx.coords[cand[nb]] - ctx.coords[cand[j]], axis=1), 1.0)
            o = np.argsort(dist)[: self.k]
            wts = 1.0 / dist[o] ** self.power
            Dn[j] = float(np.sum(wts * D[nb[o]]) / np.sum(wts))
        score, pk = self.rank_score(Dn, np.zeros_like(Dn))
        p_warn = np.where(np.isnan(Dn), 0.0, pk[:, 1])
        res = TriageResult(cand, score, Dn, np.zeros_like(Dn), p_warn, np.array([], int),
                           int(len(cand) * BYTES_PER_READING * 2), 0.0)
        res.prior_D, res.prior_sd = Dn, np.zeros_like(Dn)
        return res


class PollAll(TriageMethod):
    """ALL: refresh every stale station (upper bound on accuracy, maximal polling cost)."""
    name = "ALL"

    def select(self, cand, t, D, sd, age, ctx, budget, rng):
        st = ctx.unit_station[cand]
        return np.unique(st[age > 0])


class AgeFirst(TriageMethod):
    """AoI: refresh the B stations with the oldest delivered data (age-of-information first)."""
    name = "AoI"

    def select(self, cand, t, D, sd, age, ctx, budget, rng):
        if budget <= 0:
            return np.array([], int)
        st = ctx.unit_station[cand]
        df = pd.DataFrame({"s": st, "a": age}).groupby("s")["a"].max()
        df = df[df > 0]
        jitter = rng.random(len(df)) * 1e-3
        order = np.argsort(-(df.to_numpy() + jitter))
        return df.index.to_numpy()[order[:budget]]


class RandomRefresh(TriageMethod):
    name = "RND"

    def select(self, cand, t, D, sd, age, ctx, budget, rng):
        st = np.unique(ctx.unit_station[cand][age > 0])
        if budget <= 0 or len(st) == 0:
            return np.array([], int)
        return rng.choice(st, size=min(budget, len(st)), replace=False)


class FurrowRetrieval(TriageMethod):
    """FURROW-R: value-of-information refresh + expected-gain ranking.

    voi_min: stations whose VoI is below this value are not polled even if budget
    remains (adaptive polling count). expected_gain=False ranks by the point
    estimate (ablation)."""
    name = "FURROW-R"

    def __init__(self, voi_min: float = 0.05, expected_gain: bool = True, name: str | None = None):
        self.voi_min = voi_min
        self.expected_gain = expected_gain
        if name:
            self.name = name

    def select(self, cand, t, D, sd, age, ctx, budget, rng):
        if budget <= 0:
            return np.array([], int)
        v = _voi(np.nan_to_num(D, nan=-9.0), np.nan_to_num(sd, nan=0.0))
        st = ctx.unit_station[cand]
        s_ids, inv = np.unique(st, return_inverse=True)
        vs = np.bincount(inv, weights=v, minlength=len(s_ids))
        order = np.argsort(-vs, kind="stable")
        chosen = [s_ids[i] for i in order[:budget] if vs[i] >= self.voi_min]
        return np.array(chosen, int)

    def rank_score(self, D, sd):
        if self.expected_gain:
            return super().rank_score(D, sd)
        pk = (np.nan_to_num(D, nan=-9.0)[:, None] >= TAU[None, :]).astype(float)
        return np.where(np.isnan(D), -np.inf, D), pk


def alerts_from(res: TriageResult, k_cap: int, alert_prob: float = 0.5) -> np.ndarray:
    """Units for which a verification (U2) task is raised: P(g>=2) >= alert_prob,
    at most k_cap per triage, in ranking order."""
    order = np.argsort(-res.score, kind="stable")
    sel = [i for i in order if np.isfinite(res.score[i]) and res.p_warn[i] >= alert_prob]
    return res.unit_ids[np.array(sel[:k_cap], dtype=int)] if sel else np.array([], dtype=int)
