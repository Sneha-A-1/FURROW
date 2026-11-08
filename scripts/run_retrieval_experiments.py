"""Experiment A: urgent-triage retrieval quality on the 2012 evaluation year.

For every connectivity scenario, seed (station reporting schedule draw) and 2012
decision day with at least one relevant unit, every method ranks the root-zone
units and raises alerts. Ground truth = offline-QC reading of the decision day.

Output: results/raw/retrieval_decisions.csv.gz (one row per decision x method)
        results/processed/retrieval_summary.csv (mean over decisions per seed, then over seeds)
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from furrow.data.pipeline import load  # noqa: E402
from furrow.evaluation.metrics import alert_prf, mrr, ndcg_at_k, precision_at_k, r_precision  # noqa: E402
from furrow.retrieval.triage import (AgeFirst, FreshnessFiltered, FurrowRetrieval, IDWNowcast,  # noqa: E402
                                     NoRefresh, PollAll, RandomRefresh, alerts_from, grade)
from furrow.simulator.reporting import draw_schedule, station_horizon  # noqa: E402

BUNDLE = None


def methods():
    m = [(NoRefresh(), 0), (FreshnessFiltered(), 0), (IDWNowcast(), 0), (PollAll(), 0),
         (FurrowRetrieval(), 0)]
    for B in (2, 4, 8, 16):
        m += [(AgeFirst(), B), (FurrowRetrieval(), B)]
    for B in (4, 8):
        m += [(RandomRefresh(), B)]
    m += [(FurrowRetrieval(expected_gain=False, name="FURROW-R(point)"), 4)]
    for v in (0.0, 0.02, 0.1, 0.2):          # v_min sensitivity at B = 8 (0.05 is the default above)
        m += [(FurrowRetrieval(voi_min=v, name=f"FURROW-R(vmin={v})"), 8)]
    return m


def _init():
    global BUNDLE
    BUNDLE = load(ROOT / "data/processed")


def run_seed(args):
    scen, seed = args
    b = BUNDLE
    ctx = b.ctx
    root = b.root_units
    rng = np.random.default_rng(seed)
    P, ph = draw_schedule(int(ctx.unit_station.max()) + 1, scen, rng)
    days = np.where(b.dates.year == 2012)[0]
    rows = []
    ms = methods()
    for t in days:
        hz = station_horizon(t, P, ph)[ctx.unit_station]
        truth = np.where(b.F_off[:, t] == 0, b.X[:, t], np.nan)
        cand = root[~np.isnan(truth[root])]
        if len(cand) < 10:
            continue
        D = (ctx.fc[cand] - truth[cand]) / (ctx.fc[cand] - ctx.ll[cand])
        rel = grade(D)
        if (rel >= 1).sum() == 0:
            continue
        li, _ = ctx.store.last(hz, t)
        stale_frac = float(np.mean(t - li[cand] > 0))
        for m, B in ms:
            t0 = time.perf_counter()
            r = m(cand, t, hz, ctx, budget=B, rng=np.random.default_rng(seed * 100_003 + t))
            cpu = time.perf_counter() - t0
            erng = np.random.default_rng(seed * 7_919 + t)
            al = np.isin(cand, alerts_from(r, 10_000))
            p, rc, tp, fp, fn = alert_prf(al, rel)
            rows.append(dict(scenario=scen, seed=seed, day=int(t), month=int(b.dates[t].month),
                             method=m.name, budget=B, label=f"{m.name}@{B}" if B else m.name,
                             ndcg10=ndcg_at_k(r.score, rel, 10, erng), p10=precision_at_k(r.score, rel, 10, erng),
                             rprec=r_precision(r.score, rel, erng), mrr3=mrr(r.score, rel, erng),
                             alert_p=p, alert_r=rc, tp=tp, fp=fp, fn=fn, errors=fp + fn,
                             n_rel=int((rel >= 1).sum()), n_pos=int((rel >= 2).sum()), n_cand=len(cand),
                             polls=len(r.polled_stations), bytes=r.bytes_read, cpu_s=cpu, stale_frac=stale_frac))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--scenarios", default="daily,mixed,poor")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    jobs = [(s, k) for s in a.scenarios.split(",") for k in range(a.seeds)]
    with Pool(a.procs, initializer=_init) as pool:
        out = []
        for i, rows in enumerate(pool.imap_unordered(run_seed, jobs)):
            out.extend(rows)
            print(f"{i + 1}/{len(jobs)}", flush=True)
    df = pd.DataFrame(out)
    (ROOT / "results/raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "results/raw/retrieval_decisions.csv.gz", index=False, compression="gzip")
    cols = ["ndcg10", "p10", "rprec", "mrr3", "alert_p", "alert_r", "fp", "fn", "errors", "polls", "bytes", "cpu_s"]
    per_seed = df.groupby(["scenario", "label", "seed"])[cols].mean().reset_index()
    per_seed.to_csv(ROOT / "results/processed/retrieval_per_seed.csv", index=False)
    summ = per_seed.groupby(["scenario", "label"])[cols].mean().reset_index()
    summ.to_csv(ROOT / "results/processed/retrieval_summary.csv", index=False)
    print(summ.to_string())


if __name__ == "__main__":
    main()
