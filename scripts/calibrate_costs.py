"""Measure the real execution cost of every task type on the prepared data.

Outputs (results/calibration/):
  strategic_library.csv  one row per strategic query instance with CPU seconds,
                         bytes and partition counts of both fidelities and the
                         summary-answer error against the exact answer;
  verify_costs.csv       U2 verification task costs;
  triage_costs.csv       U1 triage CPU time per retrieval method;
  machine.json           hardware / software of the measurement host.

CPU time = wall-clock of Parquet read + decode + computation on the measurement
host (median of ``--reps`` repetitions, page cache warm). Network transfer is not
included here; it is modelled explicitly in the simulator.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from furrow.data.pipeline import load  # noqa: E402
from furrow.retrieval.strategic import TEMPLATES, QueryInstance, StrategicEngine, answer_error  # noqa: E402
from furrow.retrieval.triage import FurrowRetrieval, NoRefresh, AgeFirst, grade  # noqa: E402
from furrow.retrieval.verify import verify  # noqa: E402
from furrow.simulator.reporting import draw_schedule, station_horizon  # noqa: E402


def median_run(fn, reps):
    out, ts = None, []
    for _ in range(reps):
        out = fn()
        ts.append(out.cpu_s if hasattr(out, "cpu_s") else out[1])
    return out, float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-template", type=int, default=60)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(ROOT / "results/calibration"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    b = load(ROOT / "data/processed")
    fc = {(s, p): f for s, p, f in zip(b.units.station, b.units.port, b.units.fc)}
    ll = {(s, p): f for s, p, f in zip(b.units.station, b.units.port, b.units.ll)}
    eng = StrategicEngine(b.warm, b.summary, fc, ll)
    rng = np.random.default_rng(a.seed)
    stations = sorted(b.units.station.unique())
    days12 = b.dates[b.dates.year == 2012]

    rows = []
    for tpl in TEMPLATES:
        n_ok = 0
        while n_ok < a.per_template:
            t_end = pd.Timestamp(rng.choice(days12[30:]))
            W = int(rng.choice([30, 90, 180]))
            if tpl == "S5_analogues":
                S = stations
                tgt = (str(rng.choice(stations)), int(rng.choice([1, 2])))
            else:
                S = list(rng.choice(stations, size=int(rng.choice([10, 20, 42])), replace=False))
                tgt = None
            q = QueryInstance(tpl, t_end, W, S, tgt)
            ef, cf = median_run(lambda: eng.run(q, "full", b.weather), a.reps)
            es, cs = median_run(lambda: eng.run(q, "summary", b.weather), a.reps)
            err = answer_error(tpl, ef.answer, es.answer)
            if err is None or np.isnan(err):
                continue
            rows.append(dict(qid=len(rows), template=tpl, t_end=t_end.date(), window=W, n_stations=len(S),
                             target=str(tgt), cpu_full=cf, bytes_full=ef.bytes, parts_full=ef.n_parts,
                             cpu_summary=cs, bytes_summary=es.bytes, parts_summary=es.n_parts, error_summary=err))
            n_ok += 1
        print(tpl, "done", flush=True)
    lib = pd.DataFrame(rows)
    lib.to_csv(out / "strategic_library.csv", index=False)

    vrows = []
    root = b.root_units
    for i in range(200):
        u = int(rng.choice(root))
        t_end = pd.Timestamp(rng.choice(days12[30:]))
        r = b.units.iloc[u]
        (payload, _, nb, npart), c = median_run(lambda: verify(b.warm, r.station, int(r.port), t_end, r.fc, r.ll), a.reps)
        vrows.append(dict(unit=u, t_end=t_end.date(), cpu=c, bytes=nb, parts=npart))
    pd.DataFrame(vrows).to_csv(out / "verify_costs.csv", index=False)

    trows = []
    methods = [(NoRefresh(), 0), (AgeFirst(), 4), (FurrowRetrieval(), 4), (FurrowRetrieval(), 8)]
    P, ph = draw_schedule(42, "mixed", rng)
    ust = b.ctx.unit_station
    for t in np.where(b.dates.year == 2012)[0][::3]:
        hz = station_horizon(t, P, ph)[ust]
        for m, B in methods:
            t0 = time.perf_counter()
            for _ in range(a.reps):
                m(root, t, hz, b.ctx, budget=B, rng=np.random.default_rng(t))
            trows.append(dict(method=f"{m.name}@{B}", t=t, cpu=(time.perf_counter() - t0) / a.reps))
    pd.DataFrame(trows).to_csv(out / "triage_costs.csv", index=False)

    import pyarrow
    import scipy
    info = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                pyarrow=pyarrow.__version__, scipy=scipy.__version__, platform=platform.platform(),
                processor=platform.processor() or "unknown", cpu_count=__import__("os").cpu_count())
    try:
        info["cpu_model"] = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception:
        pass
    (out / "machine.json").write_text(json.dumps(info, indent=2))
    print(lib.groupby("template")[["cpu_full", "cpu_summary", "bytes_full", "bytes_summary", "error_summary"]].median())


if __name__ == "__main__":
    main()
