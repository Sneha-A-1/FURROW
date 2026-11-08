"""Run a simulation experiment profile from config/experiment_profiles.yaml.

Usage: python scripts/run_simulation_grid.py PROFILE [--seeds N] [--procs P]
Output: results/raw/<PROFILE>.csv (one row per run), and for the profile's
        'keep_tasks' configuration, results/raw/<PROFILE>_tasks_<label>.csv.gz
"""
import argparse
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from furrow.data.pipeline import load  # noqa: E402
from furrow.simulator.run import Library, simulate  # noqa: E402
from furrow.utils.config import build, load_default  # noqa: E402

G = {}


def _init():
    G["bundle"] = load(ROOT / "data/processed")
    G["lib"] = Library(ROOT / "results/calibration")
    G["cfg"] = load_default()
    G["c_eval"] = json.loads((ROOT / "results/calibration/c_eval.json").read_text())["c_eval_s"]


def expand(profile: dict, seeds: int):
    jobs = []
    methods = profile["methods"]
    if "sweep" in profile:
        keys = list(profile["sweep"].keys())
        for vals in itertools.product(*[profile["sweep"][k] for k in keys]):
            ov = dict(zip(keys, vals))
            for label, sched, retr, extra in methods:
                for s in range(seeds):
                    jobs.append(dict(factor="grid", level=json.dumps(ov), label=label, scheduler=sched,
                                     retrieval=retr, overrides={**ov, **extra, "retrieval": retr}, seed=s))
    base = profile.get("base", {})
    for fac, levels in profile.get("factors", {}).items():
        for lv in levels:
            for label, sched, retr, extra in methods:
                for s in range(seeds):
                    ov = {**base, fac: lv}
                    jobs.append(dict(factor=fac, level=str(lv), label=label, scheduler=sched, retrieval=retr,
                                     overrides={**ov, **extra, "retrieval": retr}, seed=s))
    for sc in profile.get("scale", []):
        for label, sched, retr, extra in methods:
            for s in range(seeds):
                ov = {**base, **sc}
                jobs.append(dict(factor="scale", level=str(sc["n_farms"]), label=label, scheduler=sched,
                                 retrieval=retr, overrides={**ov, **extra, "retrieval": retr}, seed=s))
    return jobs


def run_job(job):
    scen, plat, fk = build(G["cfg"], {k: v for k, v in job["overrides"].items()})
    t0 = time.perf_counter()
    res = simulate(G["bundle"], G["lib"], scen, plat, job["scheduler"], job["seed"], G["c_eval"],
                   sched_kwargs=fk if job["scheduler"].startswith("FURROW") else None)
    row = dict(res.summary)
    row.update(factor=job["factor"], level=job["level"], label=job["label"], wall_s=time.perf_counter() - t0,
               overrides=json.dumps(job["overrides"]))
    tasks = None
    if job.get("keep"):
        tasks = res.tasks
        tasks["label"] = job["label"]
    return row, tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    profiles = yaml.safe_load(open(ROOT / "config/experiment_profiles.yaml"))
    prof = profiles[a.profile]
    seeds = a.seeds or prof["seeds"]
    jobs = expand(prof, seeds)
    keep = prof.get("keep_tasks")
    if keep:
        for j in jobs:
            if j["seed"] == keep["seed"] and all(j["overrides"].get(k) == v for k, v in keep.items() if k != "seed"):
                j["keep"] = True
    out = ROOT / "results/raw"
    out.mkdir(parents=True, exist_ok=True)
    rows, task_frames = [], []
    t0 = time.time()
    with Pool(a.procs, initializer=_init) as pool:
        for i, (row, tasks) in enumerate(pool.imap_unordered(run_job, jobs, chunksize=1)):
            rows.append(row)
            if tasks is not None:
                task_frames.append(tasks)
            if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                print(f"{a.profile}: {i + 1}/{len(jobs)} runs, {time.time() - t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(out / f"{a.profile}.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"{a.profile}.csv", index=False)
    if task_frames:
        pd.concat(task_frames).to_csv(out / f"{a.profile}_tasks.csv.gz", index=False, compression="gzip")


if __name__ == "__main__":
    main()
