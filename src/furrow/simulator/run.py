"""Trace-driven end-to-end simulation of the FURROW platform (all layers).

One run simulates the morning upload window of one agricultural day for a regional
platform serving ``n_farms`` farms. Each farm replays the real CAF sensor network
with a farm-specific day shift (time-shifted replay) and its own station reporting
schedule. Urgent chains are triggered by farm data uploads (SYNC events); their
triage, polling decisions, raised alerts and ground-truth grades are computed on
the real data at run time. Strategic queries are drawn from the measured query
library (results/calibration/strategic_library.csv).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..retrieval.triage import (AgeFirst, FurrowRetrieval, NoRefresh, PollAll, RandomRefresh,
                                FreshnessFiltered, IDWNowcast, alerts_from, grade)
from ..scheduling.baselines import BASELINES
from ..scheduling.furrow import FurrowScheduler
from .engine import Chain, Engine, Task
from .platform import PlatformConfig, Variant, build_cores, exec_time
from .reporting import draw_schedule, station_horizon

TRIAGE_METHODS = {
    "THR": lambda: NoRefresh(),
    "FRESH": lambda: FreshnessFiltered(),
    "IDW": lambda: IDWNowcast(),
    "AoI": lambda: AgeFirst(),
    "RND": lambda: RandomRefresh(),
    "ALL": lambda: PollAll(),
    "FURROW-R": lambda: FurrowRetrieval(),
}


@dataclass
class ScenarioConfig:
    n_farms: int = 200
    window_s: float = 300.0
    sync_mode: str = "uniform"          # 'uniform' | 'burst'
    burst_s: float = 30.0
    rho: float = 0.8                    # offered strategic load (full fidelity, cloud estimate)
    deadline_u: float = 10.0            # urgent chain deadline after data arrival (s)
    deadline_s: float = 120.0           # strategic query deadline (s)
    connectivity: str = "mixed"
    alert_cap: int = 10_000            # no cap: one U2 verification per alerted station
    budget_max: int = 8
    retrieval: str = "FURROW-R"
    slack_budget: bool = True           # FURROW-R polling budget from scheduler slack
    budget_margin: float = 1.0
    shift_days: int = 45
    season: tuple = ("2012-04-01", "2012-11-15")
    drain_s: float = 900.0
    extra_missing: float = 0.0          # robustness: extra fraction of readings removed
    qcap: int = 2


@dataclass
class RunResult:
    summary: dict
    chains: pd.DataFrame
    tasks: pd.DataFrame


class Library:
    """Measured task costs used by the simulator."""

    def __init__(self, calib_dir: Path):
        self.strategic = pd.read_csv(calib_dir / "strategic_library.csv")
        self.verify = pd.read_csv(calib_dir / "verify_costs.csv")
        tc = pd.read_csv(calib_dir / "triage_costs.csv")
        tc["base"] = tc["method"].str.split("@").str[0]
        self.triage_cpu = tc.groupby("base")["cpu"].mean().to_dict()

    def triage_cpu_for(self, name):
        if name in self.triage_cpu:
            return float(self.triage_cpu[name])
        return float(self.triage_cpu.get("THR", 0.002))


def _expected_alert_stations(res, st_c):
    """Prior expected number of stations with at least one alert (forecast for FURROW-S)."""
    from ..retrieval.triage import _expected_gain
    import pandas as _pd
    D0 = np.nan_to_num(res.prior_D, nan=-9.0)
    sd0 = np.nan_to_num(res.prior_sd, nan=0.0)
    _, pk = _expected_gain(D0, sd0)
    p = _pd.Series(pk[:, 1]).groupby(st_c).apply(lambda x: 1.0 - float(np.prod(1.0 - x.to_numpy())))
    return float(p.sum())


def _degrade(ctx, rng, frac):
    """Remove an extra random fraction of delivered readings (robustness experiment)."""
    if frac <= 0:
        return ctx
    from ..retrieval.triage import EvidenceStore, TriageContext
    F = ctx.store.F.copy()
    drop = (rng.random(F.shape) < frac) & (F >= 0)
    F[drop] = -1
    X = ctx.store.X
    return TriageContext(EvidenceStore(X, F), ctx.fc, ctx.ll, ctx.ports, ctx.unit_station, ctx.coords,
                         ctx.stale_model, ctx.month_of_day, ctx.truth_X, F)


def simulate(bundle, lib: Library, scen: ScenarioConfig, plat: PlatformConfig, scheduler_name: str,
             seed: int, c_eval: float, sched_kwargs: dict | None = None) -> RunResult:
    rng = np.random.default_rng(seed)
    wrng = np.random.default_rng(seed + 10_000)          # workload stream (identical across schedulers)
    dates = bundle.dates
    d12 = np.where((dates >= pd.Timestamp(scen.season[0])) & (dates <= pd.Timestamp(scen.season[1])))[0]
    all12 = np.where(dates.year == 2012)[0]
    run_day = int(wrng.choice(d12))
    ctx = _degrade(bundle.ctx, np.random.default_rng(seed + 20_000), scen.extra_missing)
    root = bundle.root_units
    n_st = int(ctx.unit_station.max()) + 1

    # ---------------- farms
    farms = []
    for f in range(scen.n_farms):
        shift = int(wrng.integers(-scen.shift_days, scen.shift_days + 1))
        day = int(np.clip(run_day + shift, all12[0] + 7, all12[-1]))
        P, ph = draw_schedule(n_st, scen.connectivity, wrng)
        if scen.sync_mode == "burst":
            ts = float(wrng.uniform(0, scen.burst_s))
        else:
            ts = float(wrng.uniform(0, scen.window_s))
        farms.append(dict(farm=f, day=day, P=P, ph=ph, sync=ts, trng_seed=int(wrng.integers(1 << 31))))

    # ---------------- strategic arrivals (offered load on shared cores)
    lib_s = lib.strategic
    full_cloud = np.array([exec_time(plat, Variant("full", "warm", r.parts_full, r.bytes_full, r.cpu_full),
                                     "cloud", False, plat.s_cloud) for r in lib_s.itertuples()])
    cap = plat.n_cloud * plat.k_cloud * plat.s_cloud + plat.n_fog * plat.k_fog * plat.s_fog
    lam = scen.rho * cap / float(np.mean(full_cloud))
    n_s = int(wrng.poisson(lam * scen.window_s))
    s_times = np.sort(wrng.uniform(0, scen.window_s, n_s))
    s_rows = wrng.integers(0, len(lib_s), n_s)
    s_farms = wrng.integers(0, scen.n_farms, n_s)
    s_noise = np.exp(wrng.normal(0, plat.noise_sigma, n_s))
    u_noise_rng = np.random.default_rng(seed + 30_000)

    # ---------------- engine & scheduler
    cores = build_cores(plat, scen.n_farms)
    if scheduler_name.startswith("FURROW"):
        sched = FurrowScheduler(**(sched_kwargs or {}))
    else:
        sched = BASELINES[scheduler_name]()
    eng = Engine(plat, cores, sched, c_eval, qcap=scen.qcap,
                 horizon_cap=scen.window_s + scen.drain_s)
    tid = [0]

    def new_tid():
        tid[0] += 1
        return tid[0]

    for t0, row, fm, nz in zip(s_times, s_rows, s_farms, s_noise):
        r = lib_s.iloc[row]
        var = {"full": Variant("full", "warm", int(r.parts_full), int(r.bytes_full), float(r.cpu_full), 0.0),
               "summary": Variant("summary", "summary", int(r.parts_summary), int(r.bytes_summary),
                                  float(r.cpu_summary), float(r.error_summary))}
        task = Task(new_tid(), "S", int(fm), float(t0), float(t0 + scen.deadline_s), var, "full", float(nz))
        task.qid = int(r.qid)
        task.template = r.template
        eng.submit(task)

    method = TRIAGE_METHODS[scen.retrieval]()
    u1_cpu = lib.triage_cpu_for(method.name)
    ver = lib.verify
    mean_u2 = Variant("std", "warm", int(round(ver.parts.mean())), int(ver.bytes.mean()), float(ver.cpu.mean()))
    if isinstance(sched, FurrowScheduler):
        sched.e_u2 = exec_time(plat, mean_u2, "cloud", False, plat.s_cloud)

    def probe_eft(cls, farm, variant):
        pt = Task(-1, cls, farm, eng.now, eng.now + scen.deadline_u, {"std": variant}, "std")
        pt.ready = eng.now
        return min(eng.eft(pt, c) for c in eng.eligible(pt))

    def on_sync(eng_, fm):
        f = fm["farm"]
        t = fm["day"]
        hz = station_horizon(t, fm["P"], fm["ph"])[ctx.unit_station]
        truth = np.where(bundle.F_off[:, t] == 0, bundle.X[:, t], np.nan)
        cand = root[~np.isnan(truth[root])]
        D = (ctx.fc[cand] - truth[cand]) / (ctx.fc[cand] - ctx.ll[cand])
        g = grade(D)
        # polling budget
        if method.name == "FURROW-R" and scen.slack_budget:
            u1_var = Variant("std", "index", 1, len(cand) * 16, u1_cpu)
            t_hat = probe_eft("U1", f, u1_var) - eng_.now + probe_eft("U2", f, mean_u2) - eng_.now
            slack = scen.deadline_u - scen.budget_margin - t_hat
            rounds = max(0, int(math.floor(slack / plat.poll_latency)))
            B = min(scen.budget_max, rounds * plat.poll_concurrency)
        else:
            B = scen.budget_max
        res = method(cand, t, hz, ctx, budget=B, rng=np.random.default_rng(fm["trng_seed"]))
        npoll = len(res.polled_stations)
        poll_time = math.ceil(npoll / plat.poll_concurrency) * plat.poll_latency if npoll else 0.0
        alerts = alerts_from(res, scen.alert_cap)
        ch = Chain(len(eng_.chains), f, eng_.now, eng_.now + scen.deadline_u, npoll, poll_time,
                   [int(a) for a in alerts], truth_grade={int(u): int(gg) for u, gg in zip(cand, g)},
                   prior_alerts=0.0)
        # verification is per station: expected number of stations with an alert
        st_c = ctx.unit_station[cand]
        ch.alert_stations = sorted({int(ctx.unit_station[a]) for a in alerts})
        ch.prior_alerts = float(_expected_alert_stations(res, st_c))
        ch.budget = B
        eng_.chains[ch.cid] = ch
        eng_.sched.notify_forecast(f, ch.prior_alerts, ch.deadline)
        u1_var = Variant("std", "index", 1, int(res.bytes_read), u1_cpu)
        u1 = Task(new_tid(), "U1", f, eng_.now + poll_time, ch.deadline, {"std": u1_var}, "std",
                  float(np.exp(u_noise_rng.normal(0, plat.noise_sigma))), chain=ch.cid)
        eng_.submit(u1)

    def on_u1_done(eng_, task):
        ch = eng_.chains[task.chain]
        ch.u1_done = eng_.now
        if hasattr(eng_.sched, "update_forecast"):
            eng_.sched.update_forecast(ch.farm, len(ch.alert_stations), ch.deadline)
        else:
            eng_.sched.clear_forecast(ch.farm)
        out = []
        if not ch.alerts:
            ch.done_time = eng_.now
            return out
        crng = np.random.default_rng(seed * 100_003 + ch.cid)
        for st in ch.alert_stations:
            r = ver.iloc[int(crng.integers(0, len(ver)))]
            v = Variant("std", "warm", int(r.parts), int(r.bytes), float(r.cpu))
            out.append(Task(new_tid(), "U2", ch.farm, eng_.now, ch.deadline, {"std": v}, "std",
                            float(np.exp(crng.normal(0, plat.noise_sigma))), chain=ch.cid, unit=int(st)))
        return out

    eng.unit_station = ctx.unit_station
    eng.next_sync = {fm["farm"]: fm["sync"] for fm in farms}   # published upload schedule
    if isinstance(sched, FurrowScheduler):
        sched.next_sync = eng.next_sync
    eng.on_sync = on_sync
    eng.on_u1_done = on_u1_done
    for fm in farms:
        eng.push(fm["sync"], "SYNC", fm)
    eng.run()
    return _collect(eng, scen, plat, scheduler_name, seed, run_day)


def _collect(eng: Engine, scen: ScenarioConfig, plat: PlatformConfig, sched_name: str, seed: int, run_day: int):
    cap_t = scen.window_s + scen.drain_s
    rows = []
    for t in eng.tasks:
        fin = t.finish if not math.isnan(t.finish) else math.inf
        rows.append(dict(tid=t.tid, cls=t.cls, farm=t.farm, arrival=t.arrival, deadline=t.deadline,
                         start=t.start, finish=t.finish, fid=t.fid, core=t.core,
                         ntype=eng.cores[t.core].ntype if t.core >= 0 else "none",
                         miss=bool(fin > t.deadline + 1e-9),
                         resp=(min(fin, cap_t) - t.arrival),
                         error=(t.variants[t.fid].error if t.cls == "S" else 0.0),
                         template=getattr(t, "template", "")))
    tasks = pd.DataFrame(rows)
    crow = []
    for ch in eng.chains.values():
        done = ch.done_time if not math.isnan(ch.done_time) else math.inf
        pos = {u for u, g in ch.truth_grade.items() if g >= 2}
        al = set(ch.alerts)
        tp = al & pos
        timely = {u for u in tp if ch.u2_finish.get(int(eng.unit_station[u]), math.inf) <= ch.deadline + 1e-9}
        crow.append(dict(chain=ch.cid, farm=ch.farm, sync=ch.sync_time, polls=ch.polls, budget=ch.budget,
                         poll_time=ch.poll_time, n_alerts=len(al), tp=len(tp), fp=len(al - pos),
                         fn=len(pos - al), positives=len(pos), timely_tp=len(timely),
                         n_u2=len(ch.alert_stations),
                         latency=min(done, cap_t) - ch.sync_time, miss=bool(done > ch.deadline + 1e-9),
                         prior_alerts=ch.prior_alerts))
    chains = pd.DataFrame(crow)
    u = tasks[tasks.cls.isin(["U1", "U2"])]
    s = tasks[tasks.cls == "S"]
    fin = tasks.finish.max()
    makespan = float(fin) if np.isfinite(fin) else cap_t
    shared = [c for c in eng.cores if c.ntype != "gw"]
    util = sum(c.busy_time for c in shared) / (len(shared) * max(makespan, 1e-9))
    pos = chains.positives.sum()
    summary = dict(
        scheduler=sched_name, retrieval=scen.retrieval, seed=seed, run_day=run_day,
        n_tasks=len(tasks), n_strategic=len(s), n_urgent=len(u), n_chains=len(chains),
        chain_miss=float(chains.miss.mean()), chain_lat_mean=float(chains.latency.mean()),
        chain_lat_p95=float(chains.latency.quantile(0.95)),
        urgent_resp_mean=float(u.resp.mean()), urgent_resp_p95=float(u.resp.quantile(0.95)),
        urgent_miss=float(u.miss.mean()),
        u2_resp_mean=float(tasks[tasks.cls == "U2"].resp.mean()) if (tasks.cls == "U2").any() else 0.0,
        strat_resp_mean=float(s.resp.mean()), strat_resp_p95=float(s.resp.quantile(0.95)),
        strat_miss=float(s.miss.mean()), strat_summary_frac=float((s.fid == "summary").mean()),
        strat_error_mean=float(s.error.mean()),
        strat_completed=float(np.isfinite(s.finish).mean()),
        alerts=int(chains.n_alerts.sum()), tp=int(chains.tp.sum()), fp=int(chains.fp.sum()),
        fn=int(chains.fn.sum()), positives=int(pos),
        alert_precision=float(chains.tp.sum() / max(chains.n_alerts.sum(), 1)),
        alert_recall=float(chains.tp.sum() / max(pos, 1)),
        timely_recall=float(chains.timely_tp.sum() / max(pos, 1)),
        polls=int(chains.polls.sum()), mean_budget=float(chains.budget.mean()),
        throughput=float(np.isfinite(tasks.finish).sum() / max(makespan, 1e-9)),
        makespan=makespan, util_shared=float(util),
        energy_dyn_kJ=eng.energy_dyn / 1000.0, cost=eng.cost,
        sched_overhead_s=eng.n_evals_total * eng.c_eval,
        gw_strategic=int(((s.ntype == "gw")).sum()),
    )
    return RunResult(summary, chains, tasks)
