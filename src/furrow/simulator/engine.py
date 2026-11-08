"""Discrete-event simulation engine (Layers 5-6, Layer 8 monitoring hooks).

Model
-----
* Each core executes one task at a time, non-preemptively, and holds a bounded local
  queue (``qcap`` tasks) in the order given by the scheduler's ``local_key``.
* Schedulers keep their own global pending queues and place tasks onto cores in
  ``dispatch``; a placement costs ``n_evals * c_eval`` seconds of controller time,
  added to the task's earliest start (scheduling overhead is charged to every method
  with the same per-evaluation constant).
* Actual execution time = estimated time x lognormal noise drawn once per task.
* Urgent chains: SYNC -> (edge refresh polls) -> U1 triage -> n_alert U2 verifications.
"""
from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .platform import Core, PlatformConfig, Variant, exec_time

URGENT = ("U1", "U2")


@dataclass
class Task:
    tid: int
    cls: str
    farm: int
    arrival: float
    deadline: float
    variants: dict
    fid: str
    noise: float = 1.0
    chain: int = -1
    unit: int = -1
    ready: float = 0.0
    core: int = -1
    start: float = math.nan
    finish: float = math.nan
    est_exec: float = math.nan
    _cache: dict = field(default_factory=dict)

    @property
    def urgent(self):
        return self.cls in URGENT


@dataclass
class Chain:
    cid: int
    farm: int
    sync_time: float
    deadline: float
    polls: int = 0
    poll_time: float = 0.0
    alerts: list = field(default_factory=list)      # alerted unit ids
    alert_stations: list = field(default_factory=list)  # stations verified by U2 tasks
    n_u2_done: int = 0
    u1_done: float = math.nan
    done_time: float = math.nan
    u2_finish: dict = field(default_factory=dict)   # station -> U2 finish time
    truth_grade: dict = field(default_factory=dict) # unit -> grade (all candidates)
    prior_alerts: float = 0.0


class Engine:
    def __init__(self, cfg: PlatformConfig, cores: list[Core], scheduler, c_eval: float,
                 qcap: int = 2, horizon_cap: float = 1e9):
        self.cfg = cfg
        self.cores = cores
        self.sched = scheduler
        self.c_eval = c_eval
        self.qcap = qcap
        self.cap = horizon_cap
        self.now = 0.0
        self._heap: list = []
        self._seq = itertools.count()
        self.tasks: list[Task] = []
        self.chains: dict[int, Chain] = {}
        self.shared = [c for c in cores if c.ntype != "gw"]
        self.gw_of = {c.farm: c for c in cores if c.ntype == "gw"}
        # cores of one class share type and speed, hence identical execution estimates
        self.classes: dict = {}
        for c in self.shared:
            self.classes.setdefault((c.ntype, c.speed), []).append(c)
        self._best_cache: dict = {}
        self.on_u1_done = None          # callback(engine, task) -> list of new tasks
        self.on_sync = None             # callback(engine, chain_payload)
        self.n_evals_total = 0
        self.energy_dyn = 0.0
        self.cost = 0.0
        scheduler.attach(self)

    # ------------------------------------------------------------------ events
    def push(self, t: float, kind: str, payload=None):
        heapq.heappush(self._heap, (t, next(self._seq), kind, payload))

    # ------------------------------------------------------------------ estimates
    def eligible(self, task: Task):
        g = self.gw_of.get(task.farm)
        return self.shared + ([g] if g is not None else [])

    def est(self, task: Task, core: Core, fid: str | None = None) -> float:
        fid = fid or task.fid
        own = core.farm == task.farm
        key = (fid, core.ntype, own)
        v = task._cache.get(key)
        if v is None:
            v = exec_time(self.cfg, task.variants[fid], core.ntype, own, core.speed)
            task._cache[key] = v
        return v

    def avail(self, core: Core) -> float:
        base = max(self.now, core.busy_until) if core.running is not None else self.now
        return base + core.queued_work

    def eft(self, task: Task, core: Core, fid: str | None = None, ahead: float | None = None) -> float:
        base = max(self.now, core.busy_until) if core.running is not None else self.now
        aw = core.queued_work if ahead is None else ahead
        return max(base + aw, task.ready) + self.est(task, core, fid)

    def class_best(self, room_only: bool):
        """Per shared-core class, the core with the earliest availability (optionally only
        cores with local-queue room). Exact for EFT minimisation because estimates are
        identical within a class. Cached until the next state change."""
        key = room_only
        hit = self._best_cache.get(key)
        if hit is not None:
            return hit
        out = []
        for cl in self.classes.values():
            best, ba = None, float("inf")
            for c in cl:
                if room_only and not self.has_room(c):
                    continue
                a = self.avail(c)
                if a < ba - 1e-12:
                    best, ba = c, a
            if best is not None:
                out.append(best)
        self._best_cache[key] = out
        return out

    def candidates(self, task: Task, room_only: bool):
        """Representative cores (best per shared class + own gateway)."""
        out = list(self.class_best(room_only))
        g = self.gw_of.get(task.farm)
        if g is not None and (not room_only or self.has_room(g)):
            out.append(g)
        return out

    def invalidate(self):
        self._best_cache.clear()

    def has_room(self, core: Core, cap: int | None = None) -> bool:
        cap = self.qcap if cap is None else cap
        return core.running is None or len(core.queue) < cap

    def idle(self, core: Core) -> bool:
        return core.running is None and not core.queue

    def dyn_power(self, core: Core) -> float:
        return (self.cfg.p_max[core.ntype] - self.cfg.p_idle[core.ntype]) / core.k

    # ------------------------------------------------------------------ actions
    def place(self, task: Task, core: Core, fid: str | None = None, n_evals: int = 0):
        if fid:
            task.fid = fid
        self.n_evals_total += n_evals
        task.ready = max(task.ready, self.now + n_evals * self.c_eval)
        task.core = core.cid
        task.est_exec = self.est(task, core)
        key = self.sched.local_key(task)
        i = 0
        while i < len(core.queue) and self.sched.local_key(core.queue[i]) <= key:
            i += 1
        core.queue.insert(i, task)
        core.queued_work += task.est_exec
        self._best_cache.clear()
        if core.running is None:
            self._start(core)

    def _start(self, core: Core):
        if core.running is not None or not core.queue:
            return
        task = core.queue[0]
        if task.ready > self.now + 1e-12:
            self.push(task.ready, "WAKE", core.cid)
            return
        core.queue.pop(0)
        core.queued_work = max(0.0, core.queued_work - task.est_exec)
        self._best_cache.clear()
        actual = task.est_exec * task.noise
        task.start = self.now
        core.running = task
        core.busy_until = self.now + actual
        core.busy_time += actual
        self.energy_dyn += self.dyn_power(core) * actual
        self.cost += self.cfg.price[core.ntype] * actual
        self.push(core.busy_until, "DONE", core.cid)

    def submit(self, task: Task):
        self.tasks.append(task)
        self.push(task.arrival, "ARRIVE", task)

    # ------------------------------------------------------------------ main loop
    def run(self):
        sched = self.sched
        if sched.batch_interval:
            self.push(sched.batch_interval, "TICK")
        while self._heap:
            t, _, kind, payload = heapq.heappop(self._heap)
            if t > self.cap:
                break
            self.now = t
            self._best_cache.clear()
            if kind == "ARRIVE":
                payload.ready = max(payload.ready, t)
                sched.submit(payload)
            elif kind == "DONE":
                core = self.cores[payload]
                task = core.running
                core.running = None
                self._best_cache.clear()
                task.finish = t
                sched.on_done(task)
                if task.cls == "U1" and self.on_u1_done:
                    for nt in self.on_u1_done(self, task):
                        nt.arrival = t
                        nt.ready = t
                        self.tasks.append(nt)
                        sched.submit(nt)
                elif task.cls == "U2" and task.chain in self.chains:
                    ch = self.chains[task.chain]
                    ch.u2_finish[task.unit] = t
                    ch.n_u2_done += 1
                    if ch.n_u2_done == len(ch.alert_stations):
                        ch.done_time = t
                self._start(core)
            elif kind == "WAKE":
                self._start(self.cores[payload])
            elif kind == "SYNC":
                self.on_sync(self, payload)
            elif kind == "TICK":
                sched.on_tick()
                if self._pending_work():
                    self.push(t + sched.batch_interval, "TICK")
            sched.dispatch()
        return self

    def _pending_work(self) -> bool:
        return bool(self._heap) or self.sched.has_pending() or any(c.running or c.queue for c in self.cores)
