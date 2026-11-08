"""Baseline schedulers.

FCFS    first-come first-served; task to the earliest-available core
RR      round-robin over eligible cores, FIFO
SJF     shortest (estimated) job first; earliest-available core
EDF     earliest deadline first; minimum estimated finish time (EFT) core
PRIO    static two-class priority (urgent before strategic), FIFO within class, EFT core
MCT     minimum completion time (Braun et al., 2001), arrival order
MINMIN  batch Min-Min (Braun et al., 2001)
GA      batch genetic algorithm minimising weighted tardiness (+ flow-time tie-break)

Every baseline executes the default retrieval variant of a task (strategic queries
at full fidelity); data-access time is part of the execution-time estimate for all
methods, so all methods are locality-aware to the same degree except FCFS/RR/SJF,
which follow their textbook core-selection rules.
"""
from __future__ import annotations

import heapq

import numpy as np

from .base import Scheduler


class FCFS(Scheduler):
    name = "FCFS"

    def choose(self, task):
        c, ne = self.earliest_available_with_room(task)
        return c, None, ne


class RoundRobin(Scheduler):
    name = "RR"

    def attach(self, eng):
        super().attach(eng)
        self.ptr = 0

    def choose(self, task):
        eng = self.eng
        cores = eng.eligible(task)
        n = len(cores)
        for i in range(n):
            c = cores[(self.ptr + i) % n]
            if eng.has_room(c):
                self.ptr += i + 1
                return c, None, i + 1
        return None, None, n


class SJF(Scheduler):
    name = "SJF"

    def key(self, task):
        eng = self.eng
        e = min(eng.est(task, c) for c in eng.candidates(task, room_only=False))
        return (e, task.tid)

    def choose(self, task):
        c, ne = self.earliest_available_with_room(task)
        return c, None, ne


class EDF(Scheduler):
    name = "EDF"

    def key(self, task):
        return (task.deadline, task.tid)

    def local_key(self, task):
        return (task.deadline,)

    def choose(self, task):
        c, _, ne = self.min_eft(task)
        return c, None, ne


class PriorityEDF(Scheduler):
    """Static class priority: urgent tasks first (FIFO), then strategic (FIFO)."""
    name = "PRIO"

    def key(self, task):
        return (0 if task.urgent else 1, task.arrival, task.tid)

    def local_key(self, task):
        return (0 if task.urgent else 1,)

    def choose(self, task):
        c, _, ne = self.min_eft(task)
        return c, None, ne


class MCT(Scheduler):
    name = "MCT"

    def choose(self, task):
        c, _, ne = self.min_eft(task)
        return c, None, ne


class BatchScheduler(Scheduler):
    batch_interval = 0.5
    batch_max = 64

    def dispatch(self):          # placements only happen at ticks
        return

    def _batch(self):
        items = []
        while self.pending and len(items) < self.batch_max:
            items.append(heapq.heappop(self.pending))
        return items


class MinMin(BatchScheduler):
    name = "MINMIN"

    def on_tick(self):
        eng = self.eng
        items = self._batch()
        if not items:
            return
        tasks = [it[2] for it in items]
        cores = {c.cid: c for c in eng.shared}
        for t in tasks:
            g = eng.gw_of.get(t.farm)
            if g is not None:
                cores[g.cid] = g
        avail = {cid: eng.avail(c) for cid, c in cores.items()}
        occ = {cid: (0 if c.running is None else 1) + len(c.queue) for cid, c in cores.items()}
        groups = [cl for cl in eng.classes.values()]

        def reps(t):
            out = []
            for cl in groups:
                best, ba = None, float("inf")
                for c in cl:
                    if occ[c.cid] >= eng.qcap + 1:
                        continue
                    if avail[c.cid] < ba - 1e-12:
                        best, ba = c, avail[c.cid]
                if best is not None:
                    out.append(best)
            g = eng.gw_of.get(t.farm)
            if g is not None and occ[g.cid] < eng.qcap + 1:
                out.append(g)
            return out

        plan, evals = [], 0
        remaining = list(range(len(tasks)))
        while remaining:
            best = None
            for i in remaining:
                t = tasks[i]
                evals += self.n_eligible(t)
                for c in reps(t):
                    f = max(avail[c.cid], t.ready) + eng.est(t, c)
                    if best is None or f < best[0]:
                        best = (f, i, c)
            if best is None:
                break
            f, i, c = best
            plan.append((tasks[i], c))
            avail[c.cid] = f
            occ[c.cid] += 1
            remaining.remove(i)
        overhead = evals * eng.c_eval
        for t, c in plan:
            t.ready = max(t.ready, eng.now + overhead)
            eng.place(t, c, None, 0)
        eng.n_evals_total += evals
        for i in remaining:
            heapq.heappush(self.pending, items[i])


class GeneticScheduler(BatchScheduler):
    """Batch GA: chromosome = core index per task; per-core execution in deadline order.
    Fitness = sum_c w_c * tardiness + 1e-3 * flow time (w_U = 10, w_S = 1).
    Population 20, 30 generations, tournament(2), uniform crossover 0.9,
    mutation 0.05 per gene, elitism 2; seeded with the MCT assignment."""
    name = "GA"
    batch_interval = 1.0
    pop, gens, pc, pm, elite = 20, 30, 0.9, 0.05, 2

    def attach(self, eng):
        super().attach(eng)
        self.rng = np.random.default_rng(12345)

    def on_tick(self):
        eng = self.eng
        items = self._batch()
        if not items:
            return
        tasks = [it[2] for it in items]
        cores = {}
        for t in tasks:
            for c in eng.eligible(t):
                cores[c.cid] = c
        cl = list(cores.values())
        cidx = {c.cid: j for j, c in enumerate(cl)}
        n, m = len(tasks), len(cl)
        E = np.full((n, m), np.inf)
        for i, t in enumerate(tasks):
            for c in eng.eligible(t):
                E[i, cidx[c.cid]] = eng.est(t, c)
        avail = np.array([eng.avail(c) for c in cl])
        ready = np.array([t.ready for t in tasks])
        dl = np.array([t.deadline for t in tasks])
        w = np.array([10.0 if t.urgent else 1.0 for t in tasks])
        order = np.argsort(dl, kind="stable")
        elig = [np.where(np.isfinite(E[i]))[0] for i in range(n)]

        def fitness(ch):
            fin = np.empty(n)
            av = avail.copy()
            for i in order:
                j = ch[i]
                s = max(av[j], ready[i])
                av[j] = s + E[i, j]
                fin[i] = av[j]
            return float(np.sum(w * np.maximum(0.0, fin - dl)) + 1e-3 * np.sum(fin - ready))

        # seed: greedy MCT
        seed = np.empty(n, int)
        av = avail.copy()
        for i in order:
            f = np.maximum(av, ready[i]) + E[i]
            j = int(np.argmin(f))
            seed[i] = j
            av[j] = f[j]
        P = [seed] + [np.array([self.rng.choice(elig[i]) for i in range(n)]) for _ in range(self.pop - 1)]
        F = np.array([fitness(p) for p in P])
        evals = self.pop * n
        for _ in range(self.gens):
            elite_idx = np.argsort(F)[: self.elite]
            newP = [P[i].copy() for i in elite_idx]
            while len(newP) < self.pop:
                a, b = self.rng.integers(0, self.pop, 2)
                p1 = P[a] if F[a] < F[b] else P[b]
                a, b = self.rng.integers(0, self.pop, 2)
                p2 = P[a] if F[a] < F[b] else P[b]
                child = p1.copy()
                if self.rng.random() < self.pc:
                    mask = self.rng.random(n) < 0.5
                    child[mask] = p2[mask]
                mut = np.where(self.rng.random(n) < self.pm)[0]
                for i in mut:
                    child[i] = self.rng.choice(elig[i])
                newP.append(child)
            P = newP
            F = np.array([fitness(p) for p in P])
            evals += self.pop * n
        best = P[int(np.argmin(F))]
        overhead = evals * eng.c_eval
        for i in order:
            t = tasks[i]
            t.ready = max(t.ready, eng.now + overhead)
            # GA plans the whole batch: local queues may exceed qcap
            eng.place(t, cl[best[i]], None, 0)
        eng.n_evals_total += evals

    def local_key(self, task):
        return (task.deadline,)


BASELINES = {c.name: c for c in [FCFS, RoundRobin, SJF, EDF, PriorityEDF, MCT, MinMin, GeneticScheduler]}
