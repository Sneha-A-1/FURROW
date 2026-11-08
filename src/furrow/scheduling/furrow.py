"""FURROW-S: laxity-ordered, forecast-reserved, plan-aware scheduler (Layer 5).

Objective (lexicographic)
-------------------------
  1. minimise weighted deadline misses   sum_tau w_c(tau) * 1[C_tau > d_tau],  w_U >> w_S
  2. minimise urgent response time       sum_{tau in U} (C_tau - a_tau)
  3. minimise strategic dynamic energy   sum_{tau in S} e(tau, n) * (P_max,n - P_idle,n) / k_n
     subject to the strategic answer-error incurred by fidelity degradation
     (reported as a separate quality metric).

Rules
-----
Urgent tasks (U1, U2)
  * global queue ordered by static laxity  d_tau - min_n e(tau, n)  (least laxity first);
  * placed on the core with minimum EFT; because strategic work is never queued
    locally, an urgent task waits at most for the residual of one running task;
  * local queues order urgent tasks by laxity.
Strategic tasks (S)
  * global queue ordered by deadline (EDF); a strategic task is only started on an
    *idle* core (no local queueing). The farm's own gateway is used opportunistically
    only if it has no active urgency forecast and the task is expected to finish
    before the farm's next scheduled data upload (minus a guard), so that edge
    capacity is free when the farm's urgent work arrives;
  * capacity reservation: r(t) fastest idle shared cores are withheld, where
        r(t) = min(r_max, ceil( sum_f nhat_f * e_U2 / tau_r ))  (0 if no forecast)
    and nhat_f is the pending urgent verification work of farm f: the expected number
    of alert stations announced by FURROW-R at data arrival, replaced by the actual
    number when triage completes and decremented as verification tasks are placed
    (retrieval -> scheduler forecast signal);
  * plan-aware fidelity: among idle unreserved cores choose the full-fidelity variant
    if some placement meets the deadline (minimum dynamic energy among those),
    otherwise the summary variant (minimum energy among deadline-feasible ones),
    otherwise the summary variant on the minimum-EFT core;
  * starvation guard: a strategic task whose summary slack is below g_s may use
    reserved cores.

Ablation switches: laxity, reservation ('forecast' | 'ewma' | 'none'), fidelity,
locality (if False, data-access time is ignored in the scheduler's estimates).
"""
from __future__ import annotations

import heapq
import math

from .base import Scheduler


class FurrowScheduler(Scheduler):
    name = "FURROW-S"

    def __init__(self, laxity=True, reservation="forecast", fidelity=True, locality=True,
                 r_max=8, tau_r=0.25, guard=2.0, e_u2_hint=0.05, ewma_halflife=10.0,
                 gw_opportunistic=True, gw_guard=15.0, name=None):
        self.laxity = laxity
        self.reservation = reservation
        self.fidelity = fidelity
        self.locality = locality
        self.r_max = r_max
        self.tau_r = tau_r
        self.guard = guard
        self.e_u2 = e_u2_hint
        self.hl = ewma_halflife
        self.gw_opportunistic = gw_opportunistic
        self.gw_guard = gw_guard
        if name:
            self.name = name

    def attach(self, eng):
        super().attach(eng)
        self.qu: list = []
        self.qs: list = []
        self.forecast: dict = {}
        self.rate = 0.0
        self.rate_t = 0.0
        self.n_summary = 0
        self.next_sync = getattr(eng, "next_sync", {})

    # ------------------------------------------------------------------ helpers
    def _est(self, task, core, fid=None):
        eng = self.eng
        if self.locality:
            return eng.est(task, core, fid)
        v = task.variants[fid or task.fid]
        return v.cpu / core.speed

    def _eft(self, task, core, fid=None):
        eng = self.eng
        base = max(eng.now, core.busy_until) if core.running is not None else eng.now
        return max(base + core.queued_work, task.ready) + self._est(task, core, fid)

    def local_key(self, task):
        if task.urgent:
            return (0, task.deadline - task._lax_e if self.laxity else task.arrival)
        return (1, task.deadline)

    def has_pending(self):
        return bool(self.qu or self.qs)

    # ------------------------------------------------------------------ signals
    def notify_forecast(self, farm, n_hat, until):
        if self.reservation == "forecast" and n_hat > 0:
            self.forecast[farm] = (n_hat, until)

    def clear_forecast(self, farm):
        self.forecast.pop(farm, None)

    def update_forecast(self, farm, n_pending, until):
        """After triage the forecast becomes the number of verification tasks still to be
        started; it is decremented as they are placed and removed when it reaches zero."""
        if self.reservation != "forecast":
            return
        if n_pending > 0:
            self.forecast[farm] = (float(n_pending), until)
        else:
            self.forecast.pop(farm, None)

    def _reserve(self):
        now = self.eng.now
        if self.reservation == "none":
            return 0
        if self.reservation == "forecast":
            for f in [f for f, (_, u) in self.forecast.items() if u < now]:
                self.forecast.pop(f)
            if not self.forecast:
                return 0
            need = sum(n for n, _ in self.forecast.values()) * self.e_u2
            return min(self.r_max, max(1, math.ceil(need / self.tau_r)))
        # 'ewma': rate of urgent arrivals with exponential time decay
        decay = 0.5 ** ((now - self.rate_t) / self.hl)
        lam = self.rate * decay
        need = lam * self.e_u2 / 0.5
        return min(self.r_max, math.ceil(need)) if lam > 1e-9 else 0

    # ------------------------------------------------------------------ events
    def submit(self, task):
        eng = self.eng
        if task.urgent:
            task._lax_e = min(self._est(task, c) for c in eng.candidates(task, room_only=False))
            k = (task.deadline - task._lax_e) if self.laxity else task.arrival
            heapq.heappush(self.qu, (k, task.tid, task))
            if self.reservation == "ewma":
                now = eng.now
                self.rate = self.rate * 0.5 ** ((now - self.rate_t) / self.hl) + math.log(2) / self.hl
                self.rate_t = now
        else:
            heapq.heappush(self.qs, (task.deadline, task.tid, task))

    def dispatch(self):
        eng = self.eng
        # 1) urgent: least laxity first, minimum-EFT core with room
        held = []
        n = 0
        while self.qu and n < self.window:
            item = heapq.heappop(self.qu)
            n += 1
            t = item[2]
            best, bt = None, float("inf")
            ne = self.n_eligible(t)
            for c in eng.candidates(t, room_only=True):
                f = self._eft(t, c)
                if f < bt - 1e-12:
                    best, bt = c, f
            if best is None:
                held.append(item)
            else:
                eng.place(t, best, None, ne)
                if t.cls == "U2" and t.farm in self.forecast:
                    n_left, until = self.forecast[t.farm]
                    if n_left <= 1:
                        self.forecast.pop(t.farm)
                    else:
                        self.forecast[t.farm] = (n_left - 1, until)
        for it in held:
            heapq.heappush(self.qu, it)
        if not self.qs:
            return
        # 2) strategic: idle unreserved shared cores, plan-aware fidelity
        idle = [c for c in eng.shared if eng.idle(c)]
        if not idle and not self.gw_opportunistic:
            return
        r = self._reserve()
        idle.sort(key=lambda c: -c.speed)
        reserved = idle[:r]
        free = idle[r:]
        held = []
        n = 0
        while self.qs and n < self.window:
            item = heapq.heappop(self.qs)
            n += 1
            t = item[2]
            pool = free
            cheapest = "summary" if (self.fidelity and "summary" in t.variants) else t.fid
            min_e = min((self._est(t, c, cheapest) for c in idle), default=float("inf"))
            if t.deadline - eng.now - min_e < self.guard:
                pool = free + reserved
            g = eng.gw_of.get(t.farm)
            gw_ok = (self.gw_opportunistic and g is not None and eng.idle(g)
                     and t.farm not in self.forecast)
            if gw_ok:
                horizon = self.next_sync.get(t.farm, float("inf")) - self.gw_guard
                cheapest_g = self._est(t, g, "summary" if (self.fidelity and "summary" in t.variants) else t.fid)
                gw_ok = eng.now + cheapest_g <= horizon
            cand_pool = pool + ([g] if gw_ok else [])
            if not cand_pool:
                held.append(item)
                continue
            choice = self._choose_plan(t, cand_pool, gw=g if gw_ok else None)
            core, fid, ne = choice
            eng.place(t, core, fid, ne)
            if fid == "summary":
                self.n_summary += 1
            if core in free:
                free.remove(core)
            elif core in reserved:
                reserved.remove(core)
        for it in held:
            heapq.heappush(self.qs, it)

    def _choose_plan(self, t, pool, gw=None):
        eng = self.eng
        fids = ["full", "summary"] if (self.fidelity and "summary" in t.variants) else [t.fid]
        ne = 0
        fallback = None
        limit_gw = (self.next_sync.get(t.farm, float("inf")) - self.gw_guard) if gw is not None else None
        for fid in fids:
            best, be = None, float("inf")
            for c in pool:
                ne += 1
                fin = max(eng.now, t.ready) + self._est(t, c, fid)
                if c is gw and fin > limit_gw:
                    continue
                if fallback is None or fin < fallback[0]:
                    fallback = (fin, c, fid)
                if fin <= t.deadline:
                    en = eng.est(t, c, fid) * eng.dyn_power(c)
                    if en < be - 1e-12:
                        best, be = c, en
            if best is not None:
                return best, fid, ne
        # nothing meets the deadline: cheapest-to-finish variant on min-EFT shared core
        shared_pool = [c for c in pool if c is not gw] or pool
        if self.fidelity and "summary" in t.variants:
            best = min(shared_pool, key=lambda c: self._est(t, c, "summary"))
            return best, "summary", ne
        best = min(shared_pool, key=lambda c: self._est(t, c, t.fid))
        return best, t.fid, ne
