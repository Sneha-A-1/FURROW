"""Scheduler interface shared by FURROW-S and all baselines."""
from __future__ import annotations

import heapq


class Scheduler:
    name = "base"
    batch_interval: float | None = None
    window = 64          # tasks examined per dispatch call (look-ahead window)

    def attach(self, eng):
        self.eng = eng
        self.pending: list = []

    # ordering ----------------------------------------------------------------
    def key(self, task):
        return (task.arrival, task.tid)

    def local_key(self, task):
        return (0,)       # FIFO within a core's local queue

    # events ------------------------------------------------------------------
    def submit(self, task):
        heapq.heappush(self.pending, (self.key(task), task.tid, task))

    def has_pending(self):
        return bool(self.pending)

    def on_done(self, task):
        pass

    def on_tick(self):
        pass

    def notify_forecast(self, farm, n_hat, until):
        pass

    def clear_forecast(self, farm):
        pass

    # placement ---------------------------------------------------------------
    def choose(self, task):
        """Return (core or None, variant id or None, evaluations)."""
        raise NotImplementedError

    def dispatch(self):
        eng = self.eng
        if not self.pending:
            return
        held = []
        n = 0
        while self.pending and n < self.window:
            item = heapq.heappop(self.pending)
            n += 1
            core, fid, ne = self.choose(item[2])
            if core is not None and eng.has_room(core):
                eng.place(item[2], core, fid, ne)
            else:
                held.append(item)
        for it in held:
            heapq.heappush(self.pending, it)

    # helpers -----------------------------------------------------------------
    def n_eligible(self, task):
        return len(self.eng.shared) + (1 if task.farm in self.eng.gw_of else 0)

    def min_eft(self, task, cores=None, fid=None, room_only=False):
        """Minimum estimated finish time over all eligible cores (evaluated exactly via
        per-class representatives; the evaluation count charged is the number of
        eligible cores)."""
        eng = self.eng
        cands = cores if cores is not None else eng.candidates(task, room_only)
        best, bt = None, float("inf")
        for c in cands:
            f = eng.eft(task, c, fid)
            if f < bt - 1e-12:
                best, bt = c, f
        return best, bt, self.n_eligible(task)

    def earliest_available_with_room(self, task):
        eng = self.eng
        best, bt = None, float("inf")
        for c in eng.candidates(task, room_only=True):
            a = eng.avail(c)
            if a < bt - 1e-12:
                best, bt = c, a
        return best, self.n_eligible(task)
