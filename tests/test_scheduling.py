import math

import numpy as np
import pytest

from furrow.scheduling.baselines import EDF, FCFS
from furrow.scheduling.furrow import FurrowScheduler
from furrow.simulator.engine import Engine, Task
from furrow.simulator.platform import Core, PlatformConfig, Variant, exec_time


def test_exec_time_model():
    cfg = PlatformConfig()
    v = Variant("full", "warm", parts=16, nbytes=5_000_000, cpu=0.5)
    # cloud: 2 rounds x 20 ms + 5e6/50e6 + 0.5/1.0
    assert exec_time(cfg, v, "cloud", False, 1.0) == pytest.approx(0.04 + 0.1 + 0.5)
    # gateway: 2 x 120 ms + 5e6/1.25e6 + 0.5/0.2
    assert exec_time(cfg, v, "gw", True, 0.2) == pytest.approx(0.24 + 4.0 + 2.5)
    vi = Variant("std", "index", 1, 0, 0.01)
    assert exec_time(cfg, vi, "gw", True, 0.2) == pytest.approx(0.001 + 0.05)


def _engine(sched, n_cores=1):
    cfg = PlatformConfig()
    cores = [Core(i, i, "cloud", -1, 1.0, 1) for i in range(n_cores)]
    return Engine(cfg, cores, sched, c_eval=0.0, qcap=2)


def _task(tid, cls, arr, dl, cpu, summary_cpu=None):
    var = {"full": Variant("full", "index", 0, 0, cpu)}
    if summary_cpu is not None:
        var["summary"] = Variant("summary", "index", 0, 0, summary_cpu, error=0.1)
    return Task(tid, cls, 0, arr, dl, var, "full")


def test_fcfs_sequential_exact():
    eng = _engine(FCFS())
    eng.submit(_task(1, "S", 0.0, 100, 1.0))
    eng.submit(_task(2, "S", 0.1, 100, 2.0))
    eng.run()
    t1, t2 = eng.tasks
    assert t1.finish == pytest.approx(1.0) and t2.start == pytest.approx(1.0) and t2.finish == pytest.approx(3.0)


def test_edf_orders_by_deadline():
    eng = _engine(EDF())
    eng.submit(_task(1, "S", 0.0, 100, 1.0))      # occupies the core
    eng.submit(_task(2, "S", 0.1, 50, 1.0))
    eng.submit(_task(3, "S", 0.2, 10, 1.0))
    eng.run()
    fin = {t.tid: t.finish for t in eng.tasks}
    assert fin[3] < fin[2]


def test_furrow_urgent_bypasses_strategic_queue():
    eng = _engine(FurrowScheduler(reservation="none"))
    eng.submit(_task(1, "S", 0.0, 100, 5.0))
    eng.submit(_task(2, "S", 0.0, 100, 5.0))       # waits globally (never queued locally)
    eng.submit(_task(3, "U2", 1.0, 20, 0.1))
    eng.run()
    fin = {t.tid: t.finish for t in eng.tasks}
    assert fin[3] == pytest.approx(5.1)             # waits only for the running task
    assert fin[2] == pytest.approx(10.1)


def test_furrow_reservation_withholds_core():
    s = FurrowScheduler(reservation="forecast", r_max=1, e_u2_hint=0.05)
    eng = _engine(s, n_cores=2)
    s.notify_forecast(farm=0, n_hat=3, until=100.0)
    eng.submit(_task(1, "S", 0.0, 100, 5.0))
    eng.submit(_task(2, "S", 0.0, 100, 5.0))
    eng.submit(_task(3, "U2", 1.0, 20, 0.1))
    eng.run()
    fin = {t.tid: t.finish for t in eng.tasks}
    assert fin[3] == pytest.approx(1.1)             # reserved core serves the urgent task at once
    assert fin[2] > 5.0                             # second strategic task had to wait


def test_furrow_fidelity_switch():
    eng = _engine(FurrowScheduler(reservation="none"))
    eng.submit(_task(1, "S", 0.0, 100, 5.0, summary_cpu=0.5))
    eng.submit(_task(2, "S", 0.0, 6.0, 5.0, summary_cpu=0.5))   # full cannot meet 6 s after 5 s wait
    eng.run()
    t = {x.tid: x for x in eng.tasks}
    assert t[1].fid == "full" and t[2].fid == "summary"
    assert t[2].finish <= 6.0
