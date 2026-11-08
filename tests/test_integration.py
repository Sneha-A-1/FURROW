"""Integration tests: configuration consistency, statistics helpers and a small end-to-end run."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from furrow.evaluation.stats import holm, mean_ci, paired_test
from furrow.scheduling.baselines import GeneticScheduler, MinMin
from furrow.simulator.platform import PlatformConfig
from furrow.utils.config import build, load_default

ROOT = Path(__file__).resolve().parents[1]


def test_yaml_matches_simulator_defaults():
    cfg = load_default()
    scen, plat, fk = build(cfg)
    d = PlatformConfig()
    for k in d.__dataclass_fields__:
        assert getattr(d, k) == getattr(plat, k), k
    assert scen.deadline_u == 10.0 and scen.deadline_s == 30.0 and scen.n_farms == 200
    ga = cfg["scheduler"]["ga"]
    assert (GeneticScheduler.pop, GeneticScheduler.gens, GeneticScheduler.batch_interval) == (
        ga["population"], ga["generations"], ga["batch_interval_s"])
    assert MinMin.batch_interval == cfg["scheduler"]["minmin"]["batch_interval_s"]


def test_overrides():
    scen, plat, fk = build(load_default(), {"rho": 1.4, "n_cloud": 8, "furrow.tau_r": 0.5})
    assert scen.rho == 1.4 and plat.n_cloud == 8 and fk["tau_r"] == 0.5
    with pytest.raises(KeyError):
        build(load_default(), {"no_such_parameter": 1})


def test_stats_helpers():
    m, lo, hi, n = mean_ci([1, 2, 3, 4])
    assert m == 2.5 and lo < m < hi and n == 4
    med, p, r = paired_test([1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0])
    assert r == 1.0 and p < 0.05
    assert list(holm([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])


@pytest.mark.skipif(not (ROOT / "data/processed/X.npy").exists() or
                    not (ROOT / "results/calibration/strategic_library.csv").exists(),
                    reason="processed data / calibration not available")
def test_small_end_to_end_run_is_paired_and_consistent():
    from furrow.data.pipeline import load
    from furrow.simulator.run import Library, simulate
    b = load(ROOT / "data/processed")
    lib = Library(ROOT / "results/calibration")
    scen, plat, fk = build(load_default(), {"n_farms": 20, "window_s": 60.0, "rho": 0.8})
    r1 = simulate(b, lib, scen, plat, "EDF", seed=3, c_eval=5e-7)
    r2 = simulate(b, lib, scen, plat, "FURROW-S", seed=3, c_eval=5e-7, sched_kwargs=fk)
    # identical workload for the same seed
    assert r1.summary["n_strategic"] == r2.summary["n_strategic"]
    assert r1.summary["n_chains"] == r2.summary["n_chains"] == 20
    for r in (r1, r2):
        t = r.tasks
        done = t[np.isfinite(t.finish)]
        assert (done.finish >= done.start - 1e-9).all() and (done.start >= done.arrival - 1e-9).all()
        assert 0.0 <= r.summary["chain_miss"] <= 1.0
        assert r.summary["energy_dyn_kJ"] > 0
    # baselines never use the summary plan
    assert r1.summary["strat_summary_frac"] == 0.0
