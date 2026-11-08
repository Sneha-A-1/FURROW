"""Measure c_eval: wall-clock cost of one estimated-finish-time evaluation of a
(task, core) pair on the measurement host. Every scheduler is charged
c_eval x (number of evaluations it performs) as decision overhead.

Output: results/calibration/c_eval.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from furrow.scheduling.baselines import MCT  # noqa: E402
from furrow.simulator.engine import Engine, Task  # noqa: E402
from furrow.simulator.platform import PlatformConfig, Variant, build_cores  # noqa: E402

if __name__ == "__main__":
    cfg = PlatformConfig()
    cores = build_cores(cfg, 200)
    eng = Engine(cfg, cores, MCT(), c_eval=0.0)
    rng = np.random.default_rng(0)
    tasks = [Task(i, "S", int(rng.integers(0, 200)), 0.0, 30.0,
                  {"full": Variant("full", "warm", 80, 1_100_000, 0.26)}, "full") for i in range(2000)]
    pairs = [(tasks[int(rng.integers(0, 2000))], cores[int(rng.integers(0, 24))]) for _ in range(200_000)]
    for t, c in pairs[:1000]:
        eng.eft(t, c)
    reps = []
    for _ in range(5):
        t0 = time.perf_counter()
        for t, c in pairs:
            eng.eft(t, c)
        reps.append((time.perf_counter() - t0) / len(pairs))
    c_eval = float(np.median(reps))
    out = ROOT / "results/calibration/c_eval.json"
    out.write_text(json.dumps({"c_eval_s": c_eval, "repetitions_s": reps}, indent=2))
    print(json.dumps({"c_eval_s": c_eval}))
