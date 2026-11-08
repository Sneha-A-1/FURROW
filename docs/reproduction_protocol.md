# Reproduction protocol

## Environment
* Python >= 3.10 (results were produced with Python 3.13.16, NumPy 2.5.3, pandas 3.0.5,
  SciPy 1.18.1, PyArrow 26.0.0, Matplotlib 3.11.2; see `results/calibration/machine.json`).
* `pip install -r requirements.txt` (or `conda env create -f environment/environment.yml`).
* Paper build (optional): pandoc >= 3.1, LibreOffice (PDF rendering), TeX Live with TikZ (diagrams).

## Order of execution
```
bash scripts/run_all.sh                    # everything below, in order
python scripts/download_data.py            # fetch + SHA-256 check of cookfarm.rda
python scripts/prepare_data.py             # QC, tiers, metadata index (2011 only)
python scripts/calibrate_costs.py          # measured strategic/verification/triage costs
python scripts/measure_scheduler_overhead.py
python scripts/run_retrieval_experiments.py --seeds 30          # Experiment A
python scripts/run_simulation_grid.py B_scheduling              # Experiment B
python scripts/run_simulation_grid.py C_integrated              # Experiment C
python scripts/run_simulation_grid.py D_ablation                # Experiment D
python scripts/run_simulation_grid.py E_sensitivity             # Experiment E
python scripts/run_simulation_grid.py F_robustness              # Experiment F
python scripts/generate_paper_artifacts.py                      # tables, figures, statistics
python scripts/pilot_rejected_variants.py                       # documented negative result
python paper/build_docx.py                                      # manuscript (.docx)
python -m pytest -q                                             # unit and integration tests
```

## Randomness
* Experiment A: seed k draws the station reporting schedule (`numpy.random.default_rng(k)`);
  tie breaking in rankings uses `default_rng(seed*7919 + day)`; random refresh uses
  `default_rng(seed*100003 + day)`.
* Simulations: for seed k the workload (run day, farm shifts, schedules, upload times, strategic
  arrivals and their library rows, execution noise) is drawn from `default_rng(k + 10000)`,
  `default_rng(k + 30000)` and per-chain generators, so every scheduler sees an identical workload
  for the same seed (paired design). The GA uses its own fixed generator (seed 12345).

## Determinism caveat
Measured CPU times in `results/calibration/` depend on the host. Re-running the calibration on
different hardware changes absolute latencies; all other steps are deterministic given those
files. The released calibration files are the ones used for the paper.

## Number of repetitions
Experiment A uses 30 schedule draws x 272 decision days per scenario. Simulation experiments use
20 paired seeds (B-D) and 10 (E-F). `results/processed/stats_*_summary.csv` report 95 %
Student-t confidence intervals; `stats_*_tests.csv` report Wilcoxon signed-rank p-values with
Holm correction and rank-biserial effect sizes.

## Expected outputs and validation checks
* `data/processed/dataset_stats.json`: 107 990 readings present, 45 100 missing, 83 offline rejections.
* `pytest`: all tests pass.
* `results/raw/*.csv` row counts: B 1260, C 640, D 200, E 930, F 390 runs.
* `paper/build_report.json`: every citation key resolved against `paper/reference_ledger.md`.
