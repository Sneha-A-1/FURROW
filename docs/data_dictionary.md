# Data dictionary

## data/processed (created by scripts/prepare_data.py)

| File | Shape / columns | Meaning |
|---|---|---|
| `X.npy` | 210 x 729 float | volumetric water content (m3 m-3) per unit (station x port) and day; NaN = no reading |
| `F_on.npy` | 210 x 729 int8 | online (causal) QC flag: -1 none, 0 pass, 1 suspect, 2 fail |
| `F_off.npy` | 210 x 729 int8 | offline (two-sided) QC flag used only to build the evaluation reference |
| `units.csv` | station, port, fc, ll, easting, northing | unit id = row index; FC = P95, LL = P2 of 2011 passed readings (pooled per port if < 120 days) |
| `dates.csv` | date | day index -> calendar date (2011-01-02 ... 2012-12-30) |
| `weather.csv` | date, precip_mm, tmax_c, tmin_c | daily weather of the nearest station (WRCC) |
| `stations.csv` | station, easting, northing, soil_taxon, dem, twi, ndre_mean, crop_2011, crop_2012, grid_dist_m | station metadata (nearest 10 m grid cell) |
| `stale_sigma.npy` | 5 x 12 x 16 | RMS change of VWC over a age a = 0..14 and >14 days, per port and month (2011) |
| `stale_clim.npy` | 5 x 12 x 2 | climatological mean and sd of D per port and month (2011) |
| `warm/station=S/YYYY-MM.parquet` | date, station, vw_p1..5, temp_c_p1..5, ec_p1..5, qc_p1..5 | warm tier (one partition per station-month) |
| `summary/summary_YYYY.parquet` | station, port, week, vw_mean, vw_min, vw_max, n, tmax_mean, tmin_mean, precip_sum | summary tier (weekly aggregates; weeks start Monday) |
| `warm_catalog.csv`, `summary_catalog.csv` | path, bytes, ... | partition catalogues with measured file sizes |
| `dataset_stats.json` | - | verified counts reported in the paper |

Ports: 1 = 0.3 m, 2 = 0.6 m, 3 = 0.9 m, 4 = 1.2 m, 5 = 1.5 m. Root zone = ports 1-2.

## results/calibration (scripts/calibrate_costs.py, scripts/measure_scheduler_overhead.py)

| File | Columns | Meaning |
|---|---|---|
| `strategic_library.csv` | qid, template, t_end, window, n_stations, target, cpu_full, bytes_full, parts_full, cpu_summary, bytes_summary, parts_summary, error_summary | 300 strategic query instances executed on the real data at both fidelities; CPU s = median of 3 runs; error = normalised error of summary answer vs exact answer |
| `verify_costs.csv` | unit, t_end, cpu, bytes, parts | 200 U2 verification executions |
| `triage_costs.csv` | method, t, cpu | U1 triage CPU time per method |
| `c_eval.json` | c_eval_s | wall-clock cost of one EFT evaluation (scheduler overhead constant) |
| `machine.json` | - | measurement host and library versions |

## results/raw

| File | Row = | Key columns |
|---|---|---|
| `retrieval_decisions.csv.gz` | (scenario, seed, day, method) | ndcg10, p10, rprec, mrr3, alert_p, alert_r, tp, fp, fn, errors, polls, bytes, cpu_s |
| `B_scheduling.csv`, `C_integrated.csv`, `D_ablation.csv`, `E_sensitivity.csv`, `F_robustness.csv` | one simulation run | label, scheduler, retrieval, seed, factor, level, chain_miss, chain_lat_mean, chain_lat_p95, urgent_resp_mean, urgent_resp_p95, strat_miss, strat_resp_mean, strat_summary_frac, strat_error_mean, alert_precision, alert_recall, timely_recall, polls, energy_dyn_kJ, cost, util_shared, sched_overhead_s, throughput, wall_s |
| `B_scheduling_tasks.csv.gz` | one task of the rho = 1.4, seed 0 runs | cls, arrival, start, finish, resp, miss, fid, ntype, label |

All times are seconds of simulated time unless named `wall_s` / `cpu_s` (host wall clock).
