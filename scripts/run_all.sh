#!/usr/bin/env bash
# Full reproduction pipeline. Usage: bash scripts/run_all.sh [stage ...]
# Stages: data calib A B C D E F artifacts (default: all, in this order)
set -euo pipefail
cd "$(dirname "$0")/.."
STAGES=${@:-data calib A B C D E F artifacts}
mkdir -p results/logs
for s in $STAGES; do
  echo "=== stage $s: $(date)"
  case $s in
    data)  python3 scripts/download_data.py && python3 scripts/prepare_data.py ;;
    calib) python3 scripts/calibrate_costs.py && python3 scripts/measure_scheduler_overhead.py ;;
    A)     python3 scripts/run_retrieval_experiments.py --seeds 30 ;;
    B)     python3 scripts/run_simulation_grid.py B_scheduling ;;
    C)     python3 scripts/run_simulation_grid.py C_integrated ;;
    D)     python3 scripts/run_simulation_grid.py D_ablation ;;
    E)     python3 scripts/run_simulation_grid.py E_sensitivity ;;
    F)     python3 scripts/run_simulation_grid.py F_robustness ;;
    artifacts) python3 scripts/generate_paper_artifacts.py ;;
  esac 2>&1 | tee results/logs/stage_$s.log
done
