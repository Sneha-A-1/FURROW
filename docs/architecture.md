# FURROW architecture

Source diagrams: `paper/diagrams/fig_architecture.tex` (layers) and `paper/diagrams/fig_dataflow.tex`
(dataflow and control loop); rendered PNGs are in `results/figures/`.

| Layer | Implemented in | Inputs | Outputs |
|---|---|---|---|
| L1 Acquisition | `data/ingest.py`, `simulator/reporting.py` | CAF readings and weather; station reporting schedules | daily readings delivered every P_s days; on-demand polls |
| L2 Ingestion & QC | `data/qc.py` | delivered readings | online flags (range, spike, flat-line) |
| L3 Data management | `data/store.py`, `data/metadata.py`, `retrieval/triage.py::EvidenceStore/StalenessModel` | QC'd readings | warm tier (station-month Parquet), summary tier (weekly), hot tier (gateway), index (FC, LL, sigma, climatology) |
| L4 Retrieval & prioritisation (FURROW-R) | `retrieval/triage.py`, `retrieval/strategic.py`, `retrieval/verify.py` | delivered state, index, slack estimate | polled stations, ranked units, alerts, U1/U2/S tasks with retrieval variants, forecast n_f |
| L5 Task queues & scheduling (FURROW-S) | `scheduling/furrow.py` (+ baselines) | tasks, forecasts, core state | placements (core, variant, start) |
| L6 Execution | `simulator/engine.py`, `simulator/platform.py` | placements | completions, busy time, energy, cost |
| L7 Analytics & insight delivery | `retrieval/verify.py`, `retrieval/strategic.py` | executed tasks | advisories (days-to-critical), reports, analogues |
| L8 Monitoring & feedback | `simulator/run.py::_collect`, engine EFT state | events | metrics; EFT state used for the polling budget; forecasts used for reservation |

## Interfaces (implemented and evaluated)

1. **L4 -> L5 forecast**: at data arrival FURROW-R computes the prior expected number of alert
   stations n_f (Eq. 13 in the paper) and calls `scheduler.notify_forecast(farm, n_f, deadline)`;
   cleared when the farm's U1 completes.
2. **L5 -> L4 slack**: before polling, FURROW-R queries the scheduler state for the earliest
   finish time of a U1 and of a mean U2 task and converts the remaining slack into a polling budget.
3. **L4 -> L1 refresh**: polled stations return their current local reading (latency
   ceil(|S|/c) x l_poll).
4. **L3 -> L5 variants**: each strategic task carries two retrieval variants (full/summary) with
   measured costs and error; FURROW-S selects one at dispatch.
5. **L1 schedule -> L5**: published upload times let FURROW-S use a farm gateway for strategic
   work only when it will be free before the farm's next upload.

No urgency signal is counted twice: the forecast only reserves capacity; task ordering uses
deadlines/laxity only; the polling budget uses only the scheduler's finish-time estimate.

## Conceptual components outside the evaluated scope
Authentication, multi-tenant isolation, persistent message queues and real network stacks are
not modelled; their latency is not included for any method.
