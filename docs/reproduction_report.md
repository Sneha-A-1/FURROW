# Reproduction report

## What was implemented
* Data layer (L1-L3): CAF ingestion with SHA-256 check, online/offline QC, tiered Parquet store (1,008 warm partitions, 3 summary partitions), metadata index and staleness model fitted on 2011 only.
* FURROW-R: persistence prior with calibrated staleness variance, expected-gain ranking, value-of-information station refresh with v_min stop and slack-derived budget, alert forecast.
* FURROW-S: laxity-ordered urgent placement, forecast-driven reservation, plan-aware (full/summary) strategic placement, opportunistic gateway use bounded by the upload schedule.
* Baselines: retrieval THR, FRESH, IDW, AoI, RND, ALL; scheduling FCFS, RR, SJF, EDF, PRIO, MCT, Min-Min, GA.
* Discrete-event simulator of a 200-farm edge-fog-cloud platform with measured task costs (300 strategic query instances, 200 verification instances, triage CPU times) and a measured scheduler-overhead constant.
* 25 unit/integration tests (all pass).

## Data
Cook Agronomy Farm sensor network (Gasch et al., 2015; 2017) from CRAN landmap 0.0.13; 107,990 VWC readings present and 45,100 missing on the 210 x 729 unit-day grid. No other data were downloaded; infrastructure parameters are documented assumptions.

## Experiments completed (all with raw outputs in results/raw)
| Exp. | Content | Runs |
|---|---|---|
| A | retrieval quality, 3 connectivity scenarios x 30 seeds x 272 days x 20 method configurations | 489,600 decisions |
| B | 9 schedulers x 7 loads x 20 seeds | 1,260 |
| C | 8 retrieval/scheduling stacks x 2 loads x 2 urgent deadlines x 20 seeds | 640 |
| D | 10 ablation variants x 20 seeds | 200 |
| E | 9 one-factor sweeps x 3 stacks x 10 seeds | 930 |
| F | data loss, bursts, noise, scale to 800 farms x 3 stacks x 10 seeds | 390 |
| pilot | rejected nowcasting variants (drift, surrogate) | documented |

## Main results supporting the contributions
* C1 (retrieval): with 8 polls, alert errors per decision 0.058 vs 0.134 (AoI) and 0.643 (THR) under mixed connectivity (57 % fewer than AoI with 33 % fewer polls); poor connectivity 0.627 vs 0.836. Caveat: at budget 16 (mixed) AoI had fewer errors with twice the polls; AoI had slightly higher nDCG@10 at budget 8.
* C2 (scheduling): FURROW-S had 0 urgent-chain misses at every load (rho 0.6-1.8), strategic misses <= 0.5 %, dynamic energy up to 32 % below the most economical baseline, mean answer error <= 2.8 %. Caveat: FCFS/SJF had lower urgent p95 latency up to rho = 1.2 when retrieval does not poll.
* C3 (integration): reservation reduced urgent p95 from 2.46 s to 0.20 s; refresh with a conventional scheduler increased urgent misses (Table 9). Slack budget and laxity ordering had no effect at Delta_U = 10 s; edge opportunism traded energy for answer quality; FURROW lost its advantage under fully synchronised upload bursts.

## Commands
See docs/reproduction_protocol.md (`bash scripts/run_all.sh`; ~3 h on 2 cores), then `python paper/build_docx.py`.

## Limitations
Simulated infrastructure; single real farm replicated with time shifts; percentile-based agronomic thresholds on a non-irrigated farm; no failure model; learning-based schedulers not compared.

## Manuscript build
The journal supplies only a Word template, so the manuscript is a .docx built into the official template (`paper/ITEECS_manuscript_FURROW.docx`); equations are native Word equations. A PDF rendered with LibreOffice 24.2 is included for inspection (16 pages, about 12,000 words including references). No LaTeX version was produced because ITEECS accepts .docx/PDF and offers no LaTeX class.
