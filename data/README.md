# Data

## Primary data set: R.J. Cook Agronomy Farm (CAF) sensor network

| Item | Value |
|---|---|
| Name | Cook Agronomy Farm data set (`cookfarm`) |
| Origin | R.J. Cook Agronomy Farm, Washington State University, Pullman, WA, USA (USDA Long-Term Agroecosystem Research site) |
| Primary references | Gasch et al., Spatial Statistics 14:70-90 (2015), doi:10.1016/j.spasta.2015.04.001; Gasch et al., Water Resources Research 53:10878-10887 (2017), doi:10.1002/2017WR021307 |
| Distribution used | `data/cookfarm.rda` of the CRAN package **landmap 0.0.13** (GPL-3), read-only GitHub mirror `https://raw.githubusercontent.com/cran/landmap/master/data/cookfarm.rda` |
| SHA-256 | `59ac08cebff116aceb74a970529fdf066ef13608eb182a630010bfd87b768669` |
| Format | R data file (xz-compressed list of data frames), parsed with the Python package `rdata` |
| Content used | `readings`: 42 stations x 5 depths (0.3, 0.6, 0.9, 1.2, 1.5 m) x daily volumetric water content (m3 m-3), soil temperature (deg C), bulk EC (dS m-1), 2011-01-02 to 2012-12-30; `weather`: daily precipitation, Tmax, Tmin (729 days); `profiles`: station coordinates and soil taxon; `grids`: 10 m covariates (DEM, TWI, NDRE, crop 2011/2012) |
| Size (verified) | 30 504 station-days; 153 090 unit-days after completing the station x depth x day grid; 107 990 readings present, 45 100 missing (29.5 %) |
| Coverage | 37 ha dryland farm, Palouse region; 2 years (2011 calibration, 2012 evaluation) |
| Licence | the landmap package is GPL-3; its documentation links the original data to the USDA National Agricultural Library Ag Data Commons record "Data from: A field-scale sensor network data set for monitoring and modeling the spatial and temporal variation of soil moisture in a dryland agricultural field". The licence of that original record was not verified here and should be checked before any redistribution. This repository does not redistribute the raw file: `scripts/download_data.py` fetches and verifies it. |
| Quality issues | sensor outages (median gap 8 days, 90th percentile 64 days); 83 readings rejected by two-sided spike screening; one unit (CAF119, 1.5 m) has no readings |
| Role | real agricultural workload and ground truth for urgent triage (Experiment A) and for the integrated simulations; real inputs for the strategic query templates and the measured cost library |

## Acquisition plan (as executed)

1. `python scripts/download_data.py` downloads `cookfarm.rda` to `data/raw/` and verifies the SHA-256 checksum.
2. `python scripts/prepare_data.py` parses, reshapes, runs online/offline QC, builds the tiered store
   (`data/processed/warm/station=*/YYYY-MM.parquet`, `data/processed/summary/summary_YYYY.parquet`), the
   metadata index (FC/LL thresholds) and the staleness model (calibration year 2011 only).

If the GitHub mirror is unreachable, obtain `data/cookfarm.rda` from a source package of
landmap 0.0.13 on CRAN (current or archived) and place it in `data/raw/`; the checksum
check guarantees that the identical file is used.

