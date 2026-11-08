# ITEECS submission-readiness checklist

Status legend: [x] done and verified in this package; [ ] action required by the authors; [~] partially verified / caveat.

## Journal and scope
- [x] Scope alignment: ITEECS lists cloud computing, databases and data mining, data science, soft computing and machine learning (journal "About" page, accessed 22 Oct 2026). The article is a cloud/data-management/scheduling contribution with an agricultural application.
- [x] Format: the journal accepts Microsoft Word (.docx) or PDF submissions; no LaTeX template is offered. The manuscript is therefore delivered in the official Word template supplied (`paper/journal_template/Template-ITEECS.docx`).
- [x] Abstract 150-250 words without citations; keywords <= 10 and not repeating title words.
- [~] Length: the template states that research articles "should not be more than 10 pages" and that extra pages are charged; the journal website requires at least 6,000 words. The manuscript exceeds 10 template pages (see `paper/build_report.json` / rendered PDF page count). Article processing charges (50 USD) are waived until June 2027 according to the submission page; confirm with the editorial office whether the page limit is enforced, or move Tables 1-2 and the sensitivity figure to supplementary material.
- [x] Similarity limit (<= 20 % overall, <= 5 % single source) noted; text written from scratch for this study. Run iThenticate/Turnitin before submission.
- [ ] Cover letter (required as a supplementary file) to be written by the authors.

## Indexing verification
- [x] ISSN 2583-6471 confirmed on the ISSN portal record and in DOAJ (journal listed in DOAJ; DOAJ shows an APC of up to 50 USD with a waiver policy).
- [~] Scopus: claimed on the journal website; **not independently verified** - SCImago returned no record for ISSN 2583-6471 on 22 Oct 2026 and the Scopus source list requires an institutional login. Verify in Scopus Sources before relying on this claim.
- [x] Crossref DOI prefix 10.62760 is claimed by the journal (not needed for submission).

## Authorship and declarations
- [ ] Confirm author order (Sneha, Prabh Deep Singh, Vikas Tripathi), affiliations and e-mail addresses.
- [ ] Designate the corresponding author and complete the corresponding-author block (placeholder in the History box).
- [ ] Insert ORCID iDs (the template requires ORCID or Scopus links; the ORCID icons are present without links).
- [ ] Complete the CRediT statement, Funding statement and Conflict-of-Interest statement (placeholders, not invented).
- [ ] All authors approve the final manuscript.

## Ethics and originality
- [x] No human or animal subjects; public agricultural sensor data only.
- [x] No fabricated data: every reported number is produced by scripts from raw run files (`results/raw`), see `results/processed/paper_values.json`.
- [x] Negative results documented (rejected drift and surrogate nowcasting variants: `scripts/pilot_rejected_variants.py`).

## References
- [x] 42 references, each verified against Crossref, OpenAlex, arXiv, DOAJ, PubMed Central or dblp records (`paper/reference_ledger.md` lists the source used for each).
- [x] Every in-text citation resolves to a bibliography entry and every entry is cited (enforced by `paper/build_docx.py`, which numbers references in order of first citation).
- [~] Two preprints (arXiv) are cited and labelled as such.

## Data and code
- [x] Data source, checksum, licence notes and acquisition steps in `data/README.md`; raw data are not redistributed.
- [x] Repository with code, configurations, tests, raw outputs and figure sources; `pytest` passes.
- [ ] Create the public repository (e.g. GitHub/Zenodo DOI) and insert its URL in the Data Availability Statement.

## Figures and tables
- [x] Figures generated from code at 600 dpi (PNG) with vector PDF copies; diagrams from TikZ sources; consistent fonts (Palatino-compatible).
- [x] Result tables contain single point estimates (no ± symbols); confidence intervals and tests in `results/processed/stats_*.csv`.
- [x] Every figure and table is referenced in the text.

## Experimental consistency
- [x] Manuscript numbers are inserted from `paper_values.json` / result tables generated from raw outputs.
- [x] Infrastructure parameters are documented as assumptions and varied in sensitivity analyses.
- [~] Simulation (not a physical deployment); the replicated single-farm workload and infrastructure assumptions are stated as threats to validity.

## Final verification
- [x] Manuscript built from sources and rendered to PDF for inspection (LibreOffice); final visual check in Microsoft Word recommended because Word and LibreOffice place floats and equations slightly differently.
