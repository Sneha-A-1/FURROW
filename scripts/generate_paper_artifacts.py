"""Generate every table, figure and quoted number of the paper from raw results.

Inputs : results/raw/*.csv[.gz], results/calibration/*, data/processed/dataset_stats.json
Outputs: results/tables/*.csv, results/figures/*.png|pdf, results/processed/stats_*.csv,
         results/processed/paper_values.json
Usage  : python scripts/generate_paper_artifacts.py [--only A,B,...]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from furrow.evaluation.stats import compare, mean_ci, paired_test, summarise  # noqa: E402

RAW, PROC, TAB, FIG = (ROOT / "results" / d for d in ("raw", "processed", "tables", "figures"))
for d in (PROC, TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)
VALUES: dict = {}

# ------------------------------------------------------------------ style
_font = "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella-regular.otf"
if Path(_font).exists():
    for f in Path(_font).parent.glob("texgyrepagella-*.otf"):
        fm.fontManager.addfont(str(f))
    FAMILY = "TeX Gyre Pagella"
else:
    FAMILY = "serif"
plt.rcParams.update({
    "font.family": FAMILY, "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "lines.markersize": 4, "figure.dpi": 150, "savefig.dpi": 600,
    "legend.frameon": False, "mathtext.fontset": "stix",
})
# fixed categorical order (validated palette, light surface); colour follows the entity
C = {"FURROW": "#2a78d6", "PRIO": "#eb6834", "EDF": "#1baf7a", "MCT": "#eda100",
     "GA": "#e87ba4", "MINMIN": "#008300", "FCFS": "#6250d6", "RR": "#e34948", "SJF": "#52514e"}
CR = {"FURROW-R": "#2a78d6", "AoI": "#eb6834", "RND": "#1baf7a", "THR": "#eda100", "ALL": "#e87ba4",
      "IDW": "#008300", "FRESH": "#6250d6"}
MK = {"FURROW": "o", "PRIO": "s", "EDF": "^", "MCT": "v", "GA": "D", "MINMIN": "P", "FCFS": "X",
      "RR": "*", "SJF": "h"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def sched_key(label: str) -> str:
    return "FURROW" if label.startswith("FURROW") else label


# ------------------------------------------------------------------ A: retrieval
def exp_a():
    d = pd.read_csv(RAW / "retrieval_decisions.csv.gz")
    per_seed = d.groupby(["scenario", "label", "method", "budget", "seed"]).agg(
        ndcg10=("ndcg10", "mean"), rprec=("rprec", "mean"), mrr3=("mrr3", "mean"),
        tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"), errors=("errors", "mean"),
        polls=("polls", "mean"), bytes=("bytes", "mean"), cpu_ms=("cpu_s", lambda x: 1000 * x.mean()),
        decisions=("day", "count")).reset_index()
    per_seed["precision"] = per_seed.tp / (per_seed.tp + per_seed.fp)
    per_seed["recall"] = per_seed.tp / (per_seed.tp + per_seed.fn)
    per_seed["f1"] = 2 * per_seed.precision * per_seed.recall / (per_seed.precision + per_seed.recall)
    per_seed.to_csv(PROC / "retrieval_per_seed.csv", index=False)
    metrics = ["ndcg10", "rprec", "mrr3", "precision", "recall", "f1", "errors", "polls", "bytes", "cpu_ms"]
    summ = summarise(per_seed, ["scenario", "label"], metrics)
    summ.to_csv(PROC / "stats_retrieval_summary.csv", index=False)
    order = ["THR", "FRESH", "IDW", "FURROW-R", "RND@8", "AoI@4", "FURROW-R@4", "AoI@8", "FURROW-R@8",
             "AoI@16", "FURROW-R@16", "ALL"]
    rows = []
    for sc in ("mixed", "poor"):
        s = summ[summ.scenario == sc].set_index("label")
        for lab in order:
            r = s.loc[lab]
            rows.append({"Scenario": sc, "Method": lab.replace("FURROW-R", "FURROW-R@0") if lab == "FURROW-R" else lab,
                         "nDCG@10": f"{r.ndcg10:.4f}", "R-prec (g>=2)": f"{r.rprec:.4f}",
                         "Alert precision": f"{r.precision:.4f}", "Alert recall": f"{r.recall:.4f}",
                         "Alert errors per decision": f"{r.errors:.3f}", "Polls per decision": f"{r.polls:.2f}",
                         "CPU (ms)": f"{r.cpu_ms:.2f}"})
    pd.DataFrame(rows).to_csv(TAB / "T_retrieval.csv", index=False)
    # paired tests: FURROW-R@B vs AoI@B and vs RND@B, per scenario (errors and polls)
    tests = []
    for sc in ("mixed", "poor"):
        ps = per_seed[per_seed.scenario == sc]
        fam = []
        for B, other in [(4, "AoI@4"), (8, "AoI@8"), (16, "AoI@16"), (8, "RND@8"), (0, "THR")]:
            f = ps[ps.label == (f"FURROW-R@{B}" if B else "FURROW-R")].set_index("seed")
            o = ps[ps.label == other].set_index("seed")
            for m in ("errors", "polls", "ndcg10", "rprec"):
                med, p, r = paired_test(f[m], o[m])
                fam.append(dict(scenario=sc, ref=f"FURROW-R@{B}", other=other, metric=m,
                                ref_mean=f[m].mean(), other_mean=o[m].mean(), median_diff=med, p=p,
                                rank_biserial=r))
        from furrow.evaluation.stats import holm
        adj = holm([x["p"] for x in fam])
        for x, a in zip(fam, adj):
            x["p_holm"] = a
        tests += fam
    pd.DataFrame(tests).to_csv(PROC / "stats_retrieval_tests.csv", index=False)
    # v_min sensitivity
    vm = summ[summ.label.str.contains("vmin") | (summ.label == "FURROW-R@8")].copy()
    vm["v_min"] = vm.label.str.extract(r"vmin=([0-9.]+)")[0].astype(float).fillna(0.05)
    vm[["scenario", "v_min", "errors", "polls", "ndcg10", "rprec"]].sort_values(["scenario", "v_min"]).to_csv(
        TAB / "T_retrieval_vmin.csv", index=False)
    VALUES["A_vmin"] = {sc: {str(r.v_min): {"errors": float(r.errors), "polls": float(r.polls)}
                             for r in vm[vm.scenario == sc].itertuples()} for sc in ("daily", "mixed", "poor")}

    # figure: errors vs polls
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.85), sharey=False)
    for ax, sc in zip(axes, ("mixed", "poor")):
        s = summ[summ.scenario == sc].set_index("label")
        for meth, budgets, mk in (("FURROW-R", [0, 2, 4, 8, 16], "o"), ("AoI", [2, 4, 8, 16], "s"),
                                  ("RND", [4, 8], "^")):
            labs = [("FURROW-R" if b == 0 else f"{meth}@{b}") if meth == "FURROW-R" else f"{meth}@{b}" for b in budgets]
            xs = [s.loc[lb, "polls"] for lb in labs]
            ys = [s.loc[lb, "errors"] for lb in labs]
            ax.plot(xs, ys, marker=mk, color=CR[meth], label=meth, markeredgecolor="white", markeredgewidth=0.5)
        ax.plot([s.loc["THR", "polls"]], [s.loc["THR", "errors"]], marker="D", ls="none", color=CR["THR"], label="THR")
        ax.plot([s.loc["ALL", "polls"]], [s.loc["ALL", "errors"]], marker="*", ms=7, ls="none", color=CR["ALL"],
                label="ALL")
        ax.set_title(f"{sc} connectivity")
        ax.set_xlabel("stations polled per decision")
        ax.set_ylim(bottom=-0.05)
    axes[0].set_ylabel("alert errors per decision")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08), handlelength=1.5, columnspacing=0.8)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig3_retrieval_tradeoff")
    for sc in ("mixed", "poor"):
        s = summ[summ.scenario == sc].set_index("label")
        VALUES[f"A_{sc}"] = {lab: {k: float(s.loc[lab, k]) for k in ("errors", "polls", "ndcg10", "rprec", "precision", "recall", "f1", "cpu_ms")}
                             for lab in order}
    VALUES["A_decisions_per_scenario"] = int(d[(d.scenario == "mixed") & (d.label == "THR")].shape[0] / d.seed.nunique())
    VALUES["A_seeds"] = int(d.seed.nunique())


# ------------------------------------------------------------------ static tables
def static_tables():
    st = json.loads((ROOT / "data/processed/dataset_stats.json").read_text())
    VALUES["dataset"] = st
    lib = pd.read_csv(ROOT / "results/calibration/strategic_library.csv")
    rows = []
    names = {"S1_water_status": "S1 water status", "S2_deficit_days": "S2 deficit days", "S3_spatial_cv": "S3 spatial CV",
             "S4_drying_rate": "S4 drying rate", "S5_analogues": "S5 analogue top-5"}
    for t, g in lib.groupby("template"):
        rows.append({"Template": names[t], "Instances": len(g),
                     "CPU full (s)": f"{g.cpu_full.median():.3f}", "CPU summary (s)": f"{g.cpu_summary.median():.3f}",
                     "Partitions full": int(g.parts_full.median()), "Partitions summary": int(g.parts_summary.median()),
                     "MB full": f"{g.bytes_full.median() / 1e6:.2f}", "MB summary": f"{g.bytes_summary.median() / 1e6:.3f}",
                     "Summary error (mean)": f"{g.error_summary.mean():.3f}"})
    pd.DataFrame(rows).to_csv(TAB / "T_strategic_library.csv", index=False)
    v = pd.read_csv(ROOT / "results/calibration/verify_costs.csv")
    tc = pd.read_csv(ROOT / "results/calibration/triage_costs.csv")
    VALUES["calibration"] = {"verify_cpu_ms_mean": 1000 * v.cpu.mean(), "verify_kB_mean": v.bytes.mean() / 1e3,
                             "verify_parts_median": float(v.parts.median()),
                             "triage_cpu_ms": (tc.groupby("method").cpu.mean() * 1000).to_dict(),
                             "library_instances": int(len(lib)),
                             "summary_error_mean_by_template": lib.groupby("template").error_summary.mean().to_dict(),
                             "cpu_ratio_median": float((lib.cpu_full / lib.cpu_summary).median()),
                             "bytes_ratio_median": float((lib.bytes_full / lib.bytes_summary).median())}
    VALUES["c_eval_s"] = json.loads((ROOT / "results/calibration/c_eval.json").read_text())["c_eval_s"]


# ------------------------------------------------------------------ B: scheduling
SCHED_METRICS = ["chain_miss", "chain_lat_mean", "chain_lat_p95", "urgent_resp_mean", "urgent_resp_p95",
                 "strat_miss", "strat_resp_mean", "strat_resp_p95", "strat_summary_frac", "strat_error_mean",
                 "energy_dyn_kJ", "cost", "util_shared", "sched_overhead_s", "throughput", "timely_recall",
                 "alert_precision", "alert_recall", "polls", "wall_s"]


def _rho(df):
    df = df.copy()
    df["rho"] = df.overrides.map(lambda s: json.loads(s).get("rho"))
    return df


def exp_b():
    df = _rho(pd.read_csv(RAW / "B_scheduling.csv"))
    summ = summarise(df, ["rho", "label"], SCHED_METRICS)
    summ.to_csv(PROC / "stats_B_summary.csv", index=False)
    tests = compare(df, ["rho"], "label", "FURROW-S",
                    ["chain_miss", "urgent_resp_mean", "urgent_resp_p95", "strat_miss", "energy_dyn_kJ", "cost"])
    tests.to_csv(PROC / "stats_B_tests.csv", index=False)
    order = ["FCFS", "RR", "SJF", "EDF", "PRIO", "MCT", "MINMIN", "GA", "FURROW-S"]
    rows = []
    for rho in (1.0, 1.4, 1.8):
        s = summ[summ.rho == rho].set_index("label")
        for lab in order:
            r = s.loc[lab]
            rows.append({"rho": f"{rho:.1f}", "Scheduler": lab, "Chain miss (%)": f"{100 * r.chain_miss:.2f}",
                         "Urgent mean (s)": f"{r.urgent_resp_mean:.3f}", "Urgent p95 (s)": f"{r.urgent_resp_p95:.2f}",
                         "Strategic miss (%)": f"{100 * r.strat_miss:.2f}", "Strategic mean (s)": f"{r.strat_resp_mean:.2f}",
                         "Summary (%)": f"{100 * r.strat_summary_frac:.2f}", "Answer error (%)": f"{100 * r.strat_error_mean:.2f}",
                         "Energy (kJ)": f"{r.energy_dyn_kJ:.1f}", "Cost (units)": f"{r.cost:.4f}",
                         "Overhead (s)": f"{r.sched_overhead_s:.3f}"})
    pd.DataFrame(rows).to_csv(TAB / "T_scheduling.csv", index=False)
    show = ["FURROW-S", "PRIO", "EDF", "MCT", "MINMIN", "GA", "FCFS"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.9))
    panels = [("chain_miss", "urgent chain deadline misses (%)", 100, False),
              ("urgent_resp_p95", "urgent response time, p95 (s)", 1, True),
              ("strat_miss", "strategic deadline misses (%)", 100, False),
              ("energy_dyn_kJ", "dynamic energy per window (kJ)", 1, False)]
    for ax, (m, lab, sc, logy) in zip(axes.ravel(), panels):
        for name in show:
            s = summ[summ.label == name].sort_values("rho")
            key = sched_key(name)
            ax.plot(s.rho, sc * s[m], marker=MK[key], color=C[key], label=name,
                    markeredgecolor="white", markeredgewidth=0.5)
            if m in ("chain_miss", "strat_miss", "energy_dyn_kJ"):
                ax.fill_between(s.rho, sc * s[f"{m}_ci_lo"], sc * s[f"{m}_ci_hi"], color=C[key], alpha=0.12, lw=0)
        ax.set_ylabel(lab)
        ax.set_xlabel(r"offered strategic load $\rho$")
        if logy:
            ax.set_yscale("log")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(h_pad=0.8, w_pad=1.2)
    fig.subplots_adjust(top=0.91)
    save(fig, "fig4_scheduling_load")
    VALUES["B"] = {f"{r.label}@{r.rho}": {m: float(getattr(r, m)) for m in SCHED_METRICS} for r in summ.itertuples()}
    VALUES["B_runs"] = int(len(df))


# ------------------------------------------------------------------ C: integrated
def exp_c():
    df = pd.read_csv(RAW / "C_integrated.csv")
    ov = df.overrides.map(json.loads)
    df["rho"] = ov.map(lambda o: o["rho"])
    df["deadline_u"] = ov.map(lambda o: o["deadline_u"])
    summ = summarise(df, ["rho", "deadline_u", "label"], SCHED_METRICS)
    summ.to_csv(PROC / "stats_C_summary.csv", index=False)
    tests = compare(df, ["rho", "deadline_u"], "label", "FURROW",
                    ["chain_miss", "timely_recall", "alert_precision", "urgent_resp_mean", "strat_miss", "energy_dyn_kJ"])
    tests.to_csv(PROC / "stats_C_tests.csv", index=False)
    order = ["THR+PRIO", "THR+EDF", "AoI8+PRIO", "ALL+PRIO", "FURROW-R8+PRIO", "THR+FURROW-S", "AoI8+FURROW-S", "FURROW"]
    rows = []
    for (rho, du), s in summ.groupby(["rho", "deadline_u"]):
        s = s.set_index("label")
        for lab in order:
            r = s.loc[lab]
            rows.append({"rho": f"{rho:.1f}", "Delta_U (s)": f"{du:.0f}", "Configuration": lab,
                         "Chain miss (%)": f"{100 * r.chain_miss:.2f}", "Timely recall (%)": f"{100 * r.timely_recall:.2f}",
                         "Alert precision (%)": f"{100 * r.alert_precision:.2f}",
                         "Polls per farm": f"{r.polls / 200:.2f}", "Chain latency (s)": f"{r.chain_lat_mean:.2f}",
                         "Urgent mean (s)": f"{r.urgent_resp_mean:.3f}", "Urgent p95 (s)": f"{r.urgent_resp_p95:.2f}",
                         "Strategic miss (%)": f"{100 * r.strat_miss:.2f}", "Answer error (%)": f"{100 * r.strat_error_mean:.2f}",
                         "Energy (kJ)": f"{r.energy_dyn_kJ:.1f}"})
    pd.DataFrame(rows).to_csv(TAB / "T_integrated.csv", index=False)
    VALUES["C"] = {f"{r.label}@{r.rho}@{r.deadline_u}": {m: float(getattr(r, m)) for m in SCHED_METRICS}
                   for r in summ.itertuples()}


# ------------------------------------------------------------------ D: ablation
def exp_d():
    df = pd.read_csv(RAW / "D_ablation.csv")
    summ = summarise(df, ["label"], SCHED_METRICS)
    summ.to_csv(PROC / "stats_D_summary.csv", index=False)
    tests = compare(df.assign(g=1), ["g"], "label", "FURROW",
                    ["chain_miss", "timely_recall", "alert_precision", "urgent_resp_mean", "urgent_resp_p95",
                     "strat_miss", "strat_error_mean", "energy_dyn_kJ", "polls"])
    tests.to_csv(PROC / "stats_D_tests.csv", index=False)
    order = ["FURROW", "-VoI (AoI refresh)", "-refresh (THR)", "-slack budget", "-laxity order", "-reservation",
             "-forecast (EWMA)", "-fidelity", "-locality", "-edge opportunism"]
    s = summ.set_index("label")
    rows = []
    for lab in order:
        r = s.loc[lab]
        rows.append({"Variant": lab, "Chain miss (%)": f"{100 * r.chain_miss:.2f}",
                     "Timely recall (%)": f"{100 * r.timely_recall:.2f}", "Alert precision (%)": f"{100 * r.alert_precision:.2f}",
                     "Polls per farm": f"{r.polls / 200:.2f}", "Urgent mean (s)": f"{r.urgent_resp_mean:.3f}",
                     "Urgent p95 (s)": f"{r.urgent_resp_p95:.2f}", "Strategic miss (%)": f"{100 * r.strat_miss:.2f}",
                     "Answer error (%)": f"{100 * r.strat_error_mean:.2f}", "Energy (kJ)": f"{r.energy_dyn_kJ:.1f}"})
    pd.DataFrame(rows).to_csv(TAB / "T_ablation.csv", index=False)
    VALUES["D"] = {r.label: {m: float(getattr(r, m)) for m in SCHED_METRICS} for r in summ.itertuples()}


# ------------------------------------------------------------------ E: sensitivity
def exp_e():
    df = pd.read_csv(RAW / "E_sensitivity.csv")
    summ = summarise(df, ["factor", "level", "label"], SCHED_METRICS)
    summ.to_csv(PROC / "stats_E_summary.csv", index=False)
    tests = compare(df, ["factor", "level"], "label", "FURROW", ["chain_miss", "timely_recall", "strat_miss", "energy_dyn_kJ"])
    tests.to_csv(PROC / "stats_E_tests.csv", index=False)
    keep = ["chain_miss", "timely_recall", "urgent_resp_p95", "strat_miss", "strat_error_mean", "energy_dyn_kJ", "polls"]
    out = summ[["factor", "level", "label"] + keep].copy()
    out.to_csv(TAB / "T_sensitivity.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.7))
    panels = [("deadline_u", "chain_miss", 100, r"urgent deadline $\Delta_U$ (s)", "urgent chain misses (%)"),
              ("deadline_u", "timely_recall", 100, r"urgent deadline $\Delta_U$ (s)", "timely severe-unit recall (%)"),
              ("deadline_s", "strat_miss", 100, r"strategic deadline $\Delta_S$ (s)", "strategic deadline misses (%)"),
              ("wan_rtt", "strat_miss", 100, "gateway-cloud round-trip time (s)", "strategic deadline misses (%)")]
    for ax, (fac, m, sc, xl, yl) in zip(axes.ravel(), panels):
        for lab in ("FURROW", "THR+PRIO", "THR+EDF"):
            s = summ[(summ.factor == fac) & (summ.label == lab)].copy()
            s["x"] = s.level.astype(float)
            s = s.sort_values("x")
            key = "FURROW" if lab == "FURROW" else lab.split("+")[1]
            ax.plot(s.x, sc * s[m], marker=MK[key], color=C[key], label=lab, markeredgecolor="white", markeredgewidth=0.5)
            ax.fill_between(s.x, sc * s[f"{m}_ci_lo"], sc * s[f"{m}_ci_hi"], color=C[key], alpha=0.12, lw=0)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(h_pad=0.8, w_pad=1.2)
    fig.subplots_adjust(top=0.91)
    save(fig, "fig5_sensitivity")
    VALUES["E"] = {f"{r.factor}={r.level}|{r.label}": {m: float(getattr(r, m)) for m in keep} for r in summ.itertuples()}


# ------------------------------------------------------------------ F: robustness and scalability
def exp_f():
    df = pd.read_csv(RAW / "F_robustness.csv")
    summ = summarise(df, ["factor", "level", "label"], SCHED_METRICS + ["n_tasks"])
    summ.to_csv(PROC / "stats_F_summary.csv", index=False)
    tests = compare(df, ["factor", "level"], "label", "FURROW", ["chain_miss", "timely_recall", "strat_miss", "energy_dyn_kJ"])
    tests.to_csv(PROC / "stats_F_tests.csv", index=False)
    keep = ["chain_miss", "timely_recall", "alert_precision", "urgent_resp_p95", "strat_miss", "strat_error_mean",
            "energy_dyn_kJ", "polls", "sched_overhead_s", "n_tasks", "wall_s"]
    summ[["factor", "level", "label"] + keep].to_csv(TAB / "T_robustness.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.95))
    for ax, (fac, m, sc, xl, yl) in zip(axes, [
            ("extra_missing", "timely_recall", 100, "extra readings removed (fraction)", "timely recall (%)"),
            ("noise_sigma", "chain_miss", 100, r"execution-time noise $\sigma$", "urgent chain misses (%)"),
            ("scale", "sched_overhead_s", 1, "farms (shared cores scaled)", "scheduler overhead per window (s)")]):
        for lab in ("FURROW", "THR+PRIO", "THR+EDF"):
            s = summ[(summ.factor == fac) & (summ.label == lab)].copy()
            s["x"] = s.level.astype(float)
            s = s.sort_values("x")
            key = "FURROW" if lab == "FURROW" else lab.split("+")[1]
            ax.plot(s.x, sc * s[m], marker=MK[key], color=C[key], label=lab, markeredgecolor="white", markeredgewidth=0.5)
            ax.fill_between(s.x, sc * s[f"{m}_ci_lo"], sc * s[f"{m}_ci_hi"], color=C[key], alpha=0.12, lw=0)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        if fac == "scale":
            ax.set_xscale("log", base=2)
            ax.set_xticks([100, 200, 400, 800], ["100", "200", "400", "800"])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=1.0)
    fig.subplots_adjust(top=0.84)
    save(fig, "fig6_robustness")
    VALUES["F"] = {f"{r.factor}={r.level}|{r.label}": {m: float(getattr(r, m)) for m in keep} for r in summ.itertuples()}


def derived():
    """Derived quantities quoted in the text (each traceable to the summaries above)."""
    X = {}
    A_m, A_p = VALUES["A_mixed"], VALUES["A_poor"]
    X["A_err_red_mixed"] = 100 * (1 - A_m["FURROW-R@8"]["errors"] / A_m["AoI@8"]["errors"])
    X["A_err_red_poor"] = 100 * (1 - A_p["FURROW-R@8"]["errors"] / A_p["AoI@8"]["errors"])
    X["A_poll_red_mixed"] = 100 * (1 - A_m["FURROW-R@8"]["polls"] / A_m["AoI@8"]["polls"])
    X["A_err_red_thr_mixed"] = 100 * (1 - A_m["FURROW-R@8"]["errors"] / A_m["THR"]["errors"])
    X["A_err_red_thr_poor"] = 100 * (1 - A_p["FURROW-R@8"]["errors"] / A_p["THR"]["errors"])
    B = VALUES.get("B", {})
    for rho in ("1.4", "1.8"):
        base = [v["energy_dyn_kJ"] for k, v in B.items() if k.endswith("@" + rho) and not k.startswith("FURROW")]
        if base:
            X[f"B_energy_red_{rho.replace('.', '')}"] = 100 * (1 - B[f"FURROW-S@{rho}"]["energy_dyn_kJ"] / min(base))
    C = VALUES.get("C", {})
    if C:
        X["C_poll_extra_aoi"] = 100 * (C["AoI8+FURROW-S@1.4@10.0"]["polls"] / C["FURROW@1.4@10.0"]["polls"] - 1)
    VALUES["X"] = X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="static,A,B,C,D,E,F,X")
    a = ap.parse_args()
    fns = {"static": static_tables, "A": exp_a, "B": exp_b, "C": exp_c, "D": exp_d, "E": exp_e, "F": exp_f,
           "X": derived}
    vpath = PROC / "paper_values.json"
    if vpath.exists():
        VALUES.update(json.loads(vpath.read_text()))
    for k in a.only.split(","):
        fns[k]()
        print("done", k, flush=True)
    vpath.write_text(json.dumps(VALUES, indent=1, default=float))


if __name__ == "__main__":
    main()
