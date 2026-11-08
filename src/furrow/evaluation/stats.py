"""Statistical summaries for repeated independent simulation runs.

* mean and two-sided 95 % confidence interval (Student t, n-1 d.o.f.);
* paired comparison of two methods over the same seeds (identical workloads):
  Wilcoxon signed-rank test (two-sided, zero differences dropped: 'wilcox'),
  Holm-Bonferroni adjustment across a family of comparisons, and the matched-pairs
  rank-biserial correlation as effect size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mean_ci(x, level: float = 0.95):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    m = float(np.mean(x))
    if n == 1:
        return m, np.nan, np.nan, 1
    h = float(stats.t.ppf(0.5 + level / 2, n - 1) * np.std(x, ddof=1) / np.sqrt(n))
    return m, m - h, m + h, n


def paired_test(a, b):
    """a, b aligned by seed. Returns (median diff a-b, p-value, rank-biserial r)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    d = a[ok] - b[ok]
    nz = d[d != 0]
    if len(nz) == 0:
        return 0.0, 1.0, 0.0
    try:
        p = float(stats.wilcoxon(a[ok], b[ok], zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        p = 1.0
    ranks = stats.rankdata(np.abs(nz))
    r = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
    return float(np.median(d)), p, r


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, (m - i) * p[k])
        adj[k] = min(1.0, running)
    return adj


def summarise(df: pd.DataFrame, by: list, metrics: list) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        for m in metrics:
            mu, lo, hi, n = mean_ci(g[m])
            row[m] = mu
            row[f"{m}_ci_lo"] = lo
            row[f"{m}_ci_hi"] = hi
        row["n_runs"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def compare(df: pd.DataFrame, group_cols: list, label_col: str, ref: str, metrics: list) -> pd.DataFrame:
    """Paired tests of `ref` against every other label within each group (Holm per metric/group)."""
    out = []
    for key, g in df.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        base = g[g[label_col] == ref].set_index("seed")
        for m in metrics:
            fam = []
            for lab, h in g[g[label_col] != ref].groupby(label_col):
                h = h.set_index("seed")
                idx = base.index.intersection(h.index)
                med, p, r = paired_test(base.loc[idx, m], h.loc[idx, m])
                fam.append(dict(zip(group_cols, key), metric=m, ref=ref, other=lab, n=len(idx),
                                ref_mean=float(base.loc[idx, m].mean()), other_mean=float(h.loc[idx, m].mean()),
                                median_diff=med, p=p, rank_biserial=r))
            if fam:
                adj = holm([f["p"] for f in fam])
                for f, a in zip(fam, adj):
                    f["p_holm"] = float(a)
                out.extend(fam)
    return pd.DataFrame(out)
