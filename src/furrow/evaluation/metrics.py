"""Retrieval and scheduling metrics (all lower/higher-is-better conventions documented)."""
from __future__ import annotations

import numpy as np


def _rank(score: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Descending order; ties (including -inf) broken uniformly at random."""
    jitter = rng.random(len(score)) * 1e-9
    s = np.where(np.isfinite(score), score, -1e18) + jitter
    return np.argsort(-s, kind="stable")


def dcg(gains: np.ndarray, k: int) -> float:
    g = gains[:k]
    return float(np.sum((2.0 ** g - 1) / np.log2(np.arange(2, len(g) + 2))))


def ndcg_at_k(score, rel, k, rng):           # higher is better
    order = _rank(score, rng)
    ideal = dcg(np.sort(rel)[::-1], k)
    return np.nan if ideal == 0 else dcg(rel[order], k) / ideal


def precision_at_k(score, rel, k, rng, min_grade=1):   # higher is better
    order = _rank(score, rng)
    return float(np.mean(rel[order[:k]] >= min_grade))


def recall_at_k(score, rel, k, rng, min_grade=1):      # higher is better
    order = _rank(score, rng)
    n = int(np.sum(rel >= min_grade))
    return np.nan if n == 0 else float(np.sum(rel[order[:k]] >= min_grade) / n)


def r_precision(score, rel, rng, min_grade=2):        # higher is better
    n = int(np.sum(rel >= min_grade))
    if n == 0:
        return np.nan
    order = _rank(score, rng)
    return float(np.mean(rel[order[:n]] >= min_grade))


def mrr(score, rel, rng, min_grade=3):                 # higher is better
    order = _rank(score, rng)
    hits = np.where(rel[order] >= min_grade)[0]
    return np.nan if len(hits) == 0 else 1.0 / (hits[0] + 1)


def alert_prf(alerted: np.ndarray, rel: np.ndarray, min_grade=2):
    """Precision / recall of the raised alert set against grade >= min_grade."""
    pos = rel >= min_grade
    tp = int(np.sum(alerted & pos))
    fp = int(np.sum(alerted & ~pos))
    fn = int(np.sum(~alerted & pos))
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    return p, r, tp, fp, fn
