import numpy as np
import pytest

from furrow.evaluation.metrics import alert_prf, dcg, mrr, ndcg_at_k, precision_at_k, r_precision, recall_at_k


def rng():
    return np.random.default_rng(0)


def test_dcg_manual():
    # gains 2^g - 1 = [7, 3, 0]; discounts log2(2), log2(3), log2(4)
    expected = 7 / 1 + 3 / np.log2(3) + 0
    assert dcg(np.array([3, 2, 0]), 3) == pytest.approx(expected)


def test_ndcg_perfect_and_reversed():
    rel = np.array([0, 1, 3, 2])
    assert ndcg_at_k(rel.astype(float), rel, 4, rng()) == pytest.approx(1.0)
    rev = -rel.astype(float)
    val = ndcg_at_k(rev, rel, 4, rng())
    ideal = dcg(np.array([3, 2, 1, 0]), 4)
    got = dcg(np.array([0, 1, 2, 3]), 4)
    assert val == pytest.approx(got / ideal)


def test_ndcg_undefined_without_relevant():
    assert np.isnan(ndcg_at_k(np.array([1.0, 2.0]), np.array([0, 0]), 2, rng()))


def test_precision_recall_rprec_mrr():
    score = np.array([0.9, 0.8, 0.1, 0.5])
    rel = np.array([2, 0, 3, 1])
    assert precision_at_k(score, rel, 2, rng()) == pytest.approx(0.5)
    assert recall_at_k(score, rel, 2, rng()) == pytest.approx(1 / 3)
    # grade>=2 units: idx 0 and 2 -> R = 2, top-2 = idx 0, 1 -> 1/2
    assert r_precision(score, rel, rng()) == pytest.approx(0.5)
    # first grade-3 unit (idx 2) is ranked 4th
    assert mrr(score, rel, rng()) == pytest.approx(0.25)


def test_minus_inf_scores_rank_last():
    score = np.array([-np.inf, 0.1, 0.2])
    rel = np.array([3, 0, 0])
    assert precision_at_k(score, rel, 2, rng()) == 0.0


def test_alert_prf():
    p, r, tp, fp, fn = alert_prf(np.array([True, True, False, False]), np.array([2, 0, 3, 1]))
    assert (tp, fp, fn) == (1, 1, 1)
    assert p == pytest.approx(0.5) and r == pytest.approx(0.5)
