"""Local copy of the leaderboard metric plus blocking diagnostics."""
from __future__ import annotations

import numpy as np


def fbeta_entity(pred, truth, beta=0.5) -> float:
    pred, truth = set(pred), set(truth)
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def macro_fbeta(pred: dict, truth: dict, beta=0.5, ids=None) -> float:
    ids = list(truth) if ids is None else list(ids)
    return float(np.mean([fbeta_entity(pred.get(i, []), truth.get(i, []), beta)
                          for i in ids]))


def blocking_report(cands: dict, truth: dict, n_targets: int, ids=None) -> dict:
    ids = list(truth) if ids is None else list(ids)
    total_true = sum(len(truth.get(i, [])) for i in ids)
    found = sum(len(set(cands.get(i, [])) & set(truth.get(i, []))) for i in ids)
    n_cands = sum(len(cands.get(i, [])) for i in ids)
    # best score reachable if the matcher were perfect on these candidates
    ceiling = macro_fbeta({i: list(set(cands.get(i, [])) & set(truth.get(i, [])))
                           for i in ids}, truth, ids=ids)
    return {
        "pair_recall": found / max(total_true, 1),
        "candidates": n_cands,
        "avg_cands_per_s1": n_cands / max(len(ids), 1),
        "reduction_ratio": 1 - n_cands / max(len(ids) * n_targets, 1),
        "f05_ceiling": ceiling,
    }
