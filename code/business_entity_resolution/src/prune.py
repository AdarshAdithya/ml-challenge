"""Learned candidate pruning: shrink each S1 entity's candidate list before matching.

Final rankings reward a SMALLER candidate set per Source 1 entity, and
candidate_pairs.tsv must be exactly what the matcher scores. So between
retrieval and matching we run a cheap LightGBM that sees only blocking-time
signals (view similarities, their rank within the entity's list, gap to the
entity's best). No string comparisons, so it costs almost nothing per pair.

The cutoff tau comes from out-of-fold scores: the lowest value that still
keeps `target_recall` of the true pairs that retrieval found. Entities whose
candidates all look weak keep ZERO candidates, which is the right answer for
singletons and scores 1.0 when the entity truly has no match.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRUNE_PARAMS = dict(
    objective="binary", learning_rate=0.08, num_leaves=31, min_child_samples=40,
    feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7,
)


def prune_features(pairs: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in pairs.columns
            if c.startswith(("cos_", "kb_score", "kb_norm"))] + ["block_score"]
    F = pairs[cols].copy()
    g = pairs.groupby("s1_id")
    for c in cols:
        F[c + "_rank"] = g[c].rank(ascending=False, method="min")
        F[c + "_gap"] = g[c].transform("max") - pairs[c]
    F["n_cands"] = g["t_id"].transform("count")
    for c in [c for c in pairs.columns if c.startswith("kb_") and c not in cols]:
        F[c] = pairs[c]
    F["t_is_s3"] = pairs["t_id"].astype(str).str.startswith("S3").astype(float)
    return F.astype(np.float32)


def choose_tau(p_oof, y, target_recall):
    pos = np.sort(p_oof[y == 1])
    if len(pos) == 0:
        return 0.5
    k = int(np.floor((1 - target_recall) * len(pos)))
    return float(pos[min(k, len(pos) - 1)])


def keep_mask(pairs: pd.DataFrame, p: np.ndarray, tau: float, max_keep: int):
    d = pd.DataFrame({"s1_id": pairs["s1_id"].values, "p": p})
    rank = d.groupby("s1_id")["p"].rank(ascending=False, method="first").values
    return (p >= tau) & (rank <= max_keep)
