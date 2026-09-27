"""Turn calibrated pair probabilities into one match list per Source 1 entity.

The leaderboard scores F_0.5 per S1 entity and averages, singletons included.
So the decision unit is the SET of matches per entity, not a single pair.

Decision-theoretic decoding (Lewis 1995; Jansche 2007; Ye et al., ICML 2012):
assuming independent pair outcomes, the set that maximizes EXPECTED F_beta is
either empty or the top-k candidates by probability. We compute E[F_beta] for
every k with Poisson-binomial distributions and keep the best.

The challenge's singleton rule is built in:
  predict []   -> 1.0 if the entity truly has no match, else 0.0
  predict k>0  -> 0.0 if the entity truly has no match

`miss_rate` models true matches that blocking never retrieved, estimated on
out-of-fold data. It lowers the value of predicting [] for entities that
probably have unseen matches.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _poisson_binomial(ps):
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= (1 - p)
    return dist


def expected_fbeta_set(probs, beta=0.5, miss_rate=0.0, max_consider=25,
                       min_p=1e-3):
    """Return (best_k, best_expected_score) for probabilities sorted desc."""
    probs = np.sort(np.asarray(probs, float))[::-1]
    probs = probs[probs >= min_p]
    b2 = beta * beta
    extra = [miss_rate] if miss_rate > 0 else []
    e_empty = float(np.prod(1 - probs)) * (1 - miss_rate)
    best_k, best_e = 0, e_empty
    m = min(len(probs), max_consider)
    for k in range(1, m + 1):
        tp = _poisson_binomial(probs[:k])                 # P(TP = t), t=0..k
        fn = _poisson_binomial(np.concatenate([probs[k:], extra]))  # true outside
        t = np.arange(k + 1)[:, None]
        f = np.arange(len(fn))[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            score = (1 + b2) * t / (k + b2 * (t + f))
        score = np.nan_to_num(score)
        e = float((tp[:, None] * fn[None, :] * score).sum())
        if e > best_e:
            best_k, best_e = k, e
    return best_k, best_e


def apply_owner_constraint(df: pd.DataFrame, mode: str = "soft",
                           factor: float = 0.3, col: str = "p") -> pd.DataFrame:
    """One owner per S2/S3 record: damp every claim except the strongest.

    mode 'off' | 'soft' (multiply losers by factor) | 'hard' (losers -> 0).
    The pipeline checks the train labels first; if records do map to several
    S1 entities, it switches this off.
    """
    if mode == "off":
        return df
    df = df.copy()
    best = df.groupby("t_id")[col].transform("max")
    loser = df[col] < best
    df.loc[loser, col] = 0.0 if mode == "hard" else df.loc[loser, col] * factor
    return df


def decode(df: pd.DataFrame, s1_ids, beta=0.5, miss_rate=0.0, method="expected",
           threshold=0.5, col="p") -> dict:
    """df has s1_id, t_id, p. Returns {s1_id: [t_id, ...]} for every S1 id.

    One global sort + numpy group boundaries, so it handles millions of pairs.
    """
    out = {sid: [] for sid in s1_ids}
    if len(df) == 0:
        return out
    d = df.sort_values(["s1_id", col], ascending=[True, False])
    ids = d["s1_id"].to_numpy()
    p = d[col].to_numpy(dtype=float)
    t = d["t_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]
    for a, b in zip(starts, ends):
        pp = p[a:b]
        if method == "threshold":
            k = int((pp >= threshold).sum())
        elif pp[0] < 1e-3:
            k = 0
        else:
            k, _ = expected_fbeta_set(pp, beta, miss_rate)
        if k:
            out[ids[a]] = t[a:a + k].tolist()
    return out
