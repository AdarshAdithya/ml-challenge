"""Stage-2 context features built on stage-1 pair probabilities (p1).

A pairwise model judges each pair alone. These features let the stage-2 model
see the neighbourhood:

  S1 side       how p1 ranks among this S1 entity's candidates, gap to its best
  record side   'one owner per record': if Source 1 is deduplicated, an S2/S3
                record belongs to at most one S1 entity. So we tell the model
                whether another S1 entity claims this record more strongly.
  triangle      S1-A ~ S2-x and S2-x ~ S3-y implies S1-A ~ S3-y. For pair (A, x)
                we measure how much x looks like A's other strong candidates.
                Consistent triangles support a link, isolated links look risky.
                (Inspired by transitive-consistency cleanup, GraLMatch/TransClean.)

Everything is vectorized (groupby transforms and one self-join), so it scales
to millions of pairs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

STRONG_P = 0.2
MAX_STRONG = 8
MIN_X_P = 0.02


def _second_best(df, key):
    """Second-highest p1 within each group, broadcast to rows (0 if none)."""
    srt = df[[key, "p1"]].sort_values([key, "p1"], ascending=[True, False])
    nth = srt.groupby(key)["p1"].nth(1)
    second = pd.Series(nth.values, index=srt.loc[nth.index, key].values)
    return df[key].map(second).fillna(0.0).values


LOCAL_COLS = ["p1", "s1_rank", "s1_max", "s1_gap_to_best", "s1_sum", "s1_n_strong",
              "s1_top2_gap", "tri_support", "tri_cross_support", "tri_mean_sim"]
GLOBAL_COLS = ["t_rank", "t_best_other", "t_margin", "t_n_claims", "t_n_strong"]
ALL_COLS = LOCAL_COLS + GLOBAL_COLS


def context_local(pairs: pd.DataFrame, p1: np.ndarray, tgt: pd.DataFrame) -> pd.DataFrame:
    """Features that only need this S1 entity's own candidates (chunk-safe)."""
    df = pairs[["s1_id", "t_id"]].reset_index(drop=True).copy()
    df["p1"] = p1
    C = pd.DataFrame(index=df.index)
    C["p1"] = p1
    g1 = df.groupby("s1_id")["p1"]
    C["s1_rank"] = g1.rank(ascending=False, method="first")
    C["s1_max"] = g1.transform("max")
    C["s1_gap_to_best"] = C["s1_max"] - df["p1"]
    C["s1_sum"] = g1.transform("sum")
    C["s1_n_strong"] = (df["p1"] > 0.5).groupby(df["s1_id"]).transform("sum")
    C["s1_top2_gap"] = C["s1_max"] - _second_best(df, "s1_id")

    info = tgt.drop_duplicates("entity_id").set_index("entity_id")[
        ["name_core", "addr_norm", "source"]]
    df["row"] = np.arange(len(df))
    xs = df[df["p1"] >= MIN_X_P][["row", "s1_id", "t_id"]]
    strong = df[df["p1"] >= STRONG_P].sort_values("p1", ascending=False)
    strong = strong.groupby("s1_id").head(MAX_STRONG)[["row", "s1_id", "t_id", "p1"]]
    j = xs.merge(strong, on="s1_id", suffixes=("_x", "_y"))
    j = j[j["row_x"] != j["row_y"]]
    tri_any = np.zeros(len(df), np.float32)
    tri_cross = np.zeros(len(df), np.float32)
    tri_mean = np.zeros(len(df), np.float32)
    if len(j):
        X = info.loc[j["t_id_x"].values]
        Y = info.loc[j["t_id_y"].values]
        n = cpdist(X["name_core"].tolist(), Y["name_core"].tolist(),
                   scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100
        a = cpdist(X["addr_norm"].tolist(), Y["addr_norm"].tolist(),
                   scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100
        both_addr = (X["addr_norm"].values != "") & (Y["addr_norm"].values != "")
        sim = 0.6 * n + 0.4 * np.where(both_addr, a, n)
        val = sim * j["p1"].values
        cross = X["source"].values != Y["source"].values
        agg = pd.DataFrame({"row": j["row_x"].values, "val": val, "sim": sim,
                            "cval": np.where(cross, val, 0.0)})
        r = agg.groupby("row").agg(val=("val", "max"), cval=("cval", "max"),
                                   sim=("sim", "mean"))
        tri_any[r.index.values] = r["val"].values
        tri_cross[r.index.values] = r["cval"].values
        tri_mean[r.index.values] = r["sim"].values
    C["tri_support"] = tri_any
    C["tri_cross_support"] = tri_cross
    C["tri_mean_sim"] = tri_mean
    return C[LOCAL_COLS].astype(np.float32)


def context_global(s1_ids, t_ids, p1) -> pd.DataFrame:
    """'One owner per record' features: need every S1 claim on each record."""
    df = pd.DataFrame({"t_id": np.asarray(t_ids), "p1": np.asarray(p1)})
    C = pd.DataFrame(index=df.index)
    gt = df.groupby("t_id")["p1"]
    C["t_rank"] = gt.rank(ascending=False, method="first")
    t_max = gt.transform("max").values
    second_t = _second_best(df, "t_id")
    comp = np.where(df["p1"].values >= t_max, second_t, t_max)
    C["t_best_other"] = comp
    C["t_margin"] = df["p1"].values - comp
    C["t_n_claims"] = gt.transform("count").values
    C["t_n_strong"] = (df["p1"] > 0.5).groupby(df["t_id"]).transform("sum")
    return C[GLOBAL_COLS].astype(np.float32)


def context_features(pairs: pd.DataFrame, p1: np.ndarray, tgt: pd.DataFrame) -> pd.DataFrame:
    L = context_local(pairs, p1, tgt)
    G = context_global(pairs["s1_id"].values, pairs["t_id"].values, p1)
    return pd.concat([L, G], axis=1)[ALL_COLS]
