"""Pairwise features for (Source 1 record, S2/S3 record) candidates.

No feature uses the country label, so the model cannot learn shortcuts that
break on France. Highlights beyond standard fuzzy scores:
  * rare-token evidence: IDF-weighted overlap, soft Monge-Elkan, and the IDF of
    the rarest token that has NO counterpart on the other side
    ('Sharma Sweets' vs 'Sharma Electronics' share 'sharma' but the unmatched
    rare token screams 'different business')
  * number evidence: postcode / house-number agreement and conflict
  * DBA / alternate-name and acronym matching
  * transliteration: consonant-skeleton similarity
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist


def _cp(a, b, scorer, **kw):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def build_idf(texts) -> dict:
    df = Counter()
    n = 0
    for t in texts:
        n += 1
        df.update(set(t.split()))
    return {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}


def _token_evidence(a: str, b: str, idf: dict, default_idf: float):
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return 0.0, 0.0, 0.0, 0.0
    w = lambda t: idf.get(t, default_idf)
    sa, sb = set(ta), set(tb)
    inter = sum(w(t) for t in sa & sb)
    union = sum(w(t) for t in sa | sb)
    idf_jac = inter / union if union else 0.0

    def soft(x, y):
        tot, acc, worst = 0.0, 0.0, 0.0
        for t in x:
            best = max(JaroWinkler.normalized_similarity(t, u) for u in y)
            wt = w(t)
            tot += wt
            acc += wt * best
            if best < 0.88:
                worst = max(worst, wt)
        return acc / tot if tot else 0.0, worst

    me_ab, un_a = soft(ta, tb)
    me_ba, un_b = soft(tb, ta)
    return idf_jac, (me_ab + me_ba) / 2, max(un_a, un_b), min(un_a, un_b)


def _set_overlap(a: pd.Series, b: pd.Series):
    both, eq, conflict = [], [], []
    for x, y in zip(a.values, b.values):
        sx, sy = set(x.split()), set(y.split())
        has = bool(sx) and bool(sy)
        both.append(has)
        eq.append(has and bool(sx & sy))
        conflict.append(has and not (sx & sy))
    return (np.array(both, np.float32), np.array(eq, np.float32),
            np.array(conflict, np.float32))


def make_features(pairs: pd.DataFrame, s1: pd.DataFrame, tgt: pd.DataFrame,
                  idf: dict) -> pd.DataFrame:
    A = s1.set_index("entity_id").loc[pairs["s1_id"]].reset_index(drop=True)
    B = tgt.set_index("entity_id").loc[pairs["t_id"]].reset_index(drop=True)
    F = pd.DataFrame(index=pairs.index)
    for c in [c for c in pairs.columns if c.startswith("cos_")]:
        F[c] = pairs[c].values

    na, nb = A["name_core"].tolist(), B["name_core"].tolist()
    F["n_ratio"] = _cp(na, nb, fuzz.ratio)
    F["n_partial"] = _cp(na, nb, fuzz.partial_ratio)
    F["n_tset"] = _cp(na, nb, fuzz.token_set_ratio)
    F["n_tsort"] = _cp(na, nb, fuzz.token_sort_ratio)
    F["n_jw"] = _cp(na, nb, JaroWinkler.normalized_similarity)
    F["n_lev"] = _cp(na, nb, Levenshtein.normalized_similarity)
    F["nfull_tset"] = _cp(A["name_full"].tolist(), B["name_full"].tolist(),
                          fuzz.token_set_ratio)
    F["nskel_ratio"] = _cp(A["name_skel"].tolist(), B["name_skel"].tolist(),
                           fuzz.ratio)
    F["nskel_tset"] = _cp(A["name_skel"].tolist(), B["name_skel"].tolist(),
                          fuzz.token_set_ratio)
    # DBA / alternate names: best of cross comparisons
    alt_a = [x or y for x, y in zip(A["name_alt"], A["name_core"])]
    alt_b = [x or y for x, y in zip(B["name_alt"], B["name_core"])]
    F["n_alt_best"] = np.maximum.reduce([
        F["n_tset"].values, _cp(alt_a, nb, fuzz.token_set_ratio),
        _cp(na, alt_b, fuzz.token_set_ratio), _cp(alt_a, alt_b, fuzz.token_set_ratio)])
    compact_a = [x.replace(" ", "") for x in na]
    compact_b = [x.replace(" ", "") for x in nb]
    F["n_compact_ratio"] = _cp(compact_a, compact_b, fuzz.ratio)
    acr_a, acr_b = A["name_acronym"].values, B["name_acronym"].values
    F["acronym_hit"] = [
        float((len(x) >= 2 and x == cb) or (len(y) >= 2 and y == ca))
        for x, y, ca, cb in zip(acr_a, acr_b, compact_a, compact_b)]
    first = lambda s: s.split()[0] if s else ""
    F["first_tok_eq"] = [float(first(x) == first(y) and x != "") for x, y in zip(na, nb)]
    F["first_skel_eq"] = [
        float(first(x) == first(y) and x != "")
        for x, y in zip(A["name_skel"], B["name_skel"])]
    sa, sb = A["name_suffix"].values, B["name_suffix"].values
    F["suffix_both"] = [float(bool(x) and bool(y)) for x, y in zip(sa, sb)]
    F["suffix_eq"] = [float(bool(x) and x == y) for x, y in zip(sa, sb)]

    default_idf = max(idf.values()) if idf else 1.0
    ev = np.array([_token_evidence(x, y, idf, default_idf) for x, y in zip(na, nb)],
                  dtype=np.float32)
    F["n_idf_jac"], F["n_soft_me"] = ev[:, 0], ev[:, 1]
    F["n_unmatched_idf_max"], F["n_unmatched_idf_min"] = ev[:, 2], ev[:, 3]

    aa, ab = A["addr_norm"].tolist(), B["addr_norm"].tolist()
    F["a_tset"] = _cp(aa, ab, fuzz.token_set_ratio)
    F["a_partial"] = _cp(aa, ab, fuzz.partial_ratio)
    F["a_ratio"] = _cp(aa, ab, fuzz.ratio)
    F["aw_tset"] = _cp(A["addr_words"].tolist(), B["addr_words"].tolist(),
                       fuzz.token_set_ratio)
    F["askel_tset"] = _cp(A["addr_skel"].tolist(), B["addr_skel"].tolist(),
                          fuzz.token_set_ratio)
    F["a_empty_any"] = [float(not x or not y) for x, y in zip(aa, ab)]
    F["pc_both"], F["pc_eq"], F["pc_conflict"] = _set_overlap(
        A["addr_postcode"], B["addr_postcode"])
    F["num_both"], F["num_eq"], F["num_conflict"] = _set_overlap(
        A["addr_nums"], B["addr_nums"])
    la, lb = A["addr_landmark"].tolist(), B["addr_landmark"].tolist()
    F["lm_both"] = [float(bool(x) and bool(y)) for x, y in zip(la, lb)]
    F["lm_tset"] = _cp(la, lb, fuzz.token_set_ratio)

    F["len_a"] = [len(x) for x in na]
    F["len_b"] = [len(x) for x in nb]
    F["len_diff"] = np.abs(F["len_a"] - F["len_b"])
    F["ntok_diff"] = [abs(len(x.split()) - len(y.split())) for x, y in zip(na, nb)]
    F["t_is_s3"] = (B["source"].values == "S3").astype(np.float32)
    return F.astype(np.float32)
