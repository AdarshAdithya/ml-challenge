"""Scalable candidate generation: hashed-key meta-blocking + light re-ranking.

Built for the real data (about 1.8M Source 1 entities vs about 10M S2/S3
records). No all-pairs comparison anywhere.

1. Keys. Every record emits a handful of blocking keys from its normalized
   fields (country is handled by processing one country at a time):
     t   each consonant-skeleton token of the core name (typo/translit tolerant)
     q   4-char prefix of each core-name token (catches consonant typos late
         in a word, which change the skeleton)
     n2  first two skeleton tokens, sorted (specific, and robust to word swaps)
     c   first 6 chars of the space-free core name ('wilfordhancock')
     p   postcode + first name skeleton
     a   first house number + the street word after it
   Keys are hashed to uint64, so the index costs 16 bytes per key.
2. Block purging. Keys shared by more than `df_cap` pool records carry little
   evidence and create huge blocks, so they are dropped.
3. Meta-blocking. Each (S1, candidate) pair scores the sum of IDF weights of
   the keys it shares, log(1 + N/df). The top `k_retrieve` per S1 survive.
4. Re-ranking. Hashed char/word TF-IDF cosines are computed ONLY for those
   pairs in two tiers (name/address first, then the rest); the top `k_max`
   by a blended score go on to the learned pruner.

Work per S1 chunk is bounded by (#keys x df_cap), so runtime grows linearly
with the number of Source 1 entities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize as l2norm

from normalize import skeleton

KEY_TYPES = ("t", "q", "n2", "c", "p", "a")
_STREET = r"(?:^|\s)(\d+)\s+([a-z]{3,})"


def _hash(arr) -> np.ndarray:
    return pd.util.hash_array(np.asarray(arr, dtype=object))


def _keys_chunk(d: pd.DataFrame, offset: int):
    """Hashed keys for one chunk; only numeric arrays leave this function."""
    rows = np.arange(len(d)) + offset
    R, H, K = [], [], []

    def add(r, keys, kt):
        if len(r):
            R.append(np.asarray(r, np.int64))
            H.append(_hash(keys))
            K.append(np.full(len(r), kt, np.int8))

    skel = d["name_skel"].astype(object).fillna("").str.split()
    core = d["name_core"].astype(object).fillna("").str.split()
    # t: every skeleton token
    t = pd.DataFrame({"row": rows, "k": skel}).explode("k").dropna()
    t = t[t["k"].str.len() >= 2]
    add(t["row"].values, ("t|" + t["k"].astype(str)).values, 0)
    # q: 4-char prefix of every core token (typos near the end, vowel changes)
    q = pd.DataFrame({"row": rows, "k": core}).explode("k").dropna()
    q = q[q["k"].str.len() >= 4]
    add(q["row"].values, ("q|" + q["k"].astype(str).str[:4]).values, 1)
    # n2: first two skeleton tokens, order-insensitive (word transpositions)
    two = skel.map(lambda x: "_".join(sorted(x[:2])) if len(x) >= 2 else None)
    m = two.notna().values
    add(rows[m], ("n|" + two[m].astype(str)).values, 2)
    # c: space-free core-name prefix ('wilfordhancock' vs 'wilford hancock')
    comp = d["name_core"].astype(object).fillna("").str.replace(" ", "", regex=False)
    m = (comp.str.len() >= 4).values
    add(rows[m], ("c|" + comp[m].str[:6]).values, 3)
    # p: postcode + first name skeleton
    pc = d["addr_postcode"].astype(object).fillna("").str.split().str[0]
    f1 = skel.str[0]
    m = (pc.notna() & f1.notna()).values
    add(rows[m], ("p|" + pc[m].astype(str) + "|" + f1[m].astype(str)).values, 4)
    # a: house number + street word after it
    st = d["addr_norm"].astype(object).fillna("").str.extract(_STREET)
    m = st[0].notna().values
    if m.any():
        street = st.loc[m, 1].astype(str).map(skeleton)
        add(rows[m], ("a|" + st.loc[m, 0].astype(str) + "|" + street).values, 5)
    if not R:
        return np.empty(0, np.int64), np.empty(0, np.uint64), np.empty(0, np.int8)
    return np.concatenate(R), np.concatenate(H), np.concatenate(K)


def record_keys(d: pd.DataFrame, chunk: int = 200_000) -> pd.DataFrame:
    """Long table (row, h, ktype) of hashed blocking keys, built in chunks."""
    R, H, K = [], [], []
    for a in range(0, len(d), chunk):
        r, h, k = _keys_chunk(d.iloc[a:a + chunk], a)
        R.append(r), H.append(h), K.append(k)
    if not R:
        return pd.DataFrame({"row": np.empty(0, np.int64), "h": np.empty(0, np.uint64),
                             "ktype": np.empty(0, np.int8)})
    out = pd.DataFrame({"row": np.concatenate(R), "h": np.concatenate(H),
                        "ktype": np.concatenate(K)})
    return out.drop_duplicates(["row", "h"])


def _texts(d, name):
    d = d.astype({c: object for c in ("name_core", "name_alt", "addr_words",
                                      "addr_postcode", "addr_norm", "name_skel")
                  if c in d.columns})
    if name == "cos_name_char":
        return (d["name_core"] + " " + d["name_alt"]).tolist()
    if name == "cos_all_word":
        return (d["name_core"] + " " + d["addr_words"] + " " + d["addr_postcode"]).tolist()
    if name == "cos_addr_char":
        return d["addr_norm"].tolist()
    return d["name_skel"].tolist()


def _hv(analyzer, ngram=(1, 1)):
    kw = dict(analyzer=analyzer, n_features=2 ** 20, alternate_sign=False,
              norm=None, dtype=np.float32)
    if analyzer == "word":
        kw["token_pattern"] = r"\S+"
    else:
        kw["ngram_range"] = ngram
    return HashingVectorizer(**kw)


VEC = {
    "cos_name_char": _hv("char_wb", (3, 4)),
    "cos_all_word": _hv("word"),
    "cos_addr_char": _hv("char_wb", (3, 4)),
    "cos_name_skel": _hv("word"),
}
COLUMNS = (["s1_id", "t_id", "kb_score", "kb_norm", "kb_nkeys"] + [f"kb_{k}" for k in KEY_TYPES]
           + list(VEC) + ["block_score"])


class KeyBlocker:
    def __init__(self, k_retrieve=300, k_mid=60, k_max=30, df_cap=1000, chunk=2000,
                 idf_sample=300000, seed=0):
        self.k_retrieve, self.k_mid, self.k_max = k_retrieve, k_mid, k_max
        self.df_cap, self.chunk = df_cap, chunk
        self.idf_sample, self.seed = idf_sample, seed

    def index(self, pool: pd.DataFrame):
        """Build the key index and TF-IDF weights for one country's pool."""
        self.pool = pool.reset_index(drop=True)
        keys = record_keys(self.pool)
        df = keys.groupby("h").size()
        keep = df[df <= self.df_cap]
        keys = keys[keys["h"].isin(keep.index)].sort_values("h")
        self.k_hash = keys["h"].to_numpy(np.uint64)
        self.k_row = keys["row"].to_numpy(np.int64)
        self.w_by_hash = np.log1p(max(len(self.pool), 1) / keep).astype(np.float32)
        self.n_keys, self.n_purged = int(len(df)), int((df > self.df_cap).sum())
        samp = (self.pool.sample(min(len(self.pool), self.idf_sample),
                                 random_state=self.seed) if len(self.pool) > 0 else None)
        self.tfidf = {}
        for name, vec in VEC.items():
            texts = _texts(samp, name) if samp is not None else ["x"]
            self.tfidf[name] = TfidfTransformer(sublinear_tf=True).fit(vec.transform(texts))
        return self

    def _vectors(self, d, name):
        return l2norm(self.tfidf[name].transform(VEC[name].transform(_texts(d, name))))

    def _cosines(self, agg, c, names):
        uj = np.unique(agg["j"].to_numpy())
        pos = np.searchsorted(uj, agg["j"].to_numpy())
        P = self.pool.iloc[uj]
        ii = agg["i"].to_numpy()
        for name in names:
            A, B = self._vectors(c, name), self._vectors(P, name)
            agg[name] = np.asarray(A[ii].multiply(B[pos]).sum(1)).ravel().astype(np.float32)

    def query(self, s1: pd.DataFrame) -> pd.DataFrame:
        s1 = s1.reset_index(drop=True)
        out = [self._query_chunk(s1.iloc[a:a + self.chunk])
               for a in range(0, len(s1), self.chunk)]
        out = [o for o in out if len(o)]
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=COLUMNS)

    def _query_chunk(self, c: pd.DataFrame) -> pd.DataFrame:
        c = c.reset_index(drop=True)
        if len(self.k_hash) == 0:
            return pd.DataFrame()
        q = record_keys(c)
        qh = q["h"].to_numpy(np.uint64)
        lo = np.searchsorted(self.k_hash, qh, "left")
        cnt = np.searchsorted(self.k_hash, qh, "right") - lo
        m = cnt > 0
        if not m.any():
            return pd.DataFrame()
        q, lo, cnt, qh = q[m], lo[m], cnt[m], qh[m]
        rep = np.repeat(np.arange(len(q)), cnt)
        offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        e = pd.DataFrame({
            "i": q["row"].to_numpy()[rep],
            "j": self.k_row[np.repeat(lo, cnt) + offs],
            "w": self.w_by_hash.reindex(qh[rep]).to_numpy(np.float32),
        })
        kt = q["ktype"].to_numpy()[rep]
        for n, k in enumerate(KEY_TYPES):
            e[f"kb_{k}"] = (kt == n).astype(np.int8)
        agg = e.groupby(["i", "j"], sort=False).agg(
            kb_score=("w", "sum"), kb_nkeys=("w", "size"),
            **{f"kb_{k}": (f"kb_{k}", "sum") for k in KEY_TYPES}).reset_index()
        agg = (agg.sort_values(["i", "kb_score"], ascending=[True, False])
               .groupby("i").head(self.k_retrieve).reset_index(drop=True))
        kb_norm = agg["kb_score"] / agg.groupby("i")["kb_score"].transform("max")
        agg["kb_norm"] = kb_norm.astype(np.float32)
        # tier 1: cheap name + address cosines on the k_retrieve survivors
        self._cosines(agg, c, ("cos_name_char", "cos_addr_char"))
        agg["tier1"] = agg["cos_name_char"] + 0.5 * agg["cos_addr_char"] + 0.2 * agg["kb_norm"]
        agg = (agg.sort_values(["i", "tier1"], ascending=[True, False])
               .groupby("i").head(self.k_mid).reset_index(drop=True))
        # tier 2: remaining views on the k_mid survivors
        self._cosines(agg, c, ("cos_all_word", "cos_name_skel"))
        agg["block_score"] = (agg["cos_name_char"] + 0.5 * agg["cos_addr_char"]
                              + 0.3 * agg["cos_name_skel"] + 0.2 * agg["cos_all_word"]
                              + 0.2 * agg["kb_norm"])
        agg = (agg.sort_values(["i", "block_score"], ascending=[True, False])
               .groupby("i").head(self.k_max))
        agg["s1_id"] = c["entity_id"].to_numpy()[agg["i"].to_numpy()]
        agg["t_id"] = self.pool["entity_id"].to_numpy()[agg["j"].to_numpy()]
        return agg[COLUMNS].reset_index(drop=True)


def diagnose(kb: "KeyBlocker", s1c: pd.DataFrame, poolc: pd.DataFrame,
             truth: dict, retrieved: pd.DataFrame) -> dict:
    """Explain where true pairs are lost for one country (training only)."""
    pairs = [(s, t) for s in s1c["entity_id"].values for t in truth.get(s, [])]
    n = len(pairs)
    if n == 0:
        return {}
    in_pool = set(poolc["entity_id"].values)
    tp = poolc[poolc["entity_id"].isin({t for _, t in pairs})].reset_index(drop=True)
    k1 = record_keys(s1c.reset_index(drop=True))
    k1["s1_id"] = s1c["entity_id"].values[k1["row"].values]
    k2 = record_keys(tp)
    k2["t_id"] = tp["entity_id"].values[k2["row"].values]
    kept = set(kb.w_by_hash.index.values)
    # join on (true owner, key) so the merge is bounded by the true pairs;
    # a plain join on the key alone explodes on common tokens (OOM at scale)
    owner = pd.DataFrame(pairs, columns=["s1_id", "t_id"])
    k2 = k2[["t_id", "h"]].merge(owner, on="t_id")
    m = k1[["s1_id", "h", "ktype"]].merge(k2, on=["s1_id", "h"])
    true_set = set(pairs)
    shared_any = set(zip(m["s1_id"], m["t_id"]))
    mk = m[m["h"].isin(kept)]
    shared_kept = set(zip(mk["s1_id"], mk["t_id"]))
    got = set(zip(retrieved["s1_id"], retrieved["t_id"]))
    not_in_pool = sum(1 for _, t in pairs if t not in in_pool)
    no_key = sum(1 for p in pairs if p[1] in in_pool and p not in shared_any)
    purged_only = sum(1 for p in pairs if p in shared_any and p not in shared_kept)
    ranked_out = sum(1 for p in pairs if p in shared_kept and p not in got)
    by_type = mk.drop_duplicates(["s1_id", "t_id", "ktype"])["ktype"].value_counts()
    return {
        "true_pairs": n,
        "target_not_in_same_country_pool": round(not_in_pool / n, 4),
        "no_shared_key": round(no_key / n, 4),
        "only_purged_keys": round(purged_only / n, 4),
        "shared_key_but_ranked_out": round(ranked_out / n, 4),
        "retrieved": round(len(got & true_set) / n, 4),
        "key_type_coverage": {KEY_TYPES[int(k)]: round(v / n, 4) for k, v in by_type.items()},
    }
