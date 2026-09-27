"""Scalable candidate generation: hashed-key meta-blocking + light re-ranking.

Built for the real data (about 1.8M Source 1 entities vs about 10M S2/S3
records). No all-pairs comparison anywhere.

1. Keys. Every record emits a handful of blocking keys from its normalized
   fields (country is handled by processing one country at a time):
     t   each consonant-skeleton token of the core name (typo/translit tolerant)
     n2  first two skeleton tokens joined (specific even when tokens are common)
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

KEY_TYPES = ("t", "n2", "c", "p", "a")
_STREET = r"(?:^|\s)(\d+)\s+([a-z]{3,})"


def record_keys(d: pd.DataFrame) -> pd.DataFrame:
    """Return long table (row, h, ktype) of hashed blocking keys."""
    rows = np.arange(len(d))
    parts = []
    tok = d["name_skel"].fillna("").str.split()
    t = pd.DataFrame({"row": rows, "k": tok}).explode("k").dropna()
    t = t[t["k"].str.len() >= 2]
    parts.append(pd.DataFrame({"row": t["row"].to_numpy(),
                               "k": "t|" + t["k"].astype(str).to_numpy(object),
                               "ktype": 0}))
    ntok = tok.str.len().fillna(0).to_numpy()
    m = ntok >= 2
    two = tok[m].str[:2].str.join("_").astype(str)
    parts.append(pd.DataFrame({"row": rows[m], "k": "n|" + two.to_numpy(object),
                               "ktype": 1}))
    comp = d["name_core"].fillna("").str.replace(" ", "", regex=False)
    m = (comp.str.len() >= 4).to_numpy()
    parts.append(pd.DataFrame({"row": rows[m],
                               "k": "c|" + comp[m].str[:6].astype(str).to_numpy(object),
                               "ktype": 2}))
    pc = d["addr_postcode"].fillna("").str.split().str[0]
    f1 = tok.str[0]
    m = (pc.notna() & f1.notna()).to_numpy()
    if m.any():
        parts.append(pd.DataFrame({
            "row": rows[m],
            "k": ("p|" + pc[m].astype(str) + "|" + f1[m].astype(str)).to_numpy(object),
            "ktype": 3}))
    st = d["addr_norm"].fillna("").str.extract(_STREET)
    m = st[0].notna().to_numpy()
    if m.any():
        street = st.loc[m, 1].astype(str).map(skeleton)
        parts.append(pd.DataFrame({
            "row": rows[m],
            "k": ("a|" + st.loc[m, 0].astype(str) + "|" + street).to_numpy(object),
            "ktype": 4}))
    keys = pd.concat(parts, ignore_index=True)
    keys["h"] = pd.util.hash_array(keys["k"].to_numpy(dtype=object))
    return keys[["row", "h", "ktype"]].drop_duplicates(["row", "h"])


def _texts(d, name):
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
    def __init__(self, k_retrieve=150, k_mid=40, k_max=25, df_cap=500, chunk=5000,
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
