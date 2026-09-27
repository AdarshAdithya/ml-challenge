"""TRIVENI: end-to-end business entity resolution at 24M-record scale.

    normalize (parallel; Devanagari transliteration; mined rules)
      -> hashed-key meta-blocking per country, S1 in chunks
      -> learned pruning (small candidate set per S1 entity)
      -> stage-1 pair matcher (LightGBM, 50+ features)
      -> stage-2 context matcher (competition + triangle consistency)
      -> isotonic calibration -> one-owner constraint
      -> expected-F0.5 set decoding -> output/

Usage (from the folder that holds dataset/):
    python code/business_entity_resolution/src/run_pipeline.py \
        --data-dir dataset --out-dir output --workers 8
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from blocking import KeyBlocker, diagnose
from context import ALL_COLS, context_features, context_global, context_local
from decode import apply_owner_constraint, decode
from evaluate import blocking_report, macro_fbeta
from features import make_features
from model import feature_importance, fit_calibrator, fit_full, oof_predict, predict
from normalize import Normalizer, clean, fold, mine_abbreviations, mine_suffixes
from prune import PRUNE_PARAMS, choose_tau, keep_mask, prune_features

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def read_tsv(path, usecols=None):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       usecols=usecols, quoting=3, encoding="utf-8")


def iter_pool(data_dir: Path, split: str, usecols=None, chunksize=1_000_000):
    """Stream Source 2 and 3 in chunks so the full pool never sits in memory."""
    for k in (2, 3):
        yield from pd.read_csv(data_dir / split / f"{split}_source{k}.tsv", sep="\t",
                               dtype=str, keep_default_na=False, quoting=3,
                               encoding="utf-8", usecols=usecols, chunksize=chunksize)


def read_pool_country(data_dir: Path, split: str, country: str) -> pd.DataFrame:
    parts = [ch[ch["country"] == country] for ch in iter_pool(data_dir, split)]
    return pd.concat(parts, ignore_index=True)


def load_truth(data_dir: Path):
    g = read_tsv(data_dir / "train" / "train_ground_truth.tsv")
    return {s: [x for x in m.split(",") if x]
            for s, m in zip(g["source1_entity_id"], g["matched_entity_ids"])}


# ---------------------------------------------------------------- normalization
_NORM = None


def _init(norm):
    global _NORM
    _NORM = norm


def _frame(df):
    return _NORM.frame(df)


def normalize(norm, df, workers, chunk=100_000):
    if len(df) == 0:
        return norm.frame(df)
    if workers <= 1 or len(df) < 2 * chunk:
        return norm.frame(df)
    parts = [df.iloc[a:a + chunk] for a in range(0, len(df), chunk)]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(norm,)) as ex:
        out = list(ex.map(_frame, parts))
    return pd.concat(out)


def country_idf(*frames) -> dict:
    df, n = Counter(), 0
    for f in frames:
        for t in f["name_core"].values:
            n += 1
            df.update(set(t.split()))
    return {w: float(np.log((n + 1) / (c + 1)) + 1.0) for w, c in df.items()}


# ---------------------------------------------------------------- helpers
def label_pairs(pairs, truth):
    true_set = {(s, t) for s, ms in truth.items() for t in ms}
    return np.fromiter(((s, t) in true_set for s, t in
                        zip(pairs["s1_id"].values, pairs["t_id"].values)),
                       dtype=np.int8, count=len(pairs)).astype(int)


def cands_dict(pairs):
    if len(pairs) == 0:
        return {}
    return pairs.groupby("s1_id")["t_id"].apply(list).to_dict()


def fmt_report(br):
    return ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in br.items())


def choose_decoder(df, s1_ids, truth, miss_rate, owner_ok):
    results = []
    for mode in (["off", "soft", "hard"] if owner_ok else ["off"]):
        d = apply_owner_constraint(df, mode)
        for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            pred = decode(d, s1_ids, method="threshold", threshold=th)
            results.append(({"owner": mode, "method": "threshold", "threshold": th},
                            macro_fbeta(pred, truth, ids=s1_ids)))
        for mr in sorted({0.0, round(miss_rate, 4)}):
            pred = decode(d, s1_ids, method="expected", miss_rate=mr)
            results.append(({"owner": mode, "method": "expected", "miss_rate": mr},
                            macro_fbeta(pred, truth, ids=s1_ids)))
    results.sort(key=lambda r: -r[1])
    return results


def error_analysis(pairs, X1, y, pred, truth, s1n, tgtn, n_examples=6):
    """Categorize false merges and missed matches on out-of-fold predictions."""
    pred_set = {(s, t) for s, ts in pred.items() for t in ts}
    key = list(zip(pairs["s1_id"].values, pairs["t_id"].values))
    is_pred = np.fromiter((k in pred_set for k in key), bool, len(key))
    retrieved = set(key)
    a = s1n.drop_duplicates("entity_id").set_index("entity_id")
    b = tgtn.drop_duplicates("entity_id").set_index("entity_id")
    deva = b["business_name"].str.contains("[\u0900-\u0D7F]", regex=True)

    def cats(rows):
        out = Counter()
        for r in rows:
            n, ad = X1["n_tset"].iat[r], X1["a_tset"].iat[r]
            t = pairs["t_id"].iat[r]
            if deva.get(t, False):
                out["Hindi-script name"] += 1
            if X1["a_empty_any"].iat[r] > 0:
                out["address missing on one side"] += 1
            if ad >= 85 and n < 70:
                out["same address, different name"] += 1
            if n >= 85 and ad < 50:
                out["near-identical name, different address"] += 1
            if X1["pc_conflict"].iat[r] > 0 or X1["num_conflict"].iat[r] > 0:
                out["postcode/house-number conflict"] += 1
            if 60 <= n < 85:
                out["partial name overlap"] += 1
        return dict(out.most_common())

    fp = np.flatnonzero(is_pred & (y == 0))
    fn = np.flatnonzero(~is_pred & (y == 1))
    missed_block = sum(1 for s, ts in truth.items() for t in ts if (s, t) not in retrieved)

    def ex(rows):
        res = []
        for r in rows[:n_examples]:
            s, t = pairs["s1_id"].iat[r], pairs["t_id"].iat[r]
            res.append({"s1": f"{a.at[s, 'business_name']} | {a.at[s, 'business_address']}",
                        "other": f"{b.at[t, 'business_name']} | {b.at[t, 'business_address']}"})
        return res
    return {"false_merges": int(len(fp)), "missed_matched_retrieved": int(len(fn)),
            "missed_by_blocking": int(missed_block),
            "fp_categories": cats(fp), "fn_categories": cats(fn),
            "fp_examples": ex(fp), "fn_examples": ex(fn)}



# ---------------------------------------------------------------- parallel test scoring
_CTX = None


def _init_ctx(ctx):
    global _CTX
    _CTX = ctx


CTX_TSUB_COLS = ("name_core", "addr_norm", "source")


def _retrieve_chunk(ch):
    """Worker: retrieval + pruning features (no LightGBM: OpenMP breaks after fork)."""
    pr = _CTX["kb"].query(ch)
    return ch, pr, (prune_features(pr) if len(pr) else None)


def _feature_chunk(job):
    """Worker: pairwise features for the pruned pairs of one chunk."""
    ch, pr = job
    c = _CTX
    tsub = c["cp_idx"].loc[pd.unique(pr["t_id"])].reset_index(drop=True)
    raw = (tsub[["entity_id", "business_name", "business_address", "country"]]
           if c["use_llm"] else None)
    X = make_features(pr, ch, tsub, c["idf"])[c["feat_cols"]]
    return pr, tsub[["entity_id", *CTX_TSUB_COLS]], X, raw


def _pool(ctx, workers):
    import multiprocessing as mp
    method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    return ProcessPoolExecutor(workers, mp_context=mp.get_context(method),
                               initializer=_init_ctx, initargs=(ctx,))


def parallel_query(kb, df, workers, chunk=5000):
    """kb.query over row chunks in worker processes (training side)."""
    df = df.reset_index(drop=True)
    chunks = [df.iloc[a:a + chunk] for a in range(0, len(df), chunk)]
    if workers <= 1 or len(chunks) < 2:
        return kb.query(df)
    with _pool({"kb": kb}, workers) as ex:
        out = [pr for _, pr, _ in ex.map(_retrieve_chunk_nofeat, chunks) if len(pr)]
    return pd.concat(out, ignore_index=True) if out else kb.query(df.iloc[:0])


def _retrieve_chunk_nofeat(ch):
    return ch.iloc[:0], _CTX["kb"].query(ch), None


def _map_chunks(ctx, chunks, workers):
    """Retrieval and features run in workers; every model prediction runs here."""
    ex = _pool(ctx, workers) if workers > 1 and len(chunks) > 1 else None
    if ex is None:
        _init_ctx(ctx)
    mapper = ex.map if ex else map
    try:
        jobs, n_retr = [], 0
        for ch, pr, pf in mapper(_retrieve_chunk, chunks):
            n_retr += len(pr)
            if len(pr):
                pp = predict(ctx["pruner"], pf)
                pr = pr[keep_mask(pr, pp, ctx["tau"], ctx["max_keep"])].reset_index(drop=True)
            if len(pr):
                jobs.append((ch, pr))
        yield ("retrieved", n_retr)
        for pr, tsub, X, raw in mapper(_feature_chunk, jobs):
            p1 = predict(ctx["m1"], X)
            L = context_local(pr, p1, tsub)
            L.insert(0, "t_id", pr["t_id"].values)
            L.insert(0, "s1_id", pr["s1_id"].values)
            yield ("scored", L, raw)
    finally:
        if ex:
            ex.shutdown()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--report", default="run_report.json")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-sample", type=int, default=60000,
                    help="train S1 entities used for fitting (0 = all); pool stays whole")
    ap.add_argument("--prune-sample", type=int, default=50000)
    ap.add_argument("--k-retrieve", type=int, default=200)
    ap.add_argument("--k-mid", type=int, default=50)
    ap.add_argument("--k-max", type=int, default=30)
    ap.add_argument("--df-cap", type=int, default=800)
    ap.add_argument("--query-chunk", type=int, default=2000)
    ap.add_argument("--prune-recall", type=float, default=0.995)
    ap.add_argument("--max-keep", type=int, default=10)
    ap.add_argument("--test-chunk", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loco", action="store_true")
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    rep = {"args": vars(args)}
    rng = np.random.default_rng(args.seed)
    blocker_kw = dict(k_retrieve=args.k_retrieve, k_mid=args.k_mid, k_max=args.k_max,
                      df_cap=args.df_cap, chunk=args.query_chunk)

    # ============================ TRAIN ============================
    s1_all = read_tsv(data_dir / "train" / "train_source1.tsv")
    truth_all = load_truth(data_dir)
    log(f"train: S1={len(s1_all):,} workers={args.workers}")
    s1 = s1_all
    if args.train_sample and len(s1_all) > args.train_sample:
        s1 = s1_all.sample(args.train_sample, random_state=args.seed)
    s1 = s1.reset_index(drop=True)
    del s1_all
    truth = {sid: truth_all.get(sid, []) for sid in s1["entity_id"]}
    owners = Counter(t for ms in truth_all.values() for t in ms)
    multi = sum(1 for c in owners.values() if c > 1)
    owner_ok = multi <= 0.01 * max(len(owners), 1)
    del truth_all, owners
    log(f"fitting on {len(s1):,} S1 entities | singletons={np.mean([not v for v in truth.values()]):.1%}"
        f" | records with >1 owner: {multi:,} -> one-owner constraint "
        f"{'ON' if owner_ok else 'OFF'}")
    rep["multi_owned_records"] = multi
    k = np.array([len(v) for v in truth.values()])
    rep["eda"] = {"avg_matches_per_s1": round(float(k.mean()), 3),
                  "singleton_share": round(float((k == 0).mean()), 4),
                  "max_matches": int(k.max()) if len(k) else 0,
                  "s1_train_total": None}

    # one streaming pass: matched records for rule mining + a name sample for suffixes
    mine_s1 = s1.head(100000)
    want = {t for sid in mine_s1["entity_id"] for t in truth[sid]}
    matched, name_samples = [], []
    for ch in iter_pool(data_dir, "train"):
        matched.append(ch[ch["entity_id"].isin(want)])
        name_samples.append(ch["business_name"].sample(frac=0.05, random_state=1))
    matched = pd.concat(matched).set_index("entity_id")
    if (data_dir / "test").exists() and not args.skip_test:
        for ch in iter_pool(data_dir, "test", usecols=["business_name"]):
            name_samples.append(ch["business_name"].sample(frac=0.05, random_state=1))
    name_pairs, addr_pairs = [], []
    for r in mine_s1.itertuples():
        for m in truth[r.entity_id]:
            if m in matched.index:
                b = matched.loc[m]
                name_pairs.append((clean(fold(r.business_name)).split(),
                                   clean(fold(b.business_name)).split()))
                addr_pairs.append((clean(fold(r.business_address)).split(),
                                   clean(fold(b.business_address)).split()))
    del matched
    name_rules, addr_rules = mine_abbreviations(name_pairs), mine_abbreviations(addr_pairs)
    suffixes = mine_suffixes(pd.concat(name_samples))
    del name_samples
    norm = Normalizer(name_rules, addr_rules, suffixes)
    log(f"mined {len(name_rules)} name rules, {len(addr_rules)} address rules, "
        f"{len(suffixes)} suffix tokens")
    rep["mined"] = {"name": name_rules, "addr": addr_rules, "suffixes": sorted(suffixes)}

    parts, s1n_parts, tgt_parts, idf_by_country, idx_stats = [], [], [], {}, {}
    for country in sorted(s1["country"].unique()):
        s1c = normalize(norm, s1[s1["country"] == country], args.workers)
        poolc = normalize(norm, read_pool_country(data_dir, "train", country), args.workers)
        if len(poolc) == 0:
            log(f"[train:{country}] no pool records; skipped")
            continue
        kb = KeyBlocker(**blocker_kw).index(poolc)
        pr = parallel_query(kb, s1c, args.workers)
        indic = poolc["business_name"].str.contains("[\u0900-\u0D7F]", regex=True)
        idx_stats[country] = {
            "pool": len(poolc), "keys": kb.n_keys, "purged": kb.n_purged,
            "indic_script_names": round(float(indic.mean()), 4),
            "empty_address": round(float((poolc["business_address"].str.strip() == "").mean()), 4),
            "legal_form_present": round(float((poolc["name_suffix"] != "").mean()), 4),
            "postcode_present": round(float((poolc["addr_postcode"] != "").mean()), 4),
        }
        ds = s1c.sample(min(len(s1c), 5000), random_state=0)
        diag = diagnose(kb, ds, poolc, truth, pr[pr["s1_id"].isin(set(ds["entity_id"]))])
        idx_stats[country]["diagnosis"] = diag
        log(f"[diag:{country}] of true pairs: retrieved={diag.get('retrieved')}, "
            f"not in same-country pool={diag.get('target_not_in_same_country_pool')}, "
            f"no shared key={diag.get('no_shared_key')}, only purged keys="
            f"{diag.get('only_purged_keys')}, ranked out={diag.get('shared_key_but_ranked_out')}"
            f" | key coverage {diag.get('key_type_coverage')}")
        idf = country_idf(poolc, s1c)
        tsub = poolc[poolc["entity_id"].isin(set(pr["t_id"]))]
        parts.append(pr)
        s1n_parts.append(s1c)
        tgt_parts.append(tsub)
        idf_by_country[country] = idf
        log(f"[train:{country}] S1={len(s1c):,} pool={len(poolc):,} pairs={len(pr):,} "
            f"keys purged={kb.n_purged:,}/{kb.n_keys:,}")
        del poolc, kb
        gc.collect()
    pairs = pd.concat(parts, ignore_index=True)
    s1n = pd.concat(s1n_parts, ignore_index=True)
    tgtn = pd.concat(tgt_parts, ignore_index=True).drop_duplicates("entity_id")
    y = label_pairs(pairs, truth)
    groups = pairs["s1_id"].values
    br = blocking_report(cands_dict(pairs), truth, 1)
    br.pop("reduction_ratio", None)
    log("retrieval: " + fmt_report(br))
    rep["retrieval_train"], rep["index"] = br, idx_stats

    # ---- learned pruning
    PF = prune_features(pairs)
    ents = pairs["s1_id"].unique()
    fit_mask = np.ones(len(pairs), bool)
    if args.prune_sample and len(ents) > args.prune_sample:
        chosen = set(rng.choice(ents, args.prune_sample, replace=False))
        fit_mask = pairs["s1_id"].isin(chosen).values
    oofp, itp = oof_predict(PF[fit_mask], y[fit_mask], groups[fit_mask], args.folds,
                            rounds=300, params=PRUNE_PARAMS)
    tau = choose_tau(oofp, y[fit_mask], args.prune_recall)
    pruner = fit_full(PF[fit_mask], y[fit_mask], itp, params=PRUNE_PARAMS)
    p_prune = np.empty(len(pairs))
    p_prune[fit_mask] = oofp
    if (~fit_mask).any():
        p_prune[~fit_mask] = predict(pruner, PF[~fit_mask])
    keep = keep_mask(pairs, p_prune, tau, args.max_keep)
    # per-country idf needs the country of each pair
    ctry = pairs["s1_id"].map(s1n.set_index("entity_id")["country"]).values
    pairs, y, ctry = pairs[keep].reset_index(drop=True), y[keep], ctry[keep]
    groups = pairs["s1_id"].values
    br = blocking_report(cands_dict(pairs), truth, 1)
    br.pop("reduction_ratio", None)
    miss_rate = 1 - br["pair_recall"]
    log(f"pruning: tau={tau:.4f} -> " + fmt_report(br))
    rep["blocking_train"], rep["prune_tau"] = br, tau

    # ---- stage 1
    X1 = pd.concat([make_features(pairs[ctry == c].reset_index(drop=True), s1n, tgtn,
                                  idf_by_country[c]).set_index(np.flatnonzero(ctry == c))
                    for c in np.unique(ctry)]).sort_index()
    feat_cols = list(X1.columns)
    oof1, it1 = oof_predict(X1, y, groups, args.folds)
    log(f"stage 1: {X1.shape[1]} features, {len(y):,} pairs, {int(y.sum()):,} positives")

    # ---- stage 2 (context only: p1 + competition + triangle)
    C = context_features(pairs, oof1, tgtn)
    oof2, it2 = oof_predict(C, y, groups, args.folds)
    iso = fit_calibrator(oof2, y)
    df = pairs[["s1_id", "t_id"]].copy()
    df["p"] = iso.predict(oof2)
    s1_ids = s1["entity_id"].tolist()
    base1 = macro_fbeta(decode(df.assign(p=oof1), s1_ids, method="threshold"), truth, ids=s1_ids)
    results = choose_decoder(df, s1_ids, truth, miss_rate, owner_ok)
    best_cfg, best = results[0]
    log(f"OOF macro F0.5 | stage-1 @0.5: {base1:.4f} | final: {best:.4f} {best_cfg}")
    for cfg, sc in results[1:6]:
        log(f"   alt {sc:.4f} {cfg}")
    rep["oof"] = {"stage1_threshold_0.5": base1, "ranking": results}
    best_pred = decode(apply_owner_constraint(df, best_cfg["owner"]), s1_ids,
                       method=best_cfg["method"], threshold=best_cfg.get("threshold", 0.5),
                       miss_rate=best_cfg.get("miss_rate", 0.0))
    rep["errors"] = error_analysis(pairs, X1, y, best_pred, truth, s1n, tgtn)
    log(f"errors: {rep['errors']['false_merges']:,} false merges, "
        f"{rep['errors']['missed_matched_retrieved']:,} missed after retrieval, "
        f"{rep['errors']['missed_by_blocking']:,} lost in blocking")

    if args.loco and len(np.unique(ctry)) > 1:
        rep["loco"] = {}
        for c in np.unique(ctry):
            tr, te = ctry != c, ctry == c
            m1 = fit_full(X1[tr], y[tr], it1)
            p1 = oof1.copy()
            p1[te] = predict(m1, X1[te])
            Cc = context_features(pairs, p1, tgtn)
            m2 = fit_full(Cc[tr], y[tr], it2)
            d = pairs.loc[te, ["s1_id", "t_id"]].copy()
            d["p"] = predict(m2, Cc[te])
            ids = s1n.loc[s1n["country"] == c, "entity_id"].tolist()
            pred = decode(apply_owner_constraint(d, best_cfg["owner"]), ids, method="expected")
            rep["loco"][c] = macro_fbeta(pred, truth, ids=ids)
            log(f"LOCO: without '{c}' -> F0.5 on '{c}' = {rep['loco'][c]:.4f}")

    m1 = fit_full(X1, y, it1)
    m2 = fit_full(C, y, it2)
    rep["importance_stage1"] = feature_importance(m1)
    rep["importance_stage2"] = feature_importance(m2)
    log("top stage-1 features: " + ", ".join(n for n, _ in rep["importance_stage1"][:6]))
    log("top stage-2 features: " + ", ".join(n for n, _ in rep["importance_stage2"][:6]))
    del X1, C, PF, pairs, s1n, tgtn
    gc.collect()

    # ============================ TEST =============================
    if (data_dir / "test").exists() and not args.skip_test:
        u1 = read_tsv(data_dir / "test" / "test_source1.tsv")
        log(f"test: S1={len(u1):,}")
        rows, n_retr, valid, raw_keep = [], 0, set(), []
        for country in sorted(u1["country"].unique()):
            c1 = normalize(norm, u1[u1["country"] == country], args.workers)
            cp = normalize(norm, read_pool_country(data_dir, "test", country), args.workers)
            if not args.use_llm:
                cp = cp.drop(columns=["business_address", "country"], errors="ignore")
            valid.update(cp["entity_id"].values)
            gc.collect()
            if len(cp) == 0:
                log(f"[test:{country}] no pool records -> all empty")
                continue
            kb = KeyBlocker(**blocker_kw).index(cp)
            idf = country_idf(cp, c1)
            cp_idx = cp.set_index("entity_id", drop=False)
            t_c = time.time()
            chunks = [c1.iloc[a:a + args.test_chunk] for a in range(0, len(c1), args.test_chunk)]
            ctx = dict(kb=kb, pruner=pruner, tau=tau, max_keep=args.max_keep, cp_idx=cp_idx,
                       idf=idf, feat_cols=feat_cols, m1=m1, use_llm=args.use_llm)
            done = 0
            for out in _map_chunks(ctx, chunks, args.workers):
                if out[0] == "retrieved":
                    n_retr += out[1]
                    log(f"   [test:{country}] retrieval+pruning done in {time.time() - t_c:.0f}s")
                    continue
                _, L, raw = out
                rows.append(L)
                if raw is not None:
                    raw_keep.append(raw)
                done += 1
                if done % 10 == 0:
                    log(f"   [test:{country}] features: {done} chunks scored")
            log(f"[test:{country}] S1={len(c1):,} pool={len(cp):,} "
                f"kept pairs so far={sum(len(r) for r in rows):,}")
            del cp, cp_idx, kb
            gc.collect()
        T = pd.concat(rows, ignore_index=True)
        G = context_global(T["s1_id"].values, T["t_id"].values, T["p1"].values)
        X2 = pd.concat([T.drop(columns=["s1_id", "t_id"]), G], axis=1)[ALL_COLS]
        udf = T[["s1_id", "t_id"]].copy()
        udf["p"] = iso.predict(predict(m2, X2))
        udf = apply_owner_constraint(udf, best_cfg["owner"])
        if args.use_llm:
            from llm_judge import LLMJudge
            udf = LLMJudge().rerank(udf, u1, pd.concat(raw_keep, ignore_index=True))
        u_ids = u1["entity_id"].tolist()
        matches = decode(udf, u_ids, method=best_cfg["method"],
                         threshold=best_cfg.get("threshold", 0.5),
                         miss_rate=best_cfg.get("miss_rate", 0.0))
        cands = cands_dict(T)
        problems = [s for s in u_ids
                    if not set(matches[s]) <= set(cands.get(s, []))
                    or not set(matches[s]) <= valid]
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s in u_ids:
                f.write(f"{s}\t{','.join(dict.fromkeys(matches[s]))}\n")
        with open(out_dir / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s in u_ids:
                f.write(f"{s}\t{','.join(dict.fromkeys(cands.get(s, [])))}\n")
        ctry_of = u1.set_index("entity_id")["country"]
        by_c = pd.Series({s: len(matches[s]) > 0 for s in u_ids}).groupby(ctry_of).mean()
        cand_c = pd.Series({s: len(cands.get(s, [])) for s in u_ids}).groupby(ctry_of).mean()
        rep["test"] = {"n_s1": len(u_ids), "retrieved_pairs": n_retr, "n_pairs": len(T),
                       "avg_cands_per_s1": len(T) / len(u_ids),
                       "avg_cands_by_country": cand_c.round(3).to_dict(),
                       "match_rate_by_country": by_c.round(3).to_dict(),
                       "avg_matches_per_s1": sum(map(len, matches.values())) / len(u_ids),
                       "problems": len(problems)}
        log(f"test written: {len(u_ids):,} S1 rows | {len(T) / len(u_ids):.2f} candidates/S1 "
            f"| {rep['test']['avg_matches_per_s1']:.2f} matches/S1 | problems={len(problems)}")
        log("candidates/S1 by country: " + ", ".join(f"{k}={v:.2f}" for k, v in cand_c.items()))
        log("share with >=1 match by country: " + ", ".join(f"{k}={v:.2f}" for k, v in by_c.items()))
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, default=str, ensure_ascii=False)
    log(f"done. report -> {args.report}")


if __name__ == "__main__":
    main()
