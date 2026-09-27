"""Build <team_name>_submission.zip in the exact structure the organisers require.

    <team_name>_submission.zip
    ├── output/matching_results.tsv
    ├── output/candidate_pairs.tsv
    ├── code/business_entity_resolution/{src/, README.md, requirements.txt, utils/}
    └── Documentation_template.md

Steps: run the pipeline (unless --skip-run), run the official validator,
fill the methodology document from run_report.json (measured numbers only),
zip, and append a line to submissions_log.csv so you can track versions
against the 5-uploads-per-day limit.

Example, from student_resource/ (where dataset/ and utils/ live):
    python code/business_entity_resolution/utils/make_submission.py \
        --team-name MyTeam --data-dir dataset --validator utils/validate_submission.py
"""
import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
import zipfile
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]          # business_entity_resolution/


def run(cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd))).returncode


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def _tbl(d):
    return ", ".join(f"{k}: {v}" for k, v in d.items()) if d else "n/a"


def build_doc(rep: dict, team: str, members: str) -> str:
    rt, bt = rep.get("retrieval_train", {}), rep.get("blocking_train", {})
    ranking = rep.get("oof", {}).get("ranking", [])
    best_cfg, best = ranking[0] if ranking else ({}, None)
    s1_base = rep.get("oof", {}).get("stage1_threshold_0.5")
    thr = [s for c, s in ranking if c.get("method") == "threshold"]
    best_thr = max(thr) if thr else None
    loco, test, mined = rep.get("loco", {}), rep.get("test", {}), rep.get("mined", {})
    eda, idx, err = rep.get("eda", {}), rep.get("index", {}), rep.get("errors", {})
    a = rep.get("args", {})
    imp1 = ", ".join(f"`{n}`" for n, _ in rep.get("importance_stage1", [])[:8])
    imp2 = ", ".join(f"`{n}`" for n, _ in rep.get("importance_stage2", [])[:8])
    rules = ", ".join(f"{x} -> {y}" for x, y in list(mined.get("name", {}).items())[:8])
    eda_rows = "\n".join(
        f"| {c} | {v.get('pool', 0):,} | {v.get('indic_script_names', 0):.1%} | "
        f"{v.get('empty_address', 0):.1%} | {v.get('legal_form_present', 0):.1%} | "
        f"{v.get('postcode_present', 0):.1%} | {v.get('purged', 0):,} / {v.get('keys', 0):,} |"
        for c, v in idx.items()) or "| n/a | | | | | | |"
    loco_rows = "\n".join(f"| train without {c}, test on {c} | {fmt(v)} |"
                          for c, v in loco.items()) or "| (run with --loco) | |"

    def ex(lst):
        return "\n".join(f"  - `{e['s1']}`  vs  `{e['other']}`" for e in lst[:4]) or "  - n/a"

    def cats(d):
        return "; ".join(f"{k} ({v})" for k, v in list(d.items())[:5]) or "n/a"

    n_pairs = test.get("n_pairs")
    return f"""# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** {team}
**Team Members:** {members}
**Submission Date:** {dt.date.today().isoformat()}

---

## 1. Executive Summary
TRIVENI (three sources merging into one entity, after the Triveni Sangam) is a
blocking + two-stage classifier pipeline built for 24M records and the
precision-heavy, per-entity F0.5 metric. It transliterates names written in
eight Indic scripts, generates candidates with linear-time hashed-key
meta-blocking plus a learned pruner ({fmt(test.get('avg_cands_per_s1'), 2)}
candidates per test entity), scores pairs with LightGBM, adds a second stage
that sees competition between entities and triangle consistency across the
three sources, and picks each entity's match set by maximizing expected F0.5.

---

## 2. Methodology

### 2.1 Problem Analysis
Measured on the training data (Source 1 sample of {a.get('train_sample', 'n/a'):,} entities where sampled):

- A Source 1 entity has on average {eda.get('avg_matches_per_s1', 'n/a')} matches (max {eda.get('max_matches', 'n/a')});
  {fmt(eda.get('singleton_share', 0) * 100 if eda.get('singleton_share') is not None else None, 1)}% are singletons.
  Sources 2 and 3 therefore contain many duplicates of the same business.
- S2/S3 records claimed by more than one Source 1 entity: {rep.get('multi_owned_records', 'n/a')}.
  This decides whether a one-owner-per-record constraint is valid.
- Names appear in Latin and in Indic scripts (Devanagari, Bengali, Gurmukhi,
  Gujarati, Tamil, Telugu, Kannada, Malayalam), e.g. "राम मार्केटिंग प्राइवेट
  लिमिटेड" for "Ram Marketing Pvt Ltd". Addresses sometimes carry native-script
  state names ("ગુજરાત").
- Name noise: legal forms at the start, middle or end ("LLC Moncada...",
  "Holloway Peak Inc Seafood"), dotted forms (S.A.S), junk prefixes ("<<",
  "--"), random accents ("Léarning"), domain-style names ("wilfordhancock.com"),
  abbreviations and typos.
- Address noise: reordered components ("IA, Iowa City, 1064 Newton Rd"),
  state codes vs names (TX/Texas, HR/Haryana), missing postcodes, landmarks
  ("Near Wells Fargo"), French abbreviations (R. = Rue, Bd).

| country | pool records | Indic-script names | empty address | legal form present | postcode present | keys purged / total |
| --- | --- | --- | --- | --- | --- | --- |
{eda_rows}

### 2.2 Solution Strategy
**Approach Type:** Hybrid: blocking + learned pruning + two-stage gradient-boosted classifier + graph-style context features + decision-theoretic decoding.

**Core Innovation:**
1. *Expected-F0.5 set decoding.* The metric is computed per entity with a
   singleton rule, so we choose each entity's match SET (empty or top-k) with
   the highest expected F0.5 under Poisson-binomial outcome distributions,
   instead of a global threshold (Lewis 1995; Jansche 2007; Ye et al. 2012).
2. *Competition and triangle features.* A second-stage model sees whether
   another Source 1 entity claims the same record more strongly, and whether a
   link closes consistent S1-S2-S3 triangles (inspired by GraLMatch/TransClean).
3. *Script-agnostic normalization.* One transliterator covers eight Indic
   scripts by mapping each Unicode block onto Devanagari's parallel layout, and a
   voicing-blind consonant skeleton lets Tamil "குளோபல் பிசினஸ்" and "Global
   Business" produce the identical key `klpl psns`.
4. *Small candidate sets at scale.* Linear-time hashed-key meta-blocking plus
   a learned pruner that keeps {a.get('prune_recall', 0.995):.1%} of retrieved true pairs.

Country is treated as an open set: no feature uses it, and processing runs per
country label as found in the data, so France goes through the same path.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** per record, hashed to uint64 and matched within the
  same country label:
  (t) each consonant-skeleton token of the core name,
  (n2) the first two skeleton tokens joined,
  (c) the first six characters of the space-free core name,
  (p) postcode + first name skeleton,
  (a) first house number + the street word after it.
  Keys shared by more than {a.get('df_cap', 500)} pool records are purged. Each
  pair scores the sum of IDF weights of its shared keys (meta-blocking); the
  top {a.get('k_retrieve', 150)} are re-ranked with hashed TF-IDF cosines
  (name chars, address chars, skeletons, words) in two tiers down to
  {a.get('k_max', 25)}. A LightGBM pruner on blocking-time signals only then
  keeps pairs above an out-of-fold cutoff (at most {a.get('max_keep', 10)} per entity).
- **Candidate pairs generated:** {f"{n_pairs:,}" if isinstance(n_pairs, int) else 'n/a'} on the test set
  ({fmt(test.get('avg_cands_per_s1'), 2)} per Source 1 entity; by country: {_tbl(test.get('avg_cands_by_country', {}))}).
  `candidate_pairs.tsv` is exactly the pruned set the matcher scores.
- **How you ensured true matches were not lost:** five independent key
  families (a pair missed by one is usually caught by another: transliterated
  names by skeleton keys, DBA names by the house-number key, concatenated
  names by the prefix key), rarity-based purging instead of hard filters, and
  a pruning cutoff set by measured recall. Training funnel:

| stage | pair recall | candidates per S1 | F0.5 ceiling (perfect matcher) |
| --- | --- | --- | --- |
| after retrieval | {fmt(rt.get('pair_recall'))} | {fmt(rt.get('avg_cands_per_s1'), 2)} | {fmt(rt.get('f05_ceiling'))} |
| after learned pruning | {fmt(bt.get('pair_recall'))} | {fmt(bt.get('avg_cands_per_s1'), 2)} | {fmt(bt.get('f05_ceiling'))} |

Every step processes Source 1 in chunks against a per-country index, so work
grows linearly with the number of entities; no all-pairs comparison is made.

---

## 4. Matching Model

**Features used:**
- Name features: token-set/sort ratio, partial ratio, Jaro-Winkler,
  normalized Levenshtein, space-free ratio, IDF-weighted Jaccard, soft
  Monge-Elkan (Jaro-Winkler), IDF of the rarest unmatched token (separates
  "Sharma Sweets" from "Sharma Electronics"), consonant-skeleton similarity,
  acronym match, DBA/alternate-name best match, first-token agreement,
  legal-form agreement.
- Address features: token-set, partial and plain ratio, address-word and
  skeleton similarity, postcode agreement/conflict, house-number
  agreement/conflict, landmark similarity, missing-address flag.
- Other: blocking cosines; length and token-count differences; target source.
  Stage 2 adds: rank and gap within the entity's candidates, top-2 gap;
  rank of this entity among all entities claiming the record, strongest
  competing claim and margin; triangle support overall and across sources.

Normalization before features: Indic-script transliteration, accent folding,
abbreviation expansion with rules mined from training matches
({len(mined.get('name', {}))} name rules, {len(mined.get('addr', {}))} address rules; e.g. {rules or 'n/a'}),
legal forms separated wherever they occur, suffix tokens mined from train and
test names ({len(mined.get('suffixes', []))} tokens), landmark/postcode/house-number
extraction, US and Indian state codes expanded.

**Model type:** LightGBM (MIT). Stage 1 on pair features, 5-fold GroupKFold
by Source 1 entity; stage 2 on stage-1 out-of-fold probabilities plus the
context features; isotonic calibration. Most important features, stage 1:
{imp1 or 'n/a'}; stage 2: {imp2 or 'n/a'}.

**Threshold selection method:** none fixed. Per entity we choose the set with
the highest expected F0.5 given calibrated probabilities, including the
challenge's singleton rule. Expected-F decoding, fixed thresholds and the
one-owner modes compete on out-of-fold macro F0.5; the winner is
`{best_cfg}`.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** {fmt(best)} out-of-fold on training entities
  (stage 1 alone at threshold 0.5: {fmt(s1_base)}; best fixed threshold after
  stage 2: {fmt(best_thr)}).
- **Common false positives (wrong merges):** {err.get('false_merges', 'n/a')} on the out-of-fold set.
  Patterns: {cats(err.get('fp_categories', {}))}. Examples:
{ex(err.get('fp_examples', []))}
- **Common false negatives (missed matches):** {err.get('missed_matched_retrieved', 'n/a')} retrieved but not
  predicted, {err.get('missed_by_blocking', 'n/a')} lost in blocking. Patterns: {cats(err.get('fn_categories', {}))}. Examples:
{ex(err.get('fn_examples', []))}

Generalization to France (unseen in training) was rehearsed by
leave-one-country-out validation:

| setting | macro F0.5 |
| --- | --- |
{loco_rows}

Share of test entities with at least one predicted match, by country:
{_tbl(test.get('match_rate_by_country', {}))}.

---

## 6. Conclusion
Designing around the metric paid off most: set-level expected-F0.5 decoding
and the competition features target false merges directly, which F0.5 punishes
twice as hard as misses. Script-agnostic normalization turned records that
would have been empty strings into matchable names, and hashed-key
meta-blocking kept candidate sets small while scaling linearly to 24M records.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
- `src/run_pipeline.py`: entry point; train + test end to end, writes both outputs and `run_report.json`.
- `src/translit.py`: Indic-script transliteration; `src/normalize.py`: cleaning, mined rules, skeletons.
- `src/blocking.py`: hashed-key meta-blocking and TF-IDF re-ranking; `src/prune.py`: learned pruner.
- `src/features.py`: stage-1 features; `src/context.py`: stage-2 competition and triangle features.
- `src/model.py`: LightGBM, grouped out-of-fold training, calibration; `src/decode.py`: expected-F0.5 decoding, one-owner constraint.
- `src/evaluate.py`: metric and blocking diagnostics; `src/llm_judge.py`: optional listwise judge (Qwen2.5-7B-Instruct, Apache 2.0), off by default.
- `utils/make_submission.py`: run, validate, fill this document, zip.

Reproduce from the folder containing `dataset/`:
`python code/business_entity_resolution/src/run_pipeline.py --data-dir dataset --out-dir output`

### B. Additional Results
Run configuration: `{ {k: a.get(k) for k in ('train_sample', 'k_retrieve', 'k_max', 'df_cap', 'prune_recall', 'max_keep', 'folds')} }`.
Fair play: no external data, APIs, geocoding or internet sources; everything
learned comes from the provided files (the suffix miner and IDF weights also
read unlabeled test names). Models: LightGBM (MIT); optional Qwen2.5-7B-Instruct (Apache 2.0, 7.6B).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-name", required=True)
    ap.add_argument("--members", default="[List all team members]")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--validator", default="utils/validate_submission.py")
    ap.add_argument("--report", default="run_report.json")
    ap.add_argument("--skip-run", action="store_true")
    ap.add_argument("--loco", action="store_true", help="also run LOCO for the doc")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra flags passed to run_pipeline.py")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)

    if not a.skip_run:
        cmd = [sys.executable, CODE_DIR / "src" / "run_pipeline.py",
               "--data-dir", a.data_dir, "--out-dir", out_dir, "--report", a.report]
        cmd += (["--loco"] if a.loco else []) + a.extra
        if run(cmd) != 0:
            sys.exit("pipeline failed; nothing packaged")

    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        if not (out_dir / f).exists():
            sys.exit(f"missing {out_dir / f}")

    validator = Path(a.validator)
    if validator.exists():
        rc = run([sys.executable, validator, "--matching", out_dir / "matching_results.tsv",
                  "--candidate", out_dir / "candidate_pairs.tsv",
                  "--test-dir", Path(a.data_dir) / "test"])
        if rc != 0:
            sys.exit("official validator FAILED; fix the issues above before submitting")
        verdict = "PASS"
    else:
        print(f"WARNING: official validator not found at {validator}; "
              "pass --validator path/to/validate_submission.py")
        verdict = "not run"

    rep = json.loads(Path(a.report).read_text()) if Path(a.report).exists() else {}
    doc = build_doc(rep, a.team_name, a.members)
    tpl = Path("Documentation_template.md")
    if tpl.exists() and not Path("Documentation_template.orig.md").exists():
        tpl.rename("Documentation_template.orig.md")      # keep the blank template
    tpl.write_text(doc, encoding="utf-8")

    zip_path = Path(f"{a.team_name}_submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(out_dir / f, f"output/{f}")
        for p in sorted(CODE_DIR.rglob("*")):
            if (p.is_file() and "__pycache__" not in p.parts
                    and p.suffix not in (".pyc", ".json", ".tsv", ".zip", ".log")):
                z.write(p, "code/business_entity_resolution/" +
                        p.relative_to(CODE_DIR).as_posix())
        z.write("Documentation_template.md", "Documentation_template.md")

    oof = (rep.get("oof", {}).get("ranking") or [[None, None]])[0][1]
    log = Path("submissions_log.csv")
    new = not log.exists()
    with open(log, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["timestamp", "zip", "validator", "oof_f05",
                        "test_cands_per_s1", "extra_flags"])
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"), zip_path,
                    verdict, oof, rep.get("test", {}).get("avg_cands_per_s1"),
                    " ".join(a.extra)])
    print(f"\nready: {zip_path}  (validator: {verdict})")
    print(f"upload {out_dir / 'matching_results.tsv'} to the leaderboard; "
          f"submit {zip_path} as the final package")


if __name__ == "__main__":
    main()
