# TRIVENI: business entity resolution at 24M-record scale

Three sources, one entity, like the Triveni Sangam where three rivers meet.
For every Source 1 business, this pipeline finds all Source 2 and Source 3
records of the same real-world business.

## Run everything with one command

From the repository root (the folder with `dataset/`, `utils/validate_submission.py`
and `Documentation_template.md`):

```bash
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/utils/make_submission.py \
    --team-name YOUR_TEAM --members "Name 1, Name 2" \
    --data-dir dataset --validator utils/validate_submission.py
```

It trains, predicts the test set, runs the official validator, fills in
`Documentation_template.md` with the measured numbers (the blank template is
kept as `Documentation_template.orig.md`), builds `YOUR_TEAM_submission.zip`
in the required structure, and logs the version in `submissions_log.csv`.
Upload `output/matching_results.tsv` to the leaderboard.

Pipeline only:

```bash
python code/business_entity_resolution/src/run_pipeline.py --data-dir dataset --out-dir output
```

Useful flags (pass them to `make_submission.py` after `--extra`):

| flag | default | effect |
| --- | --- | --- |
| `--workers N` | cores-1 (max 8) | parallel normalization processes |
| `--train-sample N` | 60000 | Source 1 training entities used for fitting; pool always complete |
| `--test-chunk N` | 100000 | S1 entities per test chunk; lower it if memory is tight |
| `--df-cap N` | 800 | purge blocking keys shared by more pool records than this |
| `--prune-recall R` | 0.995 | share of retrieved true pairs the pruner keeps |
| `--max-keep N` | 10 | hard cap on final candidates per entity |
| `--k-retrieve N` | 200 | candidates per entity kept after key scoring |
| `--k-max N` | 30 | candidates per entity passed to the pruner |
| `--query-chunk N` | 2000 | S1 entities per blocking query (memory per step) |
| `--loco` | off | leave-one-country-out check (France rehearsal) |
| `--use-llm` | off | Qwen2.5-7B-Instruct judge on uncertain entities (GPU) |

On the full data the run prints a per-country ETA during the test phase.
Memory peaks while one country's pool is indexed (the US pool is about 6M
records). With 16 GB of RAM use the defaults; if Windows starts paging, lower
`--train-sample` (e.g. 30000) and `--query-chunk` (e.g. 1000).

The training log includes a blocking diagnosis per country: the share of true
pairs retrieved, missing from the same-country pool, sharing no key, sharing
only over-common keys, or ranked out. Use it to tune `--df-cap` (raise it if
"only purged keys" is large) and `--k-retrieve` / `--k-max` (raise them if
"ranked out" is large).

## Pipeline

1. **Normalize** (`translit.py`, `normalize.py`). Names in eight Indic
   scripts are transliterated by mapping each Unicode block onto the parallel
   Devanagari layout ("राम मार्केटिंग प्राइवेट लिमिटेड" -> "ram marketing" +
   "private limited"). Then accent folding, abbreviation expansion with rules
   mined from training matches, legal forms removed wherever they occur (start,
   middle, end, dotted), domain names split, state codes expanded, landmarks,
   postcodes and house numbers extracted, and a voicing-blind consonant
   skeleton (Tamil "குளோபல் பிசினஸ்" and "Global Business" both give `klpl psns`).
2. **Block** (`blocking.py`). Per country, each record emits hashed keys: name
   skeleton tokens, skeleton bigram, space-free name prefix, postcode + name,
   house number + street. Over-common keys are purged. Pairs score the sum of
   IDF weights of shared keys (meta-blocking), then two tiers of hashed TF-IDF
   cosines re-rank the survivors. Source 1 is processed in chunks, so work is
   linear in the number of entities.
3. **Prune** (`prune.py`). A LightGBM on blocking-time signals keeps the pairs
   above an out-of-fold cutoff that preserves 99.5% of retrieved true pairs.
   `candidate_pairs.tsv` is this set, exactly what the matcher scores.
4. **Stage-1 matcher** (`features.py`, `model.py`). 43 pair features,
   LightGBM with 5-fold GroupKFold by Source 1 entity.
5. **Stage-2 context matcher** (`context.py`). Stage-1 probability plus
   entity-side ranks and gaps, record-side competition (is another entity
   claiming this record more strongly?), and S1-S2-S3 triangle support.
6. **Calibrate and decode** (`decode.py`). Isotonic calibration, a
   one-owner-per-record step when the labels support it, and per-entity
   expected-F0.5 set decoding with the challenge's singleton rule.

Nothing uses the country label as a feature, and processing loops over the
country labels found in the data, so France (unseen in training) takes the
same path. No external data, APIs or geocoding. LightGBM is MIT-licensed; the
optional LLM judge is Apache 2.0 (7.6B parameters).

## Files

```
src/run_pipeline.py   entry point, train + test, report
src/translit.py       Indic-script transliteration
src/normalize.py      cleaning, mined rules, skeletons
src/blocking.py       hashed-key meta-blocking + TF-IDF re-ranking
src/prune.py          learned candidate pruner
src/features.py       stage-1 pair features
src/context.py        stage-2 competition and triangle features
src/model.py          LightGBM, grouped OOF, calibration
src/decode.py         expected-F0.5 decoding, one-owner constraint
src/evaluate.py       metric, blocking diagnostics
src/llm_judge.py      optional listwise LLM judge
utils/make_submission.py  run + validate + document + zip
utils/score.py        score predictions against a truth file
utils/make_synthetic_data.py  fake data for smoke tests
```
