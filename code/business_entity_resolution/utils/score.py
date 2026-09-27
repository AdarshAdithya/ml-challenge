"""Score a matching_results.tsv against a ground-truth TSV with the challenge metric.

python utils/score.py --pred output/matching_results.tsv --truth path/to/truth.tsv
Use it on a held-out slice of train, or on the synthetic hidden test truth.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import fbeta_entity  # noqa: E402


def load(path, col):
    out = {}
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        k = header.index(col)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            ids = parts[k] if len(parts) > k else ""
            out[parts[0]] = [x for x in ids.split(",") if x]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--source1", help="optional source1 TSV to break scores down by country")
    a = ap.parse_args()
    pred = load(a.pred, "matched_entity_ids")
    truth = load(a.truth, "matched_entity_ids")
    scores = {i: fbeta_entity(pred.get(i, []), truth[i]) for i in truth}
    print(f"macro F0.5 = {sum(scores.values()) / len(scores):.4f} over {len(scores)} entities")
    if a.source1:
        country = {}
        with open(a.source1, encoding="utf-8") as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            ci = hdr.index("country")
            for line in fh:
                p = line.rstrip("\n").split("\t")
                country[p[0]] = p[ci]
        by = {}
        for i, s in scores.items():
            by.setdefault(country.get(i, "?"), []).append(s)
        for c, v in sorted(by.items()):
            print(f"  {c:10s} F0.5 = {sum(v) / len(v):.4f}  (n={len(v)})")


if __name__ == "__main__":
    main()
