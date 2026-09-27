"""Optional: listwise LLM judge for the uncertain band, with permutation voting.

Wang et al. (COLING 2025) found that showing an LLM the anchor record with its
whole candidate list ('selecting') beats judging pairs one by one and costs
less. They also found strong position bias: accuracy depends on where the true
match sits in the list. We cancel that bias by asking several times with the
candidate order shuffled and using the vote share.

Only S1 entities with a candidate inside the uncertain probability band go to
the LLM, so cost stays small. Default model: Qwen2.5-7B-Instruct (Apache 2.0,
7.6B parameters, within the 8B limit). Needs a GPU; off by default.
"""
from __future__ import annotations

import random
import re

import numpy as np
import pandas as pd

PROMPT = """You are resolving business records. Decide which candidates are the \
SAME real-world business as the anchor. Names may use abbreviations, legal \
suffixes, typos or transliterations. Addresses may be partial or reordered. \
Different branches at different addresses are NOT the same record unless the \
address agrees.

Anchor: {anchor}

Candidates:
{cands}

Answer with the numbers of ALL matching candidates, comma-separated, or 0 if \
none match. Answer with numbers only."""


def _fmt(row) -> str:
    return f"{row.business_name} | {row.business_address} | {row.country}"


def _parse(text: str, n: int):
    nums = {int(x) for x in re.findall(r"\d+", text)}
    return {k for k in nums if 1 <= k <= n}


class LLMJudge:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", n_perm=3,
                 band=(0.2, 0.8), max_cands=6, weight=0.5, generate_fn=None,
                 seed=13):
        self.model_name, self.n_perm, self.band = model_name, n_perm, band
        self.max_cands, self.weight = max_cands, weight
        self.rng = random.Random(seed)
        self._gen = generate_fn

    def _generate(self, prompt: str) -> str:
        if self._gen is None:
            from transformers import pipeline  # optional dependency
            pipe = pipeline("text-generation", model=self.model_name,
                            device_map="auto", torch_dtype="auto")

            def gen(p):
                msgs = [{"role": "user", "content": p}]
                out = pipe(msgs, max_new_tokens=16, do_sample=False)
                return out[0]["generated_text"][-1]["content"]
            self._gen = gen
        return self._gen(prompt)

    def rerank(self, df: pd.DataFrame, s1_raw: pd.DataFrame,
               tgt_raw: pd.DataFrame, col="p") -> pd.DataFrame:
        df = df.copy()
        lo, hi = self.band
        a_info = s1_raw.set_index("entity_id")
        t_info = tgt_raw.set_index("entity_id")
        touched = 0
        for sid, grp in df.groupby("s1_id", sort=False):
            if not ((grp[col] >= lo) & (grp[col] <= hi)).any():
                continue
            top = grp.sort_values(col, ascending=False).head(self.max_cands)
            ids = top["t_id"].tolist()
            votes = np.zeros(len(ids))
            for _ in range(self.n_perm):
                order = ids[:]
                self.rng.shuffle(order)
                lines = "\n".join(f"{k + 1}. {_fmt(t_info.loc[t])}"
                                  for k, t in enumerate(order))
                reply = self._generate(PROMPT.format(anchor=_fmt(a_info.loc[sid]),
                                                     cands=lines))
                for k in _parse(reply, len(order)):
                    votes[ids.index(order[k - 1])] += 1
            share = votes / self.n_perm
            df.loc[top.index, col] = ((1 - self.weight) * top[col].values
                                      + self.weight * share)
            touched += 1
        print(f"[llm] re-judged {touched} S1 entities")
        return df
