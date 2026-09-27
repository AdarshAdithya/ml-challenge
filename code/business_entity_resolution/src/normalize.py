"""Country-agnostic text normalization for business names and addresses.

Design rule: nothing here may depend on the country label, because the test set
contains France, which training never shows. Dictionaries hold generic domain
knowledge (legal suffixes, street types). Two data-driven miners extend them:
  * mine_abbreviations: learns short->long token rewrites from matched train pairs
  * mine_suffixes: finds legal-suffix-like tokens from name endings in ANY split
    (only the provided files are read; no external data).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

from translit import transliterate

# Legal-form tokens, including French forms so France is not a cold start.
LEGAL_SUFFIXES = {
    "limited", "private", "pvt", "ltd", "llc", "llp", "lp", "plc", "inc",
    "incorporated", "corporation", "corp", "company", "co", "pllc", "pc",
    "gmbh", "ag", "bv", "nv", "pty", "opc",
    # France
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scop", "scp", "selarl",
    "cie", "et", "ets", "etablissements",
}
LEADING_STOP = {"the", "m/s", "ms", "messrs", "le", "la", "les", "l"}

NAME_ABBREV = {
    "pvt": "private", "prv": "private", "ltd": "limited", "lmtd": "limited",
    "corp": "corporation", "inc": "incorporated", "co": "company",
    "intl": "international", "int'l": "international", "mfg": "manufacturing",
    "svc": "service", "svcs": "services", "bros": "brothers",
    "assoc": "associates", "assn": "association", "ent": "enterprises",
    "entp": "enterprises", "inds": "industries", "ind": "industries",
    "mgmt": "management", "dept": "department", "natl": "national",
    "univ": "university", "hosp": "hospital", "ctr": "center",
    "centre": "center", "sons": "son", "n": "and", "et": "and",
    # transliterated Hindi legal words
    "praivet": "private", "prayvet": "private", "limitad": "limited",
    "elaelapi": "llp", "elelpi": "llp", "kampani": "company", "kanpani": "company",
    "karporeshan": "corporation", "indastrij": "industries",
}
WEB_TOKENS = {"www", "com", "net", "org", "biz", "info", "co", "in", "io", "fr", "us"}

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin",
    "wy": "wyoming", "dc": "district of columbia",
}
IN_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar",
    "cg": "chhattisgarh", "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat",
    "hr": "haryana", "hp": "himachal pradesh", "jh": "jharkhand", "ka": "karnataka",
    "kl": "kerala", "mp": "madhya pradesh", "mh": "maharashtra", "mn": "manipur",
    "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland", "od": "odisha", "or": "odisha",
    "pb": "punjab", "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu",
    "tg": "telangana", "ts": "telangana", "tr": "tripura", "up": "uttar pradesh",
    "uk": "uttarakhand", "ut": "uttarakhand", "wb": "west bengal", "dl": "delhi",
    "jk": "jammu and kashmir", "ch": "chandigarh", "py": "puducherry",
}

ADDR_ABBREV = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue",
    "av": "avenue", "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard",
    "ln": "lane", "dr": "drive", "ct": "court", "pl": "place", "hwy": "highway",
    "pkwy": "parkway", "sq": "square", "apt": "apartment", "ste": "suite",
    "fl": "floor", "flr": "floor", "bldg": "building", "blk": "block",
    "sect": "sector", "sec": "sector", "ngr": "nagar", "mkt": "market",
    "nr": "near", "opp": "opposite", "rte": "route", "chem": "chemin",
    "imp": "impasse", "fbg": "faubourg", "crs": "cours",
    "mg": "mahatma gandhi", "stn": "station", "jn": "junction",
    "jct": "junction", "n": "north", "s": "south", "e": "east", "w": "west",
    "no": "number", "num": "number", "r": "rue", "ch": "chemin",
    "tq": "taluk", "dist": "district", "distt": "district",
}

LANDMARK_RE = re.compile(
    r"\b(near|nr|opp|opposite|behind|beside|next to|in front of|adjacent to|"
    r"pres de|pres du|face a|en face de)\b[^,;]*",
    flags=re.IGNORECASE,
)
DBA_RE = re.compile(r"\b(d/?b/?a|doing business as|trading as|t/a|aka)\b", re.I)
NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
SPACES = re.compile(r"\s+")
DIGITS = re.compile(r"\d+")


def fold(text) -> str:
    """Lowercase, strip accents (e -> e for French), unify ampersands."""
    if not isinstance(text, str):
        return ""
    text = transliterate(unicodedata.normalize("NFC", text))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("&", " and ").replace("+", " and ")
    return text


def clean(text: str) -> str:
    text = NON_ALNUM.sub(" ", text)
    return SPACES.sub(" ", text).strip()


def skeleton(token: str) -> str:
    """Transliteration-tolerant consonant key: 'Laxmi'/'Lakshmi' -> 'lksm'."""
    t = token
    for a, b in (("ksh", "ks"), ("x", "ks"), ("ph", "f"), ("sh", "s"),
                 ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"),
                 ("bh", "b"), ("ch", "c"), ("ck", "k"), ("q", "k"),
                 ("w", "v"), ("z", "j"), ("c", "k")):
        t = t.replace(a, b)
    if not t:
        return t
    # voicing-blind: Tamil script writes k/g, t/d, p/b with one letter each
    t = t.replace("g", "k").replace("b", "p").replace("d", "t")
    head, tail = t[0], re.sub(r"[aeiouyh]", "", t[1:])
    t = head + tail
    return re.sub(r"(.)\1+", r"\1", t)


def mine_abbreviations(pairs, min_count: int = 3) -> dict:
    """Learn abbreviation rewrites from matched pairs of cleaned token lists.

    For every true match we compare the tokens only one side has. If token a is
    shorter, shares the first letter, and its letters appear in order inside b
    (pvt -> private, rd -> road, blvd -> boulevard), count (a, b). Frequent
    pairs become rewrite rules. This learns the noise the data actually has.
    """
    counts = Counter()
    for ta, tb in pairs:
        sa, sb = set(ta), set(tb)
        only_a, only_b = sa - sb, sb - sa
        if not only_a or not only_b or len(only_a) > 4 or len(only_b) > 4:
            continue
        for a in only_a:
            for b in only_b:
                s, l = (a, b) if len(a) < len(b) else (b, a)
                if len(s) < 1 or len(l) < 3 or s[0] != l[0] or s.isdigit():
                    continue
                it = iter(l)
                if all(ch in it for ch in s):
                    counts[(s, l)] += 1
    best = {}
    for (s, l), c in counts.most_common():
        if c >= min_count and s not in best:
            best[s] = l
    return best


def mine_suffixes(raw_names, min_share: float = 0.002, max_len: int = 6) -> set:
    """Find legal-suffix-like tokens: short tokens that END many names.

    Run on train + test names so unseen French forms (sarl, sas, ...) are
    detected from the unlabeled test file itself.
    """
    last = Counter()
    anywhere = Counter()
    n = 0
    for name in raw_names:
        toks = clean(fold(name)).split()
        if len(toks) < 2:
            continue
        n += 1
        last[toks[-1]] += 1
        anywhere.update(set(toks))
    found = set()
    for tok, c in last.items():
        if (len(tok) <= max_len and not tok.isdigit() and c / max(n, 1) >= min_share
                and c / anywhere[tok] >= 0.8):
            found.add(tok)
    return found


class Normalizer:
    def __init__(self, extra_name_abbrev=None, extra_addr_abbrev=None,
                 extra_suffixes=None):
        self.name_abbrev = dict(NAME_ABBREV)
        self.addr_abbrev = dict(ADDR_ABBREV)
        for src, dst in ((extra_name_abbrev, self.name_abbrev),
                         (extra_addr_abbrev, self.addr_abbrev)):
            for k, v in (src or {}).items():
                dst.setdefault(k, v)
        self.suffixes = set(LEGAL_SUFFIXES) | set(extra_suffixes or ())
        self.suffixes |= {self.name_abbrev.get(s, s) for s in self.suffixes}

    # ---- names -----------------------------------------------------------
    def name(self, raw) -> dict:
        text = fold(raw)
        parts = [p for p in DBA_RE.split(text) if p and not DBA_RE.fullmatch(p)]
        variants = []
        for p in parts or [text]:
            is_web = bool(re.search(r"\.(com|net|org|in|co|fr|biz|info|io)\b|www\.", p))
            toks = [self.name_abbrev.get(t, t) for t in clean(p).split()]
            toks = " ".join(toks).split()
            if is_web:
                toks = [t for t in toks if t not in WEB_TOKENS] or toks
            # merge dotted forms: 's a s' -> 'sas', 'l l c' -> 'llc'
            merged, run = [], ""
            for t in toks:
                if len(t) == 1 and t.isalpha():
                    run += t
                    continue
                if run:
                    merged.append(self.name_abbrev.get(run, run) if len(run) > 1 else run)
                    run = ""
                merged.append(t)
            if run:
                merged.append(self.name_abbrev.get(run, run) if len(run) > 1 else run)
            toks = merged
            while len(toks) > 1 and toks[0] in LEADING_STOP:
                toks = toks[1:]
            # legal forms can appear anywhere (noise injects them mid-name)
            suffix = [t for t in toks if t in self.suffixes]
            toks = [t for t in toks if t not in self.suffixes] or toks
            variants.append((toks, suffix))
        core_toks, suffix_toks = variants[0]
        return {
            "name_full": " ".join(core_toks + suffix_toks),
            "name_core": " ".join(core_toks),
            "name_alt": " ".join(variants[1][0]) if len(variants) > 1 else "",
            "name_suffix": " ".join(suffix_toks),
            "name_skel": " ".join(skeleton(t) for t in core_toks),
            "name_acronym": "".join(t[0] for t in core_toks if t),
        }

    # ---- addresses -------------------------------------------------------
    def address(self, raw) -> dict:
        text = fold(raw)
        landmarks = [m.group(0) for m in LANDMARK_RE.finditer(text)]
        text = LANDMARK_RE.sub(" ", text)
        raw_toks = clean(text).split()
        toks = []
        for k, t in enumerate(raw_toks):
            last = k == len(raw_toks) - 1 or k == 0
            if len(t) == 2 and (last or t in ("tx", "hr", "dl", "ka", "mh")):
                t = US_STATES.get(t) or IN_STATES.get(t) or self.addr_abbrev.get(t, t)
            else:
                t = self.addr_abbrev.get(t, t)
            toks.append(t)
        toks = " ".join(toks).split()
        nums = DIGITS.findall(" ".join(toks))
        postcodes = [x for x in nums if len(x) in (5, 6)]
        others = [x.lstrip("0") or "0" for x in nums if len(x) not in (5, 6)]
        words = [t for t in toks if not t.isdigit()]
        return {
            "addr_norm": " ".join(toks),
            "addr_words": " ".join(words),
            "addr_skel": " ".join(skeleton(t) for t in words),
            "addr_postcode": " ".join(postcodes),
            "addr_nums": " ".join(others),
            "addr_landmark": clean(" ".join(landmarks)),
        }

    def frame(self, df):
        import pandas as pd
        names = pd.DataFrame([self.name(x) for x in df["business_name"]],
                             index=df.index)
        addrs = pd.DataFrame([self.address(x) for x in df["business_address"]],
                             index=df.index)
        out = pd.concat([df, names, addrs], axis=1)
        out["source"] = df["entity_id"].str.slice(0, 2)
        return out
