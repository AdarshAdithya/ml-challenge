"""Rule-based Devanagari -> Latin transliteration (no external data or models).

About 5% of Source 2/3 names are written in Hindi script, e.g.
'राम मार्केटिंग प्राइवेट लिमिटेड' for 'Ram Marketing Private Limited'.
We transliterate phonetically, drop the word-final inherent vowel, and
collapse long vowels (aa->a, ee->i, oo->u) so the output lines up with
common English spellings: 'ram marketing praivet limited'.
The consonant skeleton used later then matches 'Pvt' / 'Private' (prvt).
"""
import re

CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh",
    "ज": "j", "झ": "jh", "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh",
    "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p",
    "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l",
    "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
    "य़": "y",
}
NUKTA_BASE = {"क": "क़", "ख": "ख़", "ग": "ग़", "ज": "ज़", "ड": "ड़", "ढ": "ढ़",
              "फ": "फ़", "य": "य़"}
VOWELS = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
          "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e",
          "ऎ": "e", "ऒ": "o"}
MATRAS = {"ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri",
          "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e",
          "ॆ": "e", "ॊ": "o", "ॢ": "li"}
VIRAMA, NUKTA = "्", "़"
MARKS = {"ं": "n", "ँ": "n", "ः": "h", "॰": "n", "ॱ": ""}
DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}
CONS.update({"ऴ": "zh", "ऱ": "r", "ऩ": "n"})     # Tamil/Malayalam specials
DEVA = re.compile(r"[\u0900-\u097F]")
# Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam blocks
# mirror the Devanagari layout at fixed offsets, so shifting maps each letter
# to its Devanagari counterpart.
INDIC = re.compile(r"[\u0980-\u0D7F]")
ANY_INDIC = re.compile(r"[\u0900-\u0D7F]")
CHILLU = {"\u0D7A": "ण्", "\u0D7B": "न्", "\u0D7C": "र्", "\u0D7D": "ल्",
          "\u0D7E": "ळ्", "\u0D7F": "क्", "\u09CE": "त्"}
ZW = dict.fromkeys(map(ord, "\u200c\u200d"), None)


def to_devanagari(text: str) -> str:
    out = []
    for ch in text:
        if ch in CHILLU:
            out.append(CHILLU[ch])
            continue
        cp = ord(ch)
        if cp == 0x0B9A:                 # Tamil ca: 's' is the usual reading in names
            out.append("स")
        elif 0x0980 <= cp <= 0x0D7F:
            out.append(chr(0x0900 + (cp - 0x0980) % 0x80))
        else:
            out.append(ch)
    return "".join(out)


# common English loanwords in business names, as they come out of Indic scripts
LOAN = [
    (re.compile(r"^p(i)?r(a|ai|i|e)(v|bh|w|b)(e|a|ae)t+[au]?$"), "private"),
    (re.compile(r"^l(i|e)m(i|e)t+(e|a)(d|t|dd)[au]?$"), "limited"),
    (re.compile(r"^k(a|o)mp(a|e)n(i|ee)$"), "company"),
]


def _word(w: str) -> str:
    w = w.replace("ऱ्ऱ", "ट्ट")          # Malayalam 'റ്റ' is a 'tt' sound
    out = []
    chars = list(w)
    i = 0
    while i < len(chars):
        c = chars[i]
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        if nxt == NUKTA and c in NUKTA_BASE:
            c = NUKTA_BASE[c]
            i += 1
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
        if c in CONS:
            out.append(CONS[c])
            if nxt in MATRAS:
                out.append(MATRAS[nxt])
                i += 1
            elif nxt == VIRAMA:
                i += 1
            else:
                out.append("a")          # inherent vowel
        elif c in VOWELS:
            out.append(VOWELS[c])
        elif c in MARKS:
            out.append(MARKS[c])
        elif c in DIGITS:
            out.append(DIGITS[c])
        elif c == NUKTA:
            pass
        else:
            out.append(c)
        i += 1
    s = "".join(out)
    if len(s) > 2 and s.endswith("a") and not s.endswith("aa") and DEVA.search(w[-1:] or "x"):
        s = s[:-1]                        # schwa deletion at word end
    s = s.replace("aa", "a").replace("ee", "i").replace("oo", "u")
    return s


def _loan(w: str) -> str:
    for rx, rep in LOAN:
        if rx.match(w):
            return rep
    return w


def transliterate(text: str) -> str:
    if not isinstance(text, str) or not ANY_INDIC.search(text):
        return text
    text = to_devanagari(text.translate(ZW))
    words = [_loan(_word(w)) if DEVA.search(w) else w for w in text.split()]
    out = " ".join(words)
    # 'प्रा. लि.' is the Indic abbreviation of 'Pvt. Ltd.'
    return re.sub(r"\bpra\.?\s+li\.?(?=\s|$)", "private limited", out)
