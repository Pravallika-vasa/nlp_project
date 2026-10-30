"""
textfix.py
-----------
Two small helpers that clean up the USER's question before matching:

1. expand_acronyms(): "what is nlp"  ->  "what is nlp (natural language processing)"
   The original acronym is kept (some stored questions use it) and the full
   form is added, so "nlp" and "natural language processing" reach the same entry.

2. SpellFixer: fixes typos such as "adrays" -> "arrays".
   It does NOT use a plain English dictionary alone, because that would "fix"
   technical words (tokenization, biryani, Nehru). Instead every word that
   appears in our own datasets is treated as correct, and a typo is corrected
   to the closest known word, preferring words that occur in our datasets.
"""

import os
import re
import pickle
from collections import Counter

import pandas as pd

CLEAN_DIR = os.path.join("data", "clean")
CACHE_DIR = os.path.join("data", "cache")
DOMAINS = ["cs", "cooking", "gk", "medical"]

# ---------------------------------------------------------------- acronyms
ACRONYMS = {
    "cs": {
        "nlp": "natural language processing", "ml": "machine learning", "ai": "artificial intelligence",
        "dl": "deep learning", "dbms": "database management system", "rdbms": "relational database management system",
        "oop": "object oriented programming", "oops": "object oriented programming",
        "dsa": "data structures and algorithms", "os": "operating system", "iot": "internet of things",
        "cnn": "convolutional neural network", "rnn": "recurrent neural network", "sdlc": "software development life cycle",
        "bios": "basic input output system", "cpu": "central processing unit", "gpu": "graphics processing unit",
        "ram": "random access memory", "rom": "read only memory", "ssd": "solid state drive", "hdd": "hard disk drive",
        "url": "uniform resource locator", "dns": "domain name system", "vpn": "virtual private network",
        "ide": "integrated development environment", "ui": "user interface",
    },
    "gk": {"pm": "prime minister", "cm": "chief minister", "ipl": "indian premier league"},
    "cooking": {},
    "medical": {},
}


def expand_acronyms(text: str, domain: str) -> str:
    table = ACRONYMS.get(domain, {})
    if not table:
        return text
    def repl(m):
        w = m.group(0)
        full = table.get(w.lower())
        return f"{w} ({full})" if full else w
    return re.sub(r"\b[A-Za-z]{2,5}\b", repl, text)


# ---------------------------------------------------------------- vocabulary
def _data_signature():
    sig = []
    for d in DOMAINS:
        p = os.path.join(CLEAN_DIR, f"{d}.csv")
        if os.path.exists(p):
            s = os.stat(p)
            sig.append((d, s.st_size, int(s.st_mtime)))
    return sig


def build_vocab(force: bool = False) -> dict:
    """
    Word statistics over every domain (cached on disk):
      "all": Counter of every word in questions + answers
      "q":   set of words that appear in at least one stored QUESTION
      "qc":  Counter of how often each word appears in stored questions
      "bg":  set of (word, next_word) pairs that occur in stored questions ("heart attack", ...)
    Words in stored questions are real topics (CORS, Kanban, Kadane...) and are always trusted.
    Typos that sit inside long source answers (e.g. "sympotms" in MedQuAD) are not.
    The word pairs let the corrector use the rest of the sentence ("hart attack" -> "heart attack").
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "vocab.pkl")
    sig = _data_signature()
    if not force and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                saved = pickle.load(f)
            if saved.get("sig") == sig and "bg" in saved:
                return {k: saved[k] for k in ("all", "q", "qc", "bg")}
        except Exception:
            pass
    allc, qc, bg = Counter(), Counter(), set()
    for d in DOMAINS:
        p = os.path.join(CLEAN_DIR, f"{d}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        dq = Counter()
        for q in df["question"].astype(str):
            ws = re.findall(r"[a-z]+", q.lower())
            dq.update(ws)
            bg.update(zip(ws, ws[1:]))
        qc.update(dq)
        allc.update(dq)
        atext = " ".join(df["answer"].astype(str).tolist()).lower()
        allc.update(re.findall(r"[a-z]+", atext))
    out = {"all": allc, "q": set(qc), "qc": qc, "bg": bg}
    with open(path, "wb") as f:
        pickle.dump({"sig": sig, **out}, f)
    return out


# ---------------------------------------------------------------- spell fixer
class SpellFixer:
    """
    Corrects EVERY word of the question (it is not a list of selected words):

      pass 1  words that are not real words (typos like "diabtes", "sympotms")
              -> replaced by the closest word we know; if several are equally close, the one that
                 fits the neighbouring words ("heart attack" is in our questions) or is more common wins.
              -> glued words are split ("heartattack" -> "heart attack").
      pass 2  real-word mistakes, where the typo is itself a valid word ("hart" instead of "heart").
              A dictionary can't see these, so we use our data: a word that never appears in our
              questions but sits one edit away from a topic word that appears often, and that topic
              word fits the neighbouring words, is replaced.
    """
    MIN_LEN = 4        # words shorter than this are left alone (too ambiguous to correct safely)
    TRUSTED_COUNT = 3  # a non-English word must appear this often in our data to count as "known"

    def __init__(self, vocab: dict | None = None):
        from spellchecker import SpellChecker  # imported here so matcher.py doesn't need it
        vocab = vocab if vocab is not None else build_vocab()
        self.vocab = vocab["all"]
        self.qwords = vocab["q"]
        self.qc = vocab.get("qc", Counter())
        self.bg = vocab.get("bg", set())
        self.english = SpellChecker(distance=2)              # plain English dictionary only
        self._english_words = self.english.word_frequency.dictionary

        try:
            from rapidfuzz import process, distance
            self._process = process
            self._metric = distance.DamerauLevenshtein       # a swapped pair of letters counts as 1 edit
            targets = set(self._english_words) | set(self.qwords) | \
                {w for w, c in self.vocab.items() if c >= 20}
            self.choices = sorted(w for w in targets if w.isalpha() and len(w) >= 3)
        except ImportError:
            self._process = None
            self.spell = SpellChecker(distance=2)
            self.spell.word_frequency.load_words(list(self.vocab.keys()))

    # ------------------------------------------------------------ helpers
    def _known(self, w: str) -> bool:
        # Trusted: real English, any word used in a stored question (a real topic),
        # or a word frequent enough in the answers. Rare answer-only words are
        # usually typos inside the source text (e.g. "sympotms" in MedQuAD).
        return (w in self._english_words) or (w in self.qwords) or (self.vocab.get(w, 0) >= self.TRUSTED_COUNT)

    def _target_ok(self, c: str, min_len: int = 4) -> bool:
        """May c be used as a REPLACEMENT? Real English, a topic word from our questions, or a word that
        is common in our data. Rare tokens (acronyms like 'cvid', 3-letter fragments) are never targets."""
        return len(c) >= min_len and (
            c in self._english_words or self.qc.get(c, 0) >= 1 or self.vocab.get(c, 0) >= 20)

    def _max_dist(self, lw: str) -> int:
        return 1 if len(lw) <= 5 else 2          # short words: only single-letter slips are corrected

    def _fits(self, c: str, prev: str | None, nxt: str | None) -> bool:
        """Does candidate word c form a word pair that really occurs in our stored questions?"""
        return (prev is not None and (prev, c) in self.bg) or (nxt is not None and (c, nxt) in self.bg)

    def _candidates(self, lw: str, max_d: int):
        if self._process is not None:
            hits = self._process.extract(lw, self.choices, scorer=self._metric.distance,
                                         score_cutoff=max_d, limit=40)
            return [(c, d) for c, d, _ in hits if c != lw]
        out = []
        for d, gen in ((1, self.spell.edit_distance_1), (2, self.spell.edit_distance_2)):
            if d > max_d:
                break
            out += [(c, d) for c in self.spell.known(gen(lw)) if c != lw and self._target_ok(c)]
            if out:
                break
        return out

    def _rank(self, lw: str, cands, prev, nxt):
        """Best candidate: closest spelling first; then one that fits the neighbouring words (a word pair
        that really occurs in our stored questions); then the same first letter (people rarely mistype
        the first letter); then the word most common in our data."""
        def key(item):
            c, d = item
            return (d, 0 if self._fits(c, prev, nxt) else 1, 0 if c[0] == lw[0] else 1,
                    -self.qc.get(c, 0), -self.vocab.get(c, 0), -self._english_words.get(c, 0))
        return min(cands, key=key)[0]

    def _split(self, lw: str):
        """'heartattack' -> 'heart attack' when both halves are real words (best pair wins)."""
        best, best_key = None, None
        for i in range(3, len(lw) - 2):
            a, b = lw[:i], lw[i:]
            if self._target_ok(a, 3) and self._target_ok(b, 3):
                key = ((a, b) in self.bg, min(self.qc.get(a, 0), self.qc.get(b, 0)) > 0,
                       self.qc.get(a, 0) + self.qc.get(b, 0))
                if best_key is None or key > best_key:
                    best, best_key = f"{a} {b}", key
        return best if best_key and (best_key[0] or best_key[1]) else None

    # ------------------------------------------------------------ the two passes
    def _fix_nonword(self, lw, prev, nxt):
        cands = self._candidates(lw, self._max_dist(lw))
        cands = [(c, d) for c, d in cands if self._target_ok(c)]
        if cands:
            return self._rank(lw, cands, prev, nxt)
        return self._split(lw) if len(lw) >= 7 else None

    def _fix_real_word(self, lw, prev, nxt, alone):
        """Typo that spells a real word ('hart'). Only touches words our data has never seen."""
        if self.qc.get(lw, 0) > 0 or self.vocab.get(lw, 0) > 3:
            return None                              # a topic word in our data: leave it
        cands = [(c, d) for c, d in self._candidates(lw, 1) if self.qc.get(c, 0) >= 20]
        if not cands:
            return None
        if alone:                                    # one-word question: no context, so be strict
            best = max(cands, key=lambda cd: self.qc.get(cd[0], 0))[0]
            return best if self.qc.get(best, 0) >= 100 else None
        fitting = [(c, d) for c, d in cands if self._fits(c, prev, nxt)]
        if fitting and not self._fits(lw, prev, nxt):
            return self._rank(lw, fitting, prev, nxt)
        return None

    # ------------------------------------------------------------ public API
    def fix_word(self, word: str, prev: str | None = None, nxt: str | None = None) -> str:
        lw = word.lower()
        if len(lw) < self.MIN_LEN or not lw.isalpha():
            return word
        if not self._known(lw):
            return self._fix_nonword(lw, prev, nxt) or word
        return self._fix_real_word(lw, prev, nxt, alone=(prev is None and nxt is None)) or word

    def fix_detailed(self, text: str):
        """Returns (corrected_text, [(original_word, corrected_word), ...])."""
        spans = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[A-Za-z]+", text)]
        lows = [w.lower() for _, _, w in spans]
        new = list(lows)
        n = len(new)

        def neighbours(i):
            return (new[i - 1].split()[-1] if i > 0 else None,
                    new[i + 1].split()[0] if i + 1 < n else None)

        for i, lw in enumerate(lows):                          # pass 1: non-words
            if len(lw) >= self.MIN_LEN and lw.isalpha() and not self._known(lw):
                prev, nxt = neighbours(i)
                new[i] = self._fix_nonword(lw, prev, nxt) or lw
        for i, lw in enumerate(new):                           # pass 2: real-word mistakes
            if " " not in lw and len(lw) >= self.MIN_LEN and lw.isalpha() and self._known(lw):
                prev, nxt = neighbours(i)
                rep = self._fix_real_word(lw, prev, nxt, alone=(n == 1))
                if rep:
                    new[i] = rep

        out, last, changes = [], 0, []
        for (st, en, orig), low, fixed in zip(spans, lows, new):
            out.append(text[last:st])
            if fixed != low:
                out.append(fixed); changes.append((orig, fixed))
            else:
                out.append(orig)
            last = en
        out.append(text[last:])
        return "".join(out), changes

    def fix(self, text: str):
        """Returns (corrected_text, changed_flag)."""
        fixed, changes = self.fix_detailed(text)
        return fixed, bool(changes)
