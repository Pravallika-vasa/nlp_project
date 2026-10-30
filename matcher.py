"""
matcher.py
-----------
Hybrid question-matching engine.

For a chosen domain, this:
  1. Loads that domain's cleaned {question, answer} pairs
  2. Builds a TF-IDF index over the PREPROCESSED questions (lemmatized, stopwords removed)
  3. Builds an SBERT embedding index over the RAW (original) questions
  4. For a new user question, computes both similarity scores against every
     stored question, combines them into one hybrid score, and returns the
     best-matching answer (or a fallback message if nothing scores high enough)

Why both TF-IDF and SBERT (not just one)?
  - TF-IDF is strong when the user reuses the same key terms as the dataset.
  - SBERT is strong when the user paraphrases ("reverse" vs "invert" a list).
  - Combining them covers both cases and is easy to justify/evaluate in a report.

Embeddings and the TF-IDF matrix are computed ONCE per domain and cached in
memory (see get_engine() below) so repeated questions in a chat session are fast.
"""

import os
import re

# Stop sentence-transformers/huggingface from checking the internet on every
# startup once the model is already downloaded once - this removes a real
# network-wait delay from every later app start.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import hashlib
import pickle
import functools
import threading
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from preprocess import preprocess_text
from textfix import expand_acronyms

CLEAN_DIR = os.path.join("data", "clean")
CACHE_DIR = os.path.join("data", "cache")
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
ALL_DOMAINS = ["cs", "cooking", "gk", "medical"]

# Weights for combining TF-IDF score and SBERT score into one hybrid score.
# SBERT gets more weight since it generally handles meaning/paraphrasing better;
# TF-IDF still contributes for exact keyword matches. Tune these if you like.
# TF-IDF is only a minor helper for RANKING candidates (it caught false matches
# on shared words like "unit"/"random"). The answer/fallback decision is made on
# the SBERT score alone, which was far more reliable in our tests.
TFIDF_WEIGHT = 0.35
SBERT_WEIGHT = 0.65

# The bot answers only if the best SBERT score (question-match OR answer-match)
# is at least this. Tuned on a small test set (gibberish ~0.33, wrong match
# ~0.59, correct paraphrase matches ~0.63-0.67). Re-tune with a bigger test set.
SBERT_THRESHOLD = 0.60

# If the best direct QUESTION match scores at least this high, it's trusted
# outright and answer-matching is skipped entirely for that query. This stops
# a long answer that merely mentions the query term in passing from stealing
# the win from a near-exact topic match (e.g. "cancer" vs "What is Cancer?").
QUESTION_PREFERENCE_THRESHOLD = 0.65

# Other entries must score at least this to be offered as "related questions".
RELATED_MIN = 0.50

_sbert_model = None  # lazy-loaded, shared across all domains


def get_sbert_model():
    global _sbert_model
    if _sbert_model is None:
        try:
            _sbert_model = SentenceTransformer(SBERT_MODEL_NAME, local_files_only=True)
        except Exception:
            # not downloaded yet (very first run) -> allow one online download
            os.environ["HF_HUB_OFFLINE"] = "0"
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            _sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return _sbert_model


@functools.lru_cache(maxsize=512)
def _encode_query(text: str):
    """Encode a user question once; reused when the same text is checked against several domains."""
    return get_sbert_model().encode([text], convert_to_numpy=True)


class DomainEngine:
    """Holds the TF-IDF + SBERT index for one domain."""

    def __init__(self, domain: str):
        self.domain = domain
        df = pd.read_csv(os.path.join(CLEAN_DIR, f"{domain}.csv"))
        df = df.dropna(subset=["question", "answer"]).reset_index(drop=True)
        self.questions = df["question"].tolist()
        self.answers = df["answer"].tolist()
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._check_cache_is_current(df)

        # ---- TF-IDF index (preprocessed/lemmatized text) ----
        # NLTK preprocessing is slow (~1 min for the medical set), so the finished
        # index is saved to disk and reused on every later start.
        tfidf_path = os.path.join(CACHE_DIR, f"{domain}_tfidf.pkl")
        loaded = None
        if os.path.exists(tfidf_path):
            with open(tfidf_path, "rb") as f:
                loaded = pickle.load(f)
            if loaded[1].shape[0] != len(self.questions):  # dataset changed -> rebuild
                loaded = None
        if loaded is None:
            processed = [preprocess_text(q) for q in self.questions]
            vectorizer = TfidfVectorizer()
            matrix = vectorizer.fit_transform(processed)
            with open(tfidf_path, "wb") as f:
                pickle.dump((vectorizer, matrix), f)
            loaded = (vectorizer, matrix)
        self.tfidf_vectorizer, self.tfidf_matrix = loaded

        # ---- SBERT indexes (RAW original text) ----
        # One over the stored QUESTIONS, one over the stored ANSWERS.
        self.sbert_embeddings = self._load_or_build(
            os.path.join(CACHE_DIR, f"{domain}_sbert.pkl"), self.questions
        )
        # Answers can be very long (medical). Only the first ~600 characters are
        # embedded: the model reads ~128-256 tokens anyway, so this is much faster
        # with no real loss of meaning.
        self.sbert_answer_embeddings = self._load_or_build(
            os.path.join(CACHE_DIR, f"{domain}_sbert_ans.pkl"), self.answers, max_chars=600
        )
        json.dump({"hash": self._data_hash, "rows": len(self.questions)}, open(self._meta_path, "w"))

    def _check_cache_is_current(self, df):
        """If this domain's data changed since its caches were built, throw the old caches away.
        (Row counts alone can't tell, e.g. when only answers were edited.) Caches that predate
        this check have no meta file and are trusted as long as their row counts match."""
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(df[["question", "answer"]], index=False).values.tobytes()
        ).hexdigest()
        meta_path = os.path.join(CACHE_DIR, f"{self.domain}_meta.json")
        if os.path.exists(meta_path):
            try:
                old = json.load(open(meta_path)).get("hash")
            except Exception:
                old = None
            if old != data_hash:
                for suffix in ("_tfidf.pkl", "_sbert.pkl", "_sbert_ans.pkl"):
                    p = os.path.join(CACHE_DIR, f"{self.domain}{suffix}")
                    if os.path.exists(p):
                        os.remove(p)
        self._meta_path, self._data_hash = meta_path, data_hash

    @staticmethod
    def _load_or_build(cache_path, texts, max_chars=None):
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                emb = pickle.load(f)
            if emb.shape[0] == len(texts):  # ignore stale cache if dataset changed
                return emb
        if max_chars:
            texts = [t[:max_chars] for t in texts]
        model = get_sbert_model()
        emb = model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
        with open(cache_path, "wb") as f:
            pickle.dump(emb, f)
        return emb

    def query(self, user_question: str):
        """
        Returns dict: {answer, matched_question, score, confident (bool)}
        """
        # keep the user's words but add the full form of known acronyms ("nlp" -> "nlp (natural language processing)")
        user_question = expand_acronyms(user_question, self.domain)

        # TF-IDF similarity
        proc_q = preprocess_text(user_question)
        user_tfidf = self.tfidf_vectorizer.transform([proc_q])
        tfidf_scores = cosine_similarity(user_tfidf, self.tfidf_matrix)[0]

        # SBERT similarity: against stored questions AND stored answers.
        # Question-matching is trusted FIRST (it's exact-topic matching, e.g.
        # "cancer" vs "What is Cancer?"). Answer-matching is only used as a
        # fallback for paraphrased/descriptive queries, because otherwise a
        # long answer that merely MENTIONS the query term in passing (e.g. a
        # Depression answer that mentions "cancer" as a risk factor) could
        # outscore the actual dedicated entry for that topic.
        user_emb = _encode_query(user_question)
        q_scores = cosine_similarity(user_emb, self.sbert_embeddings)[0]
        a_scores = cosine_similarity(user_emb, self.sbert_answer_embeddings)[0]

        best_q_idx = int(np.argmax(q_scores))
        best_q_score = float(q_scores[best_q_idx])

        if best_q_score >= QUESTION_PREFERENCE_THRESHOLD:
            sbert_scores = q_scores  # trust the strong direct match, ignore answers
        else:
            sbert_scores = np.maximum(q_scores, a_scores)  # rescue paraphrases

        # Hybrid score is used to RANK candidates
        hybrid_scores = TFIDF_WEIGHT * tfidf_scores + SBERT_WEIGHT * sbert_scores

        best_idx = int(np.argmax(hybrid_scores))
        best_sbert = float(sbert_scores[best_idx])

        # Up to 3 other close entries (shown as "related questions" in the chat)
        related = []
        for i in np.argsort(hybrid_scores)[::-1][1:8]:
            if sbert_scores[i] >= RELATED_MIN and self.questions[i] != self.questions[best_idx]:
                related.append(self.questions[i])
            if len(related) == 3:
                break

        result = {
            "matched_question": self.questions[best_idx],
            "answer": self.answers[best_idx],
            "score": round(float(hybrid_scores[best_idx]), 3),
            "tfidf_score": round(float(tfidf_scores[best_idx]), 3),
            "sbert_score": round(best_sbert, 3),
            "matched_by": "question" if q_scores[best_idx] >= a_scores[best_idx] else "answer",
            # decision is made on the SBERT score only
            "confident": best_sbert >= SBERT_THRESHOLD,
            "related": related,
        }
        return result


_engines = {}  # cache: domain name -> DomainEngine
_locks = {d: threading.Lock() for d in ALL_DOMAINS}  # one lock per domain: same domain never built twice


def get_engine(domain: str) -> DomainEngine:
    if domain not in _engines:
        with _locks.setdefault(domain, threading.Lock()):
            if domain not in _engines:
                _engines[domain] = DomainEngine(domain)
    return _engines[domain]


def _cache_ready(domain: str) -> bool:
    return all(os.path.exists(os.path.join(CACHE_DIR, f"{domain}{s}"))
               for s in ("_tfidf.pkl", "_sbert.pkl", "_sbert_ans.pkl"))


def preload_cached_engines():
    """Quietly load every domain whose cache already exists, in a background thread, so that
    switching domains or the cross-domain check later is instant. Domains WITHOUT a cache are
    skipped on purpose (building one would take minutes and slow the app down)."""
    def run():
        for d in ALL_DOMAINS:
            try:
                if d not in _engines and _cache_ready(d):
                    get_engine(d)
            except Exception:
                pass
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


# A spelling correction is only used when the match really depends on it. A dictionary can't tell a
# typo from a name it has never seen ("zomato" looks like "tomato"), so a correction must prove itself:
#   1. the corrected question gets a confident answer, no worse than the question as typed, AND
#   2. the matched entry actually CONTAINS the corrected words (at least half of them).
# "how to make biriyani" -> matched "How do I make biryani?" contains "biryani": used.
# "who is sachin tendulkar" -> "who is machine pendular" matched an unrelated entry: not used.
def _words(s: str) -> set:
    return set(re.findall(r"[a-z]+", s.lower()))


def correction_supported(result: dict, changes: list) -> bool:
    if not changes:
        return False
    text = _words(result["matched_question"])
    if result.get("matched_by") == "answer":
        text |= _words(result["answer"])
    stems = {t.rstrip("s") for t in text}
    parts = [p for _, new in changes for p in new.lower().split()]
    hits = sum(1 for p in parts if p in text or p.rstrip("s") in stems)
    return hits * 2 >= len(parts)


def correction_helps(original_result: dict, corrected_result: dict, changes: list) -> bool:
    return (corrected_result["confident"]
            and corrected_result["sbert_score"] >= original_result["sbert_score"]
            and correction_supported(corrected_result, changes))


def best_other_domain(question: str, current_domain: str):
    """Look for a CONFIDENT match in the other domains. Returns (domain, result) or None.
    Used when a question isn't answerable in the chosen domain, e.g. a CS question asked in Cooking."""
    best = None
    for d in ALL_DOMAINS:
        if d == current_domain:
            continue
        r = get_engine(d).query(question)
        if r["confident"] and (best is None or r["sbert_score"] > best[1]["sbert_score"]):
            best = (d, r)
    return best


if __name__ == "__main__":
    print("Building CS engine (first run builds the answer cache too)...")
    engine = get_engine("cs")

    test_questions = [
        "What is tokenization?",
        "How do you break text into smaller units?",
        "What's the way to invert a linked list?",
        "asdkjaskjd random nonsense",
        "how does a computer split a sentence into words",
        "what does overfitting mean in machine learning",
        "explain what a stack is",
    ]
    for q in test_questions:
        r = engine.query(q)
        print(f"\nUSER Q : {q}")
        print(f"MATCHED: {r['matched_question']}  (by {r['matched_by']})")
        print(f"SCORE  : hybrid={r['score']}  tfidf={r['tfidf_score']}  sbert={r['sbert_score']}")
        print(f"CONFIDENT: {r['confident']}")