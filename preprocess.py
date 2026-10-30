"""
preprocess.py
--------------
NLTK-based text preprocessing for the QA system.

Pipeline for each piece of text:
    1. Lowercase
    2. Remove punctuation / non-alphabetic tokens
    3. Tokenize (word_tokenize)
    4. POS tag (so lemmatizer knows noun vs verb vs adjective -> better lemmas)
    5. Remove stopwords
    6. Lemmatize (WordNetLemmatizer, using POS tag)

Output of preprocess_text() is a single cleaned string, e.g.:
    "What is the process of reversing a linked list?"
    -> "process reverse link list"

This cleaned string is what gets fed into TF-IDF.
(SBERT, later in matcher.py, will use the ORIGINAL raw text instead,
because sentence embedding models are trained on natural sentences,
not stripped-down keyword lists — stripping stopwords/lemmatizing
before SBERT actually hurts its accuracy. This is a deliberate design
choice, good to mention in your report.)
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def get_wordnet_pos(tag: str) -> str:
    """Map NLTK POS tag to a WordNet POS tag for accurate lemmatization."""
    if tag.startswith("J"):
        return "a"  # adjective
    elif tag.startswith("V"):
        return "v"  # verb
    elif tag.startswith("N"):
        return "n"  # noun
    elif tag.startswith("R"):
        return "r"  # adverb
    else:
        return "n"  # default to noun


def clean_text(text: str) -> str:
    """Lowercase and strip out non-alphabetic characters (keep spaces)."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str) -> str:
    """
    Full pipeline: clean -> tokenize -> POS tag -> remove stopwords -> lemmatize.
    Returns a single space-joined string of lemmas, ready for TF-IDF.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    tagged = pos_tag(tokens)

    lemmas = []
    for word, tag in tagged:
        if word in STOPWORDS or len(word) <= 1:
            continue
        wn_tag = get_wordnet_pos(tag)
        lemma = LEMMATIZER.lemmatize(word, pos=wn_tag)
        lemmas.append(lemma)

    return " ".join(lemmas)


if __name__ == "__main__":
    # Quick manual test with a few real-style questions
    samples = [
        "What is the process of reversing a linked list?",
        "How do I make adobo (Filipino)?",
        "What is (are) Glaucoma ?",
        "What's the way to invert a linked list?",  # paraphrase test
    ]
    for s in samples:
        print(f"RAW : {s}")
        print(f"PROC: {preprocess_text(s)}")
        print()
