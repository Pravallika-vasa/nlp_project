"""
data_loader.py
----------------
Reads the 4 raw domain datasets (in their original, differing formats)
and converts each into a standardized DataFrame with columns:
    question, answer, domain

Run this file directly to generate cleaned CSVs into data/clean/.
These cleaned files are what preprocess.py / matcher.py will use.
"""

import os
import re
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
CLEAN_DIR = os.path.join("data", "clean")

# Exact raw filenames -> change these if your filenames differ
FILES = {
    "cs": "1790528915321_NLP_computerScience_dataset.xlsx",
    "gk": "1790528925863_gk_dataset_1200.csv",
    "cooking": "1790528941023_cooking_nlp_dataset_1000_plus.csv",
    "medical": "1790528956947_medquad.csv",
    "cs_intents": "cs_intents.json",       # extra CS Q&A, chatbot-style (multiple phrasings per topic)
    "india_gk": "india_gk.csv",            # curated India-focused GK (PM, symbols, sports, etc.)
    "cs_basics": "cs_basics.csv",          # everyday CS basics + "difference between" questions
}


def load_cs_intents():
    """Optional extra CS file (tag/patterns/responses chatbot format). Skipped if not present."""
    path = os.path.join(RAW_DIR, FILES["cs_intents"])
    if not os.path.exists(path):
        return pd.DataFrame(columns=["question", "answer", "domain"])
    import json
    data = json.load(open(path))
    rows = []
    for item in data["intents"]:
        answer = item["responses"][0].strip()
        for pattern in item["patterns"]:
            rows.append({"question": pattern.strip(), "answer": answer})
    df = pd.DataFrame(rows)
    df["domain"] = "cs"
    return df


def load_cs_basics():
    """Optional extra CS file: everyday basics and comparison questions. Skipped if not present."""
    path = os.path.join(RAW_DIR, FILES["cs_basics"])
    if not os.path.exists(path):
        return pd.DataFrame(columns=["question", "answer", "domain"])
    df = pd.read_csv(path)[["question", "answer"]]
    df["domain"] = "cs"
    return df


def load_india_gk():
    """Optional extra GK file: curated India-focused Q&A. Skipped if not present."""
    path = os.path.join(RAW_DIR, FILES["india_gk"])
    if not os.path.exists(path):
        return pd.DataFrame(columns=["question", "answer", "domain"])
    df = pd.read_csv(path)
    df = df[["question", "answer"]]
    df["domain"] = "gk"
    return df


def load_cs():
    path = os.path.join(RAW_DIR, FILES["cs"])
    df = pd.read_excel(path)
    df = df.rename(columns={"Question": "question", "Answer": "answer"})
    df = df[["question", "answer"]]
    df["domain"] = "cs"
    return df


def load_gk():
    path = os.path.join(RAW_DIR, FILES["gk"])
    df = pd.read_csv(path)
    df = df.rename(columns={"Question": "question", "Answer": "answer"})
    df = df[["question", "answer"]]
    df["domain"] = "gk"
    return df


def load_cooking():
    path = os.path.join(RAW_DIR, FILES["cooking"])
    df = pd.read_csv(path)
    df = df.rename(columns={"Question": "question", "Answer": "answer"})
    df = df[["question", "answer"]]
    df["domain"] = "cooking"
    return df


def load_medical():
    path = os.path.join(RAW_DIR, FILES["medical"])
    df = pd.read_csv(path)
    # already lowercase: question, answer, source, focus_area
    df = df[["question", "answer"]]
    df = df.dropna(subset=["answer"])  # drop the 5 rows with no answer
    df["domain"] = "medical"
    return df


def clean_common(df: pd.DataFrame) -> pd.DataFrame:
    """Shared cleanup applied to every domain after loading."""
    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()
    df = df[(df["question"] != "") & (df["answer"] != "")]
    df = df.drop_duplicates(subset=["question", "domain"])
    df = df.reset_index(drop=True)
    return df



_DEF_START = re.compile(r"^(what is|what are|what do you understand by|define|explain the concept of|explain|describe|tell me about)\s+", re.I)
_ACR = {"nlp": "natural language processing", "ml": "machine learning", "ai": "artificial intelligence",
        "dbms": "database management system", "oop": "object oriented programming", "oops": "object oriented programming"}


def _topic_key(q: str):
    """'What is natural language processing (NLP)?' and 'Define NLP.' -> same topic key.
    Only definition-style questions get a key; 'why/when/scenario' questions are different
    question types and legitimately have different answers."""
    q = q.lower().strip().rstrip("?.! ")
    if not _DEF_START.match(q):
        return None
    q = _DEF_START.sub("", q)
    q = re.sub(r"\s*\([^)]*\)", "", q)            # drop "(NLP)"
    q = re.sub(r"^(a|an|the)\s+", "", q).strip()
    return " ".join(_ACR.get(w, w) for w in q.split())


def unify_definition_answers(df: pd.DataFrame) -> pd.DataFrame:
    """If several definition-style questions ask about the same topic but store different
    answers (e.g. a short Excel answer and a long intents.json answer), give all of them the
    most informative (longest) answer, so the reply doesn't depend on how the user phrased it."""
    keys = df["question"].map(_topic_key)
    changed_topics = 0
    df = df.copy()
    for key, idx in df[keys.notna()].groupby(keys[keys.notna()]).groups.items():
        answers = df.loc[idx, "answer"]
        if answers.nunique() > 1:
            df.loc[idx, "answer"] = max(answers.unique(), key=len)
            changed_topics += 1
    print(f"     unified answers for {changed_topics} topics that had conflicting definitions")
    return df


def build_all():
    os.makedirs(CLEAN_DIR, exist_ok=True)

    loaders = {
        "cs": load_cs,
        "gk": load_gk,
        "cooking": load_cooking,
        "medical": load_medical,
    }
    extra_loaders = {
        "cs": [load_cs_intents, load_cs_basics],   # merged INTO cs, not separate domains
        "gk": [load_india_gk],
    }

    summary = {}
    for domain, loader in loaders.items():
        df = loader()
        for extra in extra_loaders.get(domain, []):
            extra_df = extra()
            if len(extra_df) > 0:
                before = len(df)
                df = pd.concat([df, extra_df], ignore_index=True)
                print(f"     + merged {len(extra_df)} extra {domain} rows from {extra.__name__} (was {before})")
        df = clean_common(df)
        if domain == "cs":
            df = unify_definition_answers(df)
            # the Excel source is sorted by category (NLP first); mix the rows so the file is balanced to read
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        out_path = os.path.join(CLEAN_DIR, f"{domain}.csv")
        df.to_csv(out_path, index=False)
        summary[domain] = len(df)
        print(f"[OK] {domain}: {len(df)} question-answer pairs -> {out_path}")

    print("\nSummary:", summary)
    return summary


if __name__ == "__main__":
    build_all()
