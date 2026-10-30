# Question Answering System (4 domains)

Computer Science, Cooking, General Knowledge and Medical. Pick a domain, then ask in your own words.

## How it answers
1. **Spelling correction** of every word of the question, using our own datasets plus the sentence context ("hart attack" -> "heart attack"; glued words are split). A correction is used only if the match really depends on it, and the reply lists the words it changed. Try it with `py test_spelling.py`.
2. **Acronyms** are expanded (nlp -> natural language processing) so both forms reach the same entry.
3. **Matching** = TF-IDF + cosine similarity (on NLTK-preprocessed text) combined with Sentence-BERT embeddings
   (compared with stored questions and, if needed, stored answers). Hybrid score = 0.35 TF-IDF + 0.65 SBERT.
4. **Confidence gate**: answers only if the SBERT score is at least 0.60.
5. If the question isn't answerable in the chosen domain but matches another domain, the bot says which domain it
   belongs to and still gives the answer.
6. Greetings and normal conversation ("hi", "thanks", "who are you") get friendly replies.

## Setup (once)
```
py -m pip install -r requirements.txt
py setup_nltk.py
py data_loader.py          # builds data/clean/*.csv from data/raw
py build_cache.py          # builds the saved indexes + spelling word list (Medical is slow the first time)
py check_speed.py          # optional: shows how long start-up and answering take on your computer
py -m streamlit run app.py
```
Streamlit's file watcher is switched off in `.streamlit/config.toml` (it made start-up very slow with PyTorch),
so after editing a .py file, stop the app (Ctrl + C) and run it again.

## After changing a dataset
Run `py data_loader.py` and then `py build_cache.py`. Only the domains whose data changed are rebuilt.

## Files
- `data_loader.py` merges all sources (Excel, CSV, JSON) into `data/clean/*.csv`, makes definition answers consistent per topic.
- `preprocess.py` NLTK pipeline. `textfix.py` acronyms + spelling correction. `smalltalk.py` conversation replies.
- `matcher.py` TF-IDF + SBERT engine. `app.py` Streamlit chat interface. `db.py` SQLite users and chat history.
- `build_cache.py`, `check_speed.py`, `setup_nltk.py` helper scripts.
