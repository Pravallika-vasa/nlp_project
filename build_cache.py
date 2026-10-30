"""
build_cache.py
---------------
Run this after changing any dataset (py data_loader.py, then py build_cache.py).
It builds and saves, into data/cache/:
  - the TF-IDF and SBERT indexes for all 4 domains
  - the spelling word list used to correct typos
Domains whose data did NOT change keep their old cache automatically (so the big Medical
set is only rebuilt when MedQuAD itself changes). After this, the chatbot starts fast.
"""
import time

t0 = time.time()
print("Loading libraries (torch can take a while the first time)...")
from matcher import get_engine
from textfix import build_vocab

for d in ["cs", "gk", "cooking", "medical"]:
    t = time.time()
    engine = get_engine(d)
    print(f"[OK] {d:8s} index ready in {time.time() - t:6.1f}s")
    t = time.time()
    engine.query("test question")
    print(f"     one question takes {time.time() - t:.2f}s")

t = time.time()
vocab = build_vocab()
print(f"[OK] spelling word list ready in {time.time() - t:.1f}s ({len(vocab['all']):,} words)")

print(f"\nAll done in {time.time() - t0:.0f}s. Now run:  py -m streamlit run app.py")
