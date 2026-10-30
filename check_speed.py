"""
check_speed.py
---------------
Shows how long each part of start-up and answering really takes on YOUR computer, so you
can see where any waiting comes from.   Run:   py check_speed.py
(Run  py build_cache.py  first, otherwise "load index" steps will include a slow rebuild.)
"""
import time

def step(label, fn):
    t = time.time(); out = fn(); dt = time.time() - t
    print(f"{label:<46}{dt:7.1f} s"); return out, dt

print("\n--- START-UP (happens once each time you run streamlit) ---")
_, t_import = step("import PyTorch + sentence-transformers", lambda: __import__("sentence_transformers"))
from matcher import get_sbert_model, get_engine, best_other_domain, ALL_DOMAINS
from preprocess import preprocess_text
step("load NLTK language data (first question only)", lambda: preprocess_text("warm up"))
_, t_model = step("load the SBERT model into memory", get_sbert_model)
engines, t_idx = {}, 0
for d in ALL_DOMAINS:
    e, dt = step(f"load '{d}' index from the saved cache", lambda d=d: get_engine(d))
    engines[d] = e; t_idx += dt
    if dt > 20:
        print("    ^ slow: the cache for this domain is missing or outdated -> run  py build_cache.py")
from textfix import build_vocab, SpellFixer
fixer, t_spell = step("load the spelling corrector", lambda: SpellFixer(build_vocab()))
total = t_import + t_model + t_idx + t_spell
print(f"{'TOTAL start-up':<46}{total:7.1f} s")

print("\n--- ANSWERING (what you wait for after pressing Enter) ---")
samples = {"cs": ["what is a stack", "difference between tcp and udp", "what is nlp"],
           "cooking": ["how to make biryani", "what is baking soda"],
           "gk": ["who is the prime minister of india", "national bird of india"],
           "medical": ["what are the symptoms of diabetes", "what is cancer"]}
for d, qs in samples.items():
    times = []
    for q in qs:
        t = time.time(); engines[d].query(q); times.append(time.time() - t)
    print(f"{'answer one question in ' + d:<46}{sum(times)/len(times)*1000:7.0f} ms")
t = time.time(); fixer.fix("explain about adrays and polymorfism"); dt = time.time() - t
print(f"{'fix spelling mistakes in a question':<46}{dt*1000:7.0f} ms")
t = time.time(); best_other_domain("what is a compiler", "cooking"); dt = time.time() - t
print(f"{'cross-domain check (3 other domains)':<46}{dt*1000:7.0f} ms")

print("\nHow to read this: a question should take well under 1 second. Most start-up time is the")
print("PyTorch import + model load; that happens once per 'streamlit run', not per question or per user.")
