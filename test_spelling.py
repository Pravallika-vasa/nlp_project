"""
test_spelling.py
-----------------
Shows, with the REAL model on your computer, what the spelling corrector does to a question and
whether the correction is used.   Run:   py test_spelling.py
Add your own lines to TESTS (domain, question as typed) to try more.
"""
from textfix import SpellFixer, build_vocab
from matcher import get_engine, correction_helps

TESTS = [
    ("medical", "how to prevent hart attack"), ("medical", "hart"), ("medical", "sympotms of diabetis"),
    ("medical", "what causes asthama"), ("medical", "what is migrane"), ("medical", "heartattack symptoms"),
    ("medical", "what is glaucomma"), ("medical", "tretment for depresion"),
    ("cs", "explain about adrays"), ("cs", "what is polymorfism"), ("cs", "what is a compilar"),
    ("cooking", "how to make biriyani"), ("cooking", "how to bake a chocalate cake"),
    ("gk", "who is the prime ministar of india"), ("gk", "national animl of india"),
    # names / new words that must NOT be 'corrected':
    ("gk", "who is sachin tendulkar"), ("cooking", "what is zomato"), ("cs", "what is kotlin"),
]

fixer = SpellFixer(build_vocab())
print("(a correction is used only if it gives a confident answer, no worse than the typed question, and the matched entry contains the corrected words)\n")
for domain, q in TESTS:
    fixed, changes = fixer.fix_detailed(q)
    eng = get_engine(domain)
    r0 = eng.query(q)
    line = f"[{domain:7}] {q!r}\n    typed    : score {r0['sbert_score']:.2f}  {'ANSWERS' if r0['confident'] else 'no answer':9} -> {r0['matched_question'][:55]!r}"
    if changes:
        r1 = eng.query(fixed)
        used = correction_helps(r0, r1, changes)
        fx = ", ".join(f"{a}->{b}" for a, b in changes)
        line += (f"\n    corrected: score {r1['sbert_score']:.2f}  {'ANSWERS' if r1['confident'] else 'no answer':9} -> {r1['matched_question'][:55]!r}"
                 f"\n    words changed: {fx}   => {'CORRECTION USED' if used else 'correction NOT used'}")
    else:
        line += "\n    (no spelling change needed)"
    print(line + "\n")
