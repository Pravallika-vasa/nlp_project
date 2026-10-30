"""
Run this ONCE after installing requirements, before running the app.
Downloads the NLTK data packages preprocess.py needs.
"""
import nltk

for pkg in [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]:
    nltk.download(pkg)

print("\nAll NLTK data downloaded. You're ready to run the app.")
