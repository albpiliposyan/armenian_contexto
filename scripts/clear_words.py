import json
import re
import sys
from pathlib import Path

import spacy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from armenian_contexto.paths import CLEANED_WORDS_FILE, WORD_FREQ_FILE  # noqa: E402


INPUT_FILE = WORD_FREQ_FILE
OUTPUT_FILE = CLEANED_WORDS_FILE


def main():
    nlp = spacy.load("en_core_web_sm")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_words = {}

    for item in data:
        word = item[0]
        frequency = item[1]

        word = word.lower().strip()

        if not re.fullmatch(r"[a-z]+", word):
            continue

        doc = nlp(word)
        lemma = doc[0].lemma_.lower()

        if not re.fullmatch(r"[a-z]+", lemma):
            continue

        if lemma not in cleaned_words:
            cleaned_words[lemma] = frequency
        else:
            cleaned_words[lemma] = max(cleaned_words[lemma], frequency)

    output_data = [
        {"word": word, "frequency": freq}
        for word, freq in cleaned_words.items()
    ]
    output_data.sort(key=lambda x: x["frequency"], reverse=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(output_data)} cleaned words to {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
