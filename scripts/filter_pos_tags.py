import json
import re
import sys
from pathlib import Path

import spacy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from armenian_contexto.paths import CLEANED_WORDS_FILE, CONTEXTO_WORDS_FILE  # noqa: E402


INPUT_FILE = CLEANED_WORDS_FILE
OUTPUT_FILE = CONTEXTO_WORDS_FILE

ALLOWED_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


def main():
    nlp = spacy.load("en_core_web_sm")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = {}

    for item in data:
        word = item["word"]
        frequency = item["frequency"]

        if not re.fullmatch(r"[a-z]+", word):
            continue

        doc = nlp(word)
        token = doc[0]

        if token.is_stop:
            continue

        if token.pos_ not in ALLOWED_POS:
            continue

        if len(word) < 3:
            continue

        filtered[word] = frequency

    output_data = [
        {"word": word, "frequency": freq}
        for word, freq in filtered.items()
    ]

    output_data.sort(key=lambda x: x["frequency"], reverse=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(output_data)} Contexto-friendly words to {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
