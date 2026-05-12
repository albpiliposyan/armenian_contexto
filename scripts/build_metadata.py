import sys
from pathlib import Path

import numpy as np
import spacy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from armenian_contexto.english_vectors import HashEnglishVectorizer  # noqa: E402
from armenian_contexto.metadata import (  # noqa: E402
    build_category_lookup,
    category_for_english,
    load_category_map,
    load_json,
    save_json,
)
from armenian_contexto.paths import (  # noqa: E402
    ARMENIAN_WORDS_TOP5000_FILE,
    CATEGORIES_FILE,
    ENGLISH_EMBEDDINGS_FILE,
    METADATA_FILE,
)


def primary_translation(item):
    translations = item.get("translations", [])
    if not translations:
        return "", 0.0

    first = translations[0]
    return first.get("en", ""), first.get("frequency", 0.0)


def pos_tag(nlp, english_word: str) -> str | None:
    if not english_word:
        return None

    doc = nlp(english_word)
    if not doc:
        return None

    return doc[0].pos_


def english_vector(nlp, fallback_vectorizer, english_word: str) -> np.ndarray:
    doc = nlp(english_word)
    vector = doc.vector.astype(np.float32)
    norm = np.linalg.norm(vector)

    if vector.size and norm:
        return vector / norm

    if vector.size:
        return vector

    return fallback_vectorizer.vectorize(english_word)


def main():
    vocabulary = load_json(ARMENIAN_WORDS_TOP5000_FILE, [])
    category_lookup = build_category_lookup(load_category_map(CATEGORIES_FILE))
    vectorizer = HashEnglishVectorizer()
    nlp = spacy.load("en_core_web_sm")

    metadata = []
    english_vectors = []
    words = []

    for item in vocabulary:
        hy_word = item["hy"]
        en_word, frequency = primary_translation(item)

        metadata.append({
            "hy": hy_word,
            "en": en_word,
            "frequency": frequency,
            "pos": pos_tag(nlp, en_word),
            "category": category_for_english(en_word, category_lookup),
        })

        words.append(hy_word)
        english_vectors.append(english_vector(nlp, vectorizer, en_word))

    save_json(METADATA_FILE, metadata)

    ENGLISH_EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ENGLISH_EMBEDDINGS_FILE,
        words=np.array(words),
        embeddings=np.array(english_vectors, dtype=np.float32),
    )

    print(f"Saved {len(metadata)} metadata rows to {METADATA_FILE.relative_to(PROJECT_ROOT)}")
    print(f"Saved English vectors to {ENGLISH_EMBEDDINGS_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
