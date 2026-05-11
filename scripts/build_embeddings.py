import json
import sys
from pathlib import Path

import numpy as np
import fasttext


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from armenian_contexto.paths import (  # noqa: E402
    ARMENIAN_WORDS_TOP5000_FILE,
    EMBEDDINGS_FILE,
    FASTTEXT_MODEL_FILE,
)


WORDS_FILE = ARMENIAN_WORDS_TOP5000_FILE
FASTTEXT_MODEL = FASTTEXT_MODEL_FILE
OUTPUT_FILE = EMBEDDINGS_FILE


def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec


def main():
    print("Loading FastText model...")
    model = fasttext.load_model(str(FASTTEXT_MODEL))

    with open(WORDS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = []
    vectors = []

    for item in data:
        hy_word = item["hy"]

        vec = model.get_word_vector(hy_word)
        vec = normalize_vector(vec)

        words.append(hy_word)
        vectors.append(vec)

    vectors = np.array(vectors, dtype=np.float32)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_FILE,
        words=np.array(words),
        embeddings=vectors,
    )

    print(f"Saved {len(words)} words to {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    print(f"Embedding matrix shape: {vectors.shape}")


if __name__ == "__main__":
    main()
