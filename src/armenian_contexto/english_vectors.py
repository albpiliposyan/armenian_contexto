import hashlib

import numpy as np

from .metadata import normalize_english_word


class HashEnglishVectorizer:
    """Small deterministic English vectorizer used when no large English model is shipped.

    The project avoids committing another multi-GB English FastText model. This
    vectorizer builds normalized character n-gram vectors so English translation
    similarity can still contribute to the hybrid score in a reproducible way.
    """

    def __init__(self, dimensions: int = 128, min_n: int = 3, max_n: int = 5):
        self.dimensions = dimensions
        self.min_n = min_n
        self.max_n = max_n

    def vectorize(self, word: str) -> np.ndarray:
        normalized = normalize_english_word(word)
        vector = np.zeros(self.dimensions, dtype=np.float32)

        if not normalized:
            return vector

        padded = f"<{normalized}>"
        for ngram_size in range(self.min_n, self.max_n + 1):
            for start in range(0, len(padded) - ngram_size + 1):
                ngram = padded[start:start + ngram_size]
                digest = hashlib.blake2b(ngram.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                index = value % self.dimensions
                sign = 1.0 if (value >> 63) == 0 else -1.0
                vector[index] += sign

        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector
