import re
from pathlib import Path

import fasttext
import numpy as np

from .paths import EMBEDDINGS_FILE, FASTTEXT_MODEL_FILE


ARMENIAN_SUFFIXES = [
    "ներով", "ներից", "ներին", "ների", "ները",
    "ում", "ից", "ով", "ին", "ը", "ն",
]


def normalize_text(word: str) -> str:
    word = word.strip().lower()
    word = word.replace("եւ", "և")
    word = re.sub(r"[^ա-ֆև]", "", word)
    return word


def stem_armenian(word: str) -> str:
    for suffix in sorted(ARMENIAN_SUFFIXES, key=len, reverse=True):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word


def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec


class ArmenianContextoEngine:
    def __init__(
        self,
        embeddings_file: str | Path = EMBEDDINGS_FILE,
        fasttext_model_file: str | Path = FASTTEXT_MODEL_FILE,
    ):
        self.embeddings_file = Path(embeddings_file)
        self.fasttext_model_file = Path(fasttext_model_file)

        data = np.load(self.embeddings_file, allow_pickle=True)

        self.words = data["words"]
        self.embeddings = data["embeddings"]

        self.word_to_index = {
            word: idx for idx, word in enumerate(self.words)
        }

        print("Loading FastText model...")
        self.model = fasttext.load_model(str(self.fasttext_model_file))

    def process_word(self, word: str) -> str:
        word = normalize_text(word)
        stemmed = stem_armenian(word)

        if stemmed in self.word_to_index:
            return stemmed

        return word

    def get_vector(self, word: str):
        word = self.process_word(word)

        if word in self.word_to_index:
            return self.embeddings[self.word_to_index[word]]

        vec = self.model.get_word_vector(word)
        return normalize_vector(vec)

    def similarity(self, word1: str, word2: str) -> float:
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)

        return float(vec1 @ vec2)

    def get_rank(self, guess: str, target: str):
        guess_processed = self.process_word(guess)
        target_processed = self.process_word(target)

        if guess_processed == target_processed:
            return {
                "guess": guess,
                "processed_guess": guess_processed,
                "target": target_processed,
                "similarity": 1.0,
                "rank": 1,
                "is_correct": True,
            }

        target_vec = self.get_vector(target_processed)
        guess_vec = self.get_vector(guess_processed)

        all_similarities = self.embeddings @ target_vec
        guess_similarity = float(guess_vec @ target_vec)

        rank = int(np.sum(all_similarities > guess_similarity)) + 1

        return {
            "guess": guess,
            "processed_guess": guess_processed,
            "target": target_processed,
            "similarity": round(guess_similarity, 6),
            "rank": rank,
            "is_correct": False,
        }

    def closest_words(self, target: str, top_k: int = 20):
        target_vec = self.get_vector(target)
        similarities = self.embeddings @ target_vec

        indices = np.argsort(-similarities)[:top_k]

        return [
            {
                "word": str(self.words[i]),
                "similarity": round(float(similarities[i]), 6),
                "rank": rank + 1,
            }
            for rank, i in enumerate(indices)
        ]
