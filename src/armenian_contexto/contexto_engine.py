from functools import lru_cache
import re
from pathlib import Path

import fasttext
import numpy as np

from .metadata import (
    build_category_lookup,
    category_for_english,
    load_category_map,
    load_metadata,
)
from .paths import (
    CATEGORIES_FILE,
    EMBEDDINGS_FILE,
    ENGLISH_EMBEDDINGS_FILE,
    FASTTEXT_MODEL_FILE,
    METADATA_FILE,
)


ARMENIAN_SUFFIXES = [
    "ներով", "ներից", "ներին", "ների", "ները",
    "ում", "ից", "ով", "ին", "ը", "ն",
]

HYBRID_WEIGHTS = {
    "armenian": 0.70,
    "category": 0.15,
    "pos": 0.10,
    "english": 0.05,
}


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
        metadata_file: str | Path = METADATA_FILE,
        categories_file: str | Path = CATEGORIES_FILE,
        english_embeddings_file: str | Path = ENGLISH_EMBEDDINGS_FILE,
    ):
        self.embeddings_file = Path(embeddings_file)
        self.fasttext_model_file = Path(fasttext_model_file)
        self.metadata_file = Path(metadata_file)
        self.categories_file = Path(categories_file)
        self.english_embeddings_file = Path(english_embeddings_file)
        self.model = None

        data = np.load(self.embeddings_file, allow_pickle=True)

        self.words = np.array([str(word) for word in data["words"]])
        self.embeddings = data["embeddings"]

        self.word_to_index = {
            word: idx for idx, word in enumerate(self.words)
        }

        category_map = load_category_map(self.categories_file)
        self.category_lookup = build_category_lookup(category_map)
        self.metadata_by_word = load_metadata(self.metadata_file)
        self.pos_tags, self.categories = self._metadata_arrays()
        self.english_embeddings = self._load_aligned_english_embeddings()

    def process_word(self, word: str) -> str:
        word = normalize_text(word)
        stemmed = stem_armenian(word)

        if stemmed in self.word_to_index:
            return stemmed

        return word

    def _load_fasttext_model(self):
        if self.model is None:
            print("Loading FastText model...")
            self.model = fasttext.load_model(str(self.fasttext_model_file))
        return self.model

    def get_vector(self, word: str):
        word = self.process_word(word)

        if word in self.word_to_index:
            return self.embeddings[self.word_to_index[word]]

        vec = self._load_fasttext_model().get_word_vector(word)
        return normalize_vector(vec)

    def similarity(self, word1: str, word2: str) -> float:
        """Pure Armenian FastText cosine similarity."""
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)

        return float(vec1 @ vec2)

    def _metadata_arrays(self):
        pos_tags = []
        categories = []

        for word in self.words:
            metadata = self.metadata_by_word.get(word, {})
            english_word = metadata.get("en", "")

            pos_tags.append(metadata.get("pos"))
            categories.append(
                metadata.get("category")
                or category_for_english(english_word, self.category_lookup)
            )

        return np.array(pos_tags, dtype=object), np.array(categories, dtype=object)

    def _load_aligned_english_embeddings(self):
        if not self.english_embeddings_file.exists():
            return np.zeros((len(self.words), 0), dtype=np.float32)

        data = np.load(self.english_embeddings_file, allow_pickle=True)
        file_words = [str(word) for word in data["words"]]
        vectors = data["embeddings"].astype(np.float32)

        if file_words == list(self.words):
            aligned = vectors
        else:
            aligned = np.zeros((len(self.words), vectors.shape[1]), dtype=np.float32)
            file_word_to_index = {
                word: idx for idx, word in enumerate(file_words)
            }
            for idx, word in enumerate(self.words):
                source_idx = file_word_to_index.get(word)
                if source_idx is not None:
                    aligned[idx] = vectors[source_idx]

        norms = np.linalg.norm(aligned, axis=1, keepdims=True)
        return np.divide(
            aligned,
            norms,
            out=np.zeros_like(aligned),
            where=norms != 0,
        )

    def _metadata_for_word(self, word: str):
        return self.metadata_by_word.get(self.process_word(word), {})

    def pos_for_word(self, word: str) -> str | None:
        return self._metadata_for_word(word).get("pos")

    def category_for_word(self, word: str) -> str | None:
        metadata = self._metadata_for_word(word)
        return (
            metadata.get("category")
            or category_for_english(metadata.get("en", ""), self.category_lookup)
        )

    def same_category(self, word1: str, word2: str) -> float:
        category1 = self.category_for_word(word1)
        category2 = self.category_for_word(word2)
        return 1.0 if category1 and category1 == category2 else 0.0

    def same_pos(self, word1: str, word2: str) -> float:
        pos1 = self.pos_for_word(word1)
        pos2 = self.pos_for_word(word2)
        return 1.0 if pos1 and pos1 == pos2 else 0.0

    def get_english_vector(self, word: str):
        word = self.process_word(word)
        if word in self.word_to_index:
            return self.english_embeddings[self.word_to_index[word]]

        return np.zeros(self.english_embeddings.shape[1], dtype=np.float32)

    def english_similarity(self, word1: str, word2: str) -> float:
        if self.english_embeddings.shape[1] == 0:
            return 0.0

        vec1 = self.get_english_vector(word1)
        vec2 = self.get_english_vector(word2)

        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0

        return float(vec1 @ vec2)

    def hybrid_components(self, word1: str, word2: str):
        armenian_score = self.similarity(word1, word2)
        category_score = self.same_category(word1, word2)
        pos_score = self.same_pos(word1, word2)
        english_score = self.english_similarity(word1, word2)

        # Hybrid scoring makes gameplay less noisy than pure embeddings:
        # Armenian FastText provides broad semantic distance, while category,
        # POS, and English translation signals reward words that are related in
        # ways FastText alone can miss or over-smooth.
        final_score = (
            HYBRID_WEIGHTS["armenian"] * armenian_score
            + HYBRID_WEIGHTS["category"] * category_score
            + HYBRID_WEIGHTS["pos"] * pos_score
            + HYBRID_WEIGHTS["english"] * english_score
        )

        return {
            "armenian_fasttext": round(float(armenian_score), 6),
            "same_category": category_score,
            "same_pos": pos_score,
            "english_similarity": round(float(english_score), 6),
            "final_score": round(float(final_score), 6),
        }

    def hybrid_similarity(self, word1: str, word2: str) -> float:
        return self.hybrid_components(word1, word2)["final_score"]

    @lru_cache(maxsize=128)
    def _hybrid_scores_against(self, target: str):
        target = self.process_word(target)
        target_vec = self.get_vector(target)

        armenian_scores = self.embeddings @ target_vec

        target_category = self.category_for_word(target)
        if target_category:
            category_scores = (self.categories == target_category).astype(np.float32)
        else:
            category_scores = np.zeros(len(self.words), dtype=np.float32)

        target_pos = self.pos_for_word(target)
        if target_pos:
            pos_scores = (self.pos_tags == target_pos).astype(np.float32)
        else:
            pos_scores = np.zeros(len(self.words), dtype=np.float32)

        if self.english_embeddings.shape[1] == 0:
            english_scores = np.zeros(len(self.words), dtype=np.float32)
        else:
            target_english_vec = self.get_english_vector(target)
            if np.linalg.norm(target_english_vec) == 0:
                english_scores = np.zeros(len(self.words), dtype=np.float32)
            else:
                english_scores = self.english_embeddings @ target_english_vec

        # Ranking is vectorized over the full vocabulary: compute one hybrid
        # score per candidate, then count how many candidates are closer to the
        # target than the user's guess.
        return (
            HYBRID_WEIGHTS["armenian"] * armenian_scores
            + HYBRID_WEIGHTS["category"] * category_scores
            + HYBRID_WEIGHTS["pos"] * pos_scores
            + HYBRID_WEIGHTS["english"] * english_scores
        )

    def get_rank(self, guess: str, target: str):
        guess_processed = self.process_word(guess)
        target_processed = self.process_word(target)

        if guess_processed == target_processed:
            return {
                "guess": guess,
                "processed_guess": guess_processed,
                "target": target_processed,
                "similarity": 1.0,
                "score": 1.0,
                "rank": 1,
                "is_correct": True,
                "components": self.hybrid_components(guess_processed, target_processed),
            }

        all_scores = self._hybrid_scores_against(target_processed)

        if guess_processed in self.word_to_index:
            guess_score = float(all_scores[self.word_to_index[guess_processed]])
        else:
            guess_score = self.hybrid_similarity(guess_processed, target_processed)

        rank = int(np.sum(all_scores > guess_score)) + 1

        return {
            "guess": guess,
            "processed_guess": guess_processed,
            "target": target_processed,
            "similarity": round(guess_score, 6),
            "score": round(guess_score, 6),
            "rank": rank,
            "is_correct": False,
            "components": self.hybrid_components(guess_processed, target_processed),
        }

    def closest_words(self, target: str, top_k: int = 20):
        target_processed = self.process_word(target)
        scores = self._hybrid_scores_against(target_processed)

        indices = np.argsort(-scores)[:top_k]

        return [
            {
                "word": str(self.words[i]),
                "similarity": round(float(scores[i]), 6),
                "score": round(float(scores[i]), 6),
                "rank": rank + 1,
            }
            for rank, i in enumerate(indices)
        ]
