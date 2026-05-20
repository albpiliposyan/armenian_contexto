from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

MODELS_DIR = PROJECT_ROOT / "models"
FASTTEXT_DIR = MODELS_DIR / "fasttext"

WORD_FREQ_FILE = RAW_DATA_DIR / "wordfreq-en-25000-log.json"
CLEANED_WORDS_FILE = INTERIM_DATA_DIR / "cleaned_words.json"
CONTEXTO_WORDS_FILE = INTERIM_DATA_DIR / "contexto_words.json"

ARMENIAN_WORDS_FILE = PROCESSED_DATA_DIR / "armenian_contexto_words.json"
ARMENIAN_WORDS_TOP5000_FILE = PROCESSED_DATA_DIR / "armenian_contexto_words_top5000.json"
FAILED_TRANSLATIONS_FILE = PROCESSED_DATA_DIR / "failed_translations.json"
FAILED_TRANSLATIONS_TOP5000_FILE = PROCESSED_DATA_DIR / "failed_translations_top5000.json"
METADATA_FILE = PROCESSED_DATA_DIR / "armenian_contexto_metadata.json"

EMBEDDINGS_FILE = EMBEDDINGS_DIR / "armenian_contexto_embeddings.npz"
ENGLISH_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "english_contexto_embeddings.npz"
FASTTEXT_MODEL_FILE = FASTTEXT_DIR / "cc.hy.300.bin"
FASTTEXT_MODEL_GZ_FILE = FASTTEXT_DIR / "cc.hy.300.bin.gz"
