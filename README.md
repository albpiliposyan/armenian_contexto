# Armenian Contexto

An Armenian implementation of a semantic word guessing game inspired by
Contexto.

The game does not compare letters. It compares Armenian word embeddings with
cosine similarity and returns semantic ranks for guesses.

Example:

```text
Target word: դպրոց

ուսուցիչ  -> Rank #12
աշակերտ  -> Rank #18
գիրք      -> Rank #145
մեքենա    -> Rank #4200
դպրոցը    -> WIN
```

## Project Layout

```text
armenian_contexto/
├── data/
│   ├── raw/          # Original English frequency dataset
│   ├── interim/      # Cleaned and POS-filtered English words
│   ├── processed/    # Armenian vocabulary and translation logs
│   └── embeddings/   # Production-ready embedding matrix
├── models/
│   └── fasttext/     # Armenian FastText model files
├── scripts/          # Dataset, translation, and embedding build scripts
├── src/
│   └── armenian_contexto/
│       ├── contexto_engine.py
│       └── paths.py
└── tests/
    └── test_game.py
```

All code paths are centralized in
`src/armenian_contexto/paths.py`, so scripts can be run from the project root
without relying on flat-file paths.

## Main Files

- `data/raw/wordfreq-en-25000-log.json`: original English frequency dataset.
- `data/interim/cleaned_words.json`: lemmatized and deduplicated English words.
- `data/interim/contexto_words.json`: POS-filtered Contexto-friendly words.
- `data/processed/armenian_contexto_words_top5000.json`: main Armenian vocabulary.
- `data/processed/failed_translations_top5000.json`: rejected translation records.
- `models/fasttext/cc.hy.300.bin`: Armenian FastText model.
- `data/embeddings/armenian_contexto_embeddings.npz`: normalized Armenian embedding matrix.
- `src/armenian_contexto/contexto_engine.py`: semantic ranking engine.

## Git Policy

Keep source code, documentation, package metadata, and small reproducibility
artifacts in Git.

Track:

- `src/`
- `scripts/`
- `tests/`
- `README.md`
- `pyproject.toml`
- `requirements.txt`
- `.gitignore`
- small JSON datasets under `data/`
- `data/embeddings/armenian_contexto_embeddings.npz`

Do not track:

- `venv/` or `.venv/`
- `__pycache__/`
- `.egg-info/`
- `models/fasttext/cc.hy.300.bin`
- `models/fasttext/cc.hy.300.bin.gz`

The FastText files are multi-GB external artifacts. Keep them local, document
where to download them, or manage them with Git LFS, DVC, or a model registry if
the project later needs formal artifact versioning.

## Setup

Recommended Python version: 3.11.

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m spacy download en_core_web_sm
```

The project already expects the Armenian FastText model at:

```text
models/fasttext/cc.hy.300.bin
```

## Workflow

Run commands from the project root.

Inspect top English frequency words:

```bash
python scripts/word_freq.py
```

Clean and lemmatize the English frequency dataset:

```bash
python scripts/clear_words.py
```

Filter for Contexto-friendly words:

```bash
python scripts/filter_pos_tags.py
```

Translate the top English words to Armenian:

```bash
python scripts/translate_to_armenian.py
```

Build normalized Armenian embeddings:

```bash
python scripts/build_embeddings.py
```

Run the gameplay smoke test:

```bash
python tests/test_game.py
```

## Engine Usage

```python
from armenian_contexto import ArmenianContextoEngine

engine = ArmenianContextoEngine()
result = engine.get_rank("ուսուցիչ", "դպրոց")
print(result)
```

If the package is not installed, run with:

```bash
PYTHONPATH=src python -c "from armenian_contexto import ArmenianContextoEngine; print(ArmenianContextoEngine)"
```

## Core Logic

The engine uses:

```text
similarity = embedding(word1) @ embedding(word2)
```

Embeddings are normalized, so the dot product is cosine similarity.

The rank is:

```text
number of vocabulary words more similar to the target + 1
```

## Winning Condition

The winning condition is exact after Armenian normalization and lightweight
stemming. It is not epsilon-based semantic similarity.

Examples:

```text
դպրոցը -> դպրոց -> correct
ուսուցիչ -> ուսուցիչ -> close, but not a win
```

Current Armenian normalization handles:

- lowercasing
- Armenian punctuation cleanup
- `եւ` to `և`
- simple suffix stripping for definite articles, plurals, and common case endings

## Limitations

- Translation quality depends on Google-based translation output.
- Armenian stemming is heuristic-based, not a full morphological analyzer.
- FastText can produce semantic noise for abstract or ambiguous words.

## Future Work

- Add a proper Armenian lemmatizer or Stanza-based morphology.
- Add daily target words and difficulty levels.
- Add a FastAPI backend around `ArmenianContextoEngine`.
- Add a React or Next.js frontend for gameplay.
