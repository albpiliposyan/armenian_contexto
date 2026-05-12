# Armenian Contexto

An Armenian implementation of a semantic word guessing game inspired by
Contexto.

The game does not compare letters. It combines Armenian embedding similarity,
English translation similarity, POS tags, and lightweight semantic categories
to return semantic ranks for guesses.

Note: this project was generated with AI assistance and human supervision.

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
│   ├── categories/   # Manual semantic category dictionaries
│   └── embeddings/   # Production-ready embedding matrices
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
- `data/processed/armenian_contexto_metadata.json`: Armenian word metadata with English translation, POS, frequency, and category.
- `data/processed/failed_translations_top5000.json`: rejected translation records.
- `data/categories/english_categories.json`: manual English category dictionary.
- `models/fasttext/cc.hy.300.bin`: Armenian FastText model.
- `data/embeddings/armenian_contexto_embeddings.npz`: normalized Armenian embedding matrix.
- `data/embeddings/english_contexto_embeddings.npz`: normalized lightweight English translation vectors.
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
- `data/embeddings/english_contexto_embeddings.npz`

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

Build metadata, POS tags, categories, and English vectors:

```bash
python scripts/build_metadata.py
```

Run unit tests:

```bash
python -m unittest discover tests
```

Run the gameplay smoke test:

```bash
python tests/test_game.py
```

Play the terminal guessing game:

```bash
python scripts/play_terminal_game.py
```

Useful game commands:

- `:hint`: show the English translation hint.
- `:closest`: reveal the 20 closest words to the hidden target.
- `:skip`: reveal the current target and move on.
- `:quit`: exit the game.

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

The engine uses a hybrid semantic score:

```text
final_score =
    0.70 * Armenian FastText cosine similarity
  + 0.15 * same_category
  + 0.10 * same_POS
  + 0.05 * English translation similarity
```

Armenian embeddings are normalized, so the Armenian dot product is cosine
similarity. English translation vectors are also normalized and stored in
`data/embeddings/english_contexto_embeddings.npz`.

The extra category, POS, and English translation signals reduce noisy rankings
from pure FastText embeddings. For example, words in the same topic area or
part of speech get a small, explicit gameplay bonus without replacing the main
Armenian semantic signal.

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
