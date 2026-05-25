# Armenian Contexto

An Armenian implementation of a semantic word guessing game inspired by
Contexto.

The game does not compare letters. It combines Armenian embedding similarity,
English translation similarity, and POS tags to return semantic ranks for
guesses.

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
- `data/processed/armenian_contexto_metadata.json`: Armenian word metadata with English translation, POS, and frequency.
- `data/processed/failed_translations_top5000.json`: rejected translation records.
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
python3 --version
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Use a Python executable that reports version `3.11` or newer.

Then run the project workflow helper:

```bash
python scripts/run_project.py
```

The helper checks required derived files, runs tests, and asks before downloading
external models. If `models/fasttext/cc.hy.300.bin` is missing, it asks before
downloading the Armenian FastText model with:

```python
import fasttext.util
fasttext.util.download_model("hy", if_exists="ignore")
```

That download is large: `cc.hy.300.bin.gz` is about 4.2 GB compressed and
extracts to `cc.hy.300.bin`, about 6.8 GB. Keep roughly 11 GB free while both
files are present.

Useful workflow flags:

- `--yes`: answer yes to download prompts.
- `--no-download`: never download missing external artifacts.
- `--download-model-only`: only ensure `models/fasttext/cc.hy.300.bin` exists.
- `--rebuild`: rebuild derived embeddings and metadata.
- `--api`: launch the FastAPI server after setup and tests.
- `--play`: launch the terminal game after setup and tests.

## Workflow

Run commands from the project root.

Recommended one-command workflow:

```bash
python scripts/run_project.py
```

Run the Gradio game GUI locally:

```bash
python app.py
```

Create a temporary public Gradio link:

```bash
python app.py --share
```

The Gradio app uses the small precomputed `.npz` and metadata files. It does
not require `models/fasttext/cc.hy.300.bin` for normal in-vocabulary gameplay.

Regenerate all derived pipeline outputs from the raw dataset and write a log:

```bash
python scripts/regenerate_pipeline.py
```

The regeneration script removes generated artifacts, rebuilds the full pipeline,
and writes a timestamped log under `logs/`. If the Armenian FastText model is
missing, it asks before downloading the large model. Use `--no-download` to fail
instead of downloading, or `--yes` to accept prerequisite download prompts.

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

Build metadata, POS tags, and English vectors:

```bash
python scripts/build_metadata.py
```

If `en_core_web_sm` is missing, either let `scripts/run_project.py` install it
or run:

```bash
python -m spacy download en_core_web_sm
```

Run unit tests:

```bash
python -m pytest tests
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

Run the API:

```bash
python scripts/run_project.py --api
```

Or directly:

```bash
uvicorn armenian_contexto.api:app --host 127.0.0.1 --port 8000
```

Example API calls:

```bash
curl "http://127.0.0.1:8000/api/guess?target=դպրոց&guess=ուսուցիչ"
curl "http://127.0.0.1:8000/api/closest?target=դպրոց&top_k=20"
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

The engine uses a hybrid semantic score:

```text
final_score =
    0.85 * Armenian FastText cosine similarity
  + 0.10 * same_POS
  + 0.05 * English translation similarity
```

Armenian embeddings are normalized, so the Armenian dot product is cosine
similarity. English translation vectors are also normalized and stored in
`data/embeddings/english_contexto_embeddings.npz`.

The POS and English translation signals are small stabilizers on top of the
main Armenian semantic signal.

The rank is:

```text
number of vocabulary words more similar to the target + 1
```

## Architecture

The API is FastAPI-based and loads one `ArmenianContextoEngine` during startup.
That engine keeps the precomputed NumPy matrices in memory for the life of the
process:

- `data/embeddings/armenian_contexto_embeddings.npz`: about 4.4 MB, currently `(4112, 300)`.
- `data/embeddings/english_contexto_embeddings.npz`: about 1.4 MB, currently `(4112, 96)`.

The normal gameplay path does not load the 6.8 GB FastText model. That model is
only needed for rebuilding Armenian embeddings or scoring out-of-vocabulary
guesses.

Concurrency choice:

- The API endpoints are `async`, so HTTP handling is I/O-friendly.
- Engine calls run in a `ThreadPoolExecutor`, keeping the FastAPI event loop
  responsive.
- This is a CPU-light, NumPy-heavy workload. NumPy does the expensive vector math
  in optimized native code and can release the GIL during array operations.
- Multiprocessing is not a good default here because each process would need its
  own copy of the loaded matrices, increasing memory use for a small per-request
  computation.

Memory management:

- Embeddings are precomputed and stored as `float32`.
- Matrices are loaded once at API startup and reused.
- Ranking is vectorized over the vocabulary instead of looping through Python
  objects one word at a time.
- The large FastText binary is kept outside Git and outside the normal request
  path.

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
