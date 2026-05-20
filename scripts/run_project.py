import argparse
import contextlib
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from armenian_contexto.paths import (  # noqa: E402
    EMBEDDINGS_FILE,
    ENGLISH_EMBEDDINGS_FILE,
    FASTTEXT_DIR,
    FASTTEXT_MODEL_FILE,
    FASTTEXT_MODEL_GZ_FILE,
    METADATA_FILE,
)


FASTTEXT_DOWNLOAD_NOTE = (
    "This downloads cc.hy.300.bin.gz, about 4.2 GB compressed, "
    "and extracts cc.hy.300.bin, about 6.8 GB. Keep roughly 11 GB free "
    "while both files are present."
)
SPACY_DOWNLOAD_NOTE = "This downloads the spaCy English model en_core_web_sm, about 12-15 MB."


@contextlib.contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def confirm(message: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True

    try:
        answer = input(f"{message} [y/N]: ").strip().lower()
    except EOFError:
        return False

    return answer in {"y", "yes"}


def run_step(command: list[str], description: str):
    print(f"\n==> {description}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def ensure_spacy_model(assume_yes: bool, no_download: bool) -> bool:
    try:
        import spacy

        spacy.load("en_core_web_sm")
        print("spaCy model found: en_core_web_sm")
        return True
    except OSError:
        if no_download:
            print("spaCy model missing and downloads are disabled.")
            return False

        if not confirm(f"{SPACY_DOWNLOAD_NOTE} Download it now?", assume_yes):
            print("Skipped spaCy model download.")
            return False

        run_step(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            "Downloading spaCy English model",
        )
        return True


def ensure_fasttext_model(assume_yes: bool, no_download: bool) -> bool:
    if FASTTEXT_MODEL_FILE.exists():
        print(f"FastText model found: {relative(FASTTEXT_MODEL_FILE)}")
        return True

    if no_download:
        print(f"FastText model missing: {relative(FASTTEXT_MODEL_FILE)}")
        print("Downloads are disabled, so OOV guesses and embedding rebuilds may fail.")
        return False

    if not confirm(f"{FASTTEXT_DOWNLOAD_NOTE} Download it now?", assume_yes):
        print("Skipped Armenian FastText model download.")
        print("The game can still use prebuilt embeddings, but OOV guesses may fail.")
        return False

    FASTTEXT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n==> Downloading Armenian FastText model into {relative(FASTTEXT_DIR)}")
    import fasttext.util

    with working_directory(FASTTEXT_DIR):
        fasttext.util.download_model("hy", if_exists="ignore")

    if not FASTTEXT_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"fasttext.util.download_model finished, but {FASTTEXT_MODEL_FILE} was not created."
        )

    print(f"Downloaded {relative(FASTTEXT_MODEL_FILE)}")
    if FASTTEXT_MODEL_GZ_FILE.exists():
        print(f"Kept compressed archive: {relative(FASTTEXT_MODEL_GZ_FILE)}")

    return True


def build_missing_artifacts(args) -> bool:
    model_available = ensure_fasttext_model(args.yes, args.no_download)

    if args.download_model_only:
        return model_available

    needs_arm_embeddings = args.rebuild or not EMBEDDINGS_FILE.exists()
    if needs_arm_embeddings:
        if not model_available:
            print(f"Cannot build {relative(EMBEDDINGS_FILE)} without the FastText model.")
            return False

        run_step(
            [sys.executable, "scripts/build_embeddings.py"],
            "Building Armenian embedding matrix",
        )
    else:
        print(f"Armenian embeddings found: {relative(EMBEDDINGS_FILE)}")

    needs_metadata = (
        args.rebuild
        or not METADATA_FILE.exists()
        or not ENGLISH_EMBEDDINGS_FILE.exists()
    )
    if needs_metadata:
        if not ensure_spacy_model(args.yes, args.no_download):
            print("Cannot build metadata without en_core_web_sm.")
            return False

        run_step(
            [sys.executable, "scripts/build_metadata.py"],
            "Building metadata and English vectors",
        )
    else:
        print(f"Metadata found: {relative(METADATA_FILE)}")
        print(f"English vectors found: {relative(ENGLISH_EMBEDDINGS_FILE)}")

    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare and run the Armenian Contexto project workflow.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Answer yes to download prompts.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download missing external artifacts.",
    )
    parser.add_argument(
        "--download-model-only",
        action="store_true",
        help="Only ensure models/fasttext/cc.hy.300.bin exists.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild derived embeddings and metadata even if they already exist.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip the unittest suite.",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Launch the terminal game after setup and tests.",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Launch the FastAPI server after setup and tests.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for --api. Defaults to 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for --api. Defaults to 8000.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Armenian Contexto workflow")
    print(f"Project root: {PROJECT_ROOT}")

    if not build_missing_artifacts(args):
        return 1

    if args.download_model_only:
        return 0

    if not args.skip_tests:
        run_step(
            [sys.executable, "-m", "pytest", "tests"],
            "Running tests",
        )

    if args.api:
        run_step(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "armenian_contexto.api:app",
                "--host",
                args.host,
                "--port",
                str(args.port),
            ],
            "Starting FastAPI server",
        )
        return 0

    if args.play:
        run_step(
            [sys.executable, "scripts/play_terminal_game.py"],
            "Starting terminal game",
        )
    else:
        print("\nSetup complete. Start the game with:")
        print("python scripts/play_terminal_game.py")
        print("Or start the API with:")
        print("python scripts/run_project.py --api")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
