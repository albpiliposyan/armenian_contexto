import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from armenian_contexto.paths import (  # noqa: E402
    ARMENIAN_WORDS_FILE,
    ARMENIAN_WORDS_TOP5000_FILE,
    CLEANED_WORDS_FILE,
    CONTEXTO_WORDS_FILE,
    EMBEDDINGS_FILE,
    ENGLISH_EMBEDDINGS_FILE,
    FAILED_TRANSLATIONS_FILE,
    FAILED_TRANSLATIONS_TOP5000_FILE,
    FASTTEXT_MODEL_FILE,
    METADATA_FILE,
    WORD_FREQ_FILE,
)


FASTTEXT_DOWNLOAD_NOTE = (
    "This downloads cc.hy.300.bin.gz, about 4.2 GB compressed, "
    "and extracts cc.hy.300.bin, about 6.8 GB. Keep roughly 11 GB free "
    "while both files are present."
)
SPACY_DOWNLOAD_NOTE = (
    "This downloads the spaCy English model en_core_web_sm, about 12-15 MB."
)

GENERATED_FILES_TO_REMOVE = [
    CLEANED_WORDS_FILE,
    CONTEXTO_WORDS_FILE,
    ARMENIAN_WORDS_FILE,
    ARMENIAN_WORDS_TOP5000_FILE,
    FAILED_TRANSLATIONS_FILE,
    FAILED_TRANSLATIONS_TOP5000_FILE,
    METADATA_FILE,
    EMBEDDINGS_FILE,
    ENGLISH_EMBEDDINGS_FILE,
]

EXPECTED_PIPELINE_OUTPUTS = [
    CLEANED_WORDS_FILE,
    CONTEXTO_WORDS_FILE,
    ARMENIAN_WORDS_TOP5000_FILE,
    FAILED_TRANSLATIONS_TOP5000_FILE,
    METADATA_FILE,
    EMBEDDINGS_FILE,
    ENGLISH_EMBEDDINGS_FILE,
]

GENERATED_DIRECTORIES = [
    PROJECT_ROOT / ".pytest_cache",
    PROJECT_ROOT / "src" / "armenian_contexto.egg-info",
    PROJECT_ROOT / "src" / "armenian_contexto" / "__pycache__",
    PROJECT_ROOT / "scripts" / "__pycache__",
    PROJECT_ROOT / "tests" / "__pycache__",
]

PIPELINE_STEPS = [
    ("Clean English frequency words", ["scripts/clear_words.py"]),
    ("POS-filter Contexto words", ["scripts/filter_pos_tags.py"]),
    ("Translate English words to Armenian", ["scripts/translate_to_armenian.py"]),
    ("Build Armenian embeddings", ["scripts/build_embeddings.py"]),
    ("Build metadata and English vectors", ["scripts/build_metadata.py"]),
]


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_line(log_file, message: str = ""):
    print(message)
    log_file.write(message + "\n")
    log_file.flush()


def confirm(log_file, message: str, assume_yes: bool) -> bool:
    if assume_yes:
        write_line(log_file, f"{message} [auto yes]")
        return True

    log_file.write(message + " [prompted]\n")
    log_file.flush()

    try:
        answer = input(f"{message} [y/N]: ").strip().lower()
    except EOFError:
        write_line(log_file, "No interactive input available; assuming no.")
        return False

    accepted = answer in {"y", "yes"}
    write_line(
        log_file,
        "User accepted download." if accepted else "User declined download.",
    )
    return accepted


def ensure_spacy_model(log_file, assume_yes: bool, no_download: bool):
    try:
        import spacy

        spacy.load("en_core_web_sm")
        write_line(log_file, "  found spaCy model: en_core_web_sm")
        return
    except OSError:
        if no_download:
            raise RuntimeError(
                "Missing spaCy model `en_core_web_sm` and downloads are disabled. "
                "Run `python -m spacy download en_core_web_sm` first."
            )

        if not confirm(log_file, f"{SPACY_DOWNLOAD_NOTE} Download it now?", assume_yes):
            raise RuntimeError(
                "Missing spaCy model `en_core_web_sm`. "
                "Run `python -m spacy download en_core_web_sm` first."
            )

        run_command(
            log_file,
            "Download spaCy English model",
            ["-m", "spacy", "download", "en_core_web_sm"],
        )


def ensure_fasttext_model(log_file, assume_yes: bool, no_download: bool):
    if FASTTEXT_MODEL_FILE.exists():
        write_line(log_file, f"  found FastText model: {relative(FASTTEXT_MODEL_FILE)}")
        return

    if no_download:
        raise FileNotFoundError(
            "Missing Armenian FastText model and downloads are disabled: "
            f"{relative(FASTTEXT_MODEL_FILE)}."
        )

    if not confirm(log_file, f"{FASTTEXT_DOWNLOAD_NOTE} Download it now?", assume_yes):
        raise FileNotFoundError(
            "Missing Armenian FastText model: "
            f"{relative(FASTTEXT_MODEL_FILE)}. "
            "Run `python scripts/run_project.py --download-model-only` first."
        )

    run_command(
        log_file,
        "Download Armenian FastText model",
        ["scripts/run_project.py", "--download-model-only", "--yes"],
    )


def preflight(log_file, assume_yes: bool, no_download: bool):
    write_line(log_file, "Preflight checks:")

    if not WORD_FREQ_FILE.exists():
        raise FileNotFoundError(f"Missing raw input dataset: {relative(WORD_FREQ_FILE)}")
    write_line(log_file, f"  found raw dataset: {relative(WORD_FREQ_FILE)}")

    ensure_spacy_model(log_file, assume_yes, no_download)
    ensure_fasttext_model(log_file, assume_yes, no_download)


def clean_generated_outputs(log_file):
    write_line(log_file)
    write_line(log_file, "Removing generated files:")

    for path in GENERATED_FILES_TO_REMOVE:
        if path.exists():
            size = path.stat().st_size
            path.unlink()
            write_line(log_file, f"  removed {relative(path)} ({size} bytes)")
        else:
            write_line(log_file, f"  already missing {relative(path)}")

    write_line(log_file)
    write_line(log_file, "Removing generated cache directories:")

    for path in GENERATED_DIRECTORIES:
        if path.exists():
            remove_directory(path)
            write_line(log_file, f"  removed {relative(path)}")
        else:
            write_line(log_file, f"  already missing {relative(path)}")


def remove_directory(path: Path):
    for child in path.iterdir():
        if child.is_dir():
            remove_directory(child)
        else:
            child.unlink()
    path.rmdir()


def run_command(log_file, description: str, script_args: list[str]):
    command = [sys.executable, *script_args]

    write_line(log_file)
    write_line(log_file, f"==> {description}")
    write_line(log_file, "Command: " + " ".join(command))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        log_file.write(line)
        log_file.flush()

    return_code = process.wait()
    write_line(log_file, f"Exit code: {return_code}")

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def generated_file_report(log_file):
    write_line(log_file)
    write_line(log_file, "Generated file report:")
    for path in EXPECTED_PIPELINE_OUTPUTS:
        if path.exists():
            write_line(log_file, f"  {relative(path)} - {path.stat().st_size} bytes")
        else:
            write_line(log_file, f"  missing: {relative(path)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate all Armenian Contexto pipeline outputs with logging.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log path. Defaults to logs/pipeline_YYYYmmdd_HHMMSS.log.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Regenerate artifacts but skip pytest verification.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Answer yes to prerequisite download prompts.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download missing external artifacts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log_path = args.log_file
    if log_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = PROJECT_ROOT / "logs" / f"pipeline_{timestamp}.log"
    elif not log_path.is_absolute():
        log_path = PROJECT_ROOT / log_path

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log_file:
        write_line(log_file, "Armenian Contexto pipeline regeneration")
        write_line(log_file, f"Started: {datetime.now().isoformat(timespec='seconds')}")
        write_line(log_file, f"Project root: {PROJECT_ROOT}")
        write_line(log_file, f"Python: {sys.version}")
        write_line(log_file, f"Log file: {relative(log_path)}")
        write_line(log_file)
        write_line(
            log_file,
            "Note: translation uses Google-based translation APIs and may be slow or unstable.",
        )

        preflight(log_file, args.yes, args.no_download)
        clean_generated_outputs(log_file)

        for description, command in PIPELINE_STEPS:
            run_command(log_file, description, command)

        if not args.skip_tests:
            run_command(log_file, "Run pytest", ["-m", "pytest", "tests"])

        generated_file_report(log_file)
        write_line(log_file)
        write_line(log_file, f"Finished: {datetime.now().isoformat(timespec='seconds')}")

    print(f"\nPipeline log written to: {relative(log_path)}")


if __name__ == "__main__":
    main()
