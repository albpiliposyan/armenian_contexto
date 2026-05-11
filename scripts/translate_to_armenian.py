import json
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from deep_translator import GoogleTranslator
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from armenian_contexto.paths import (  # noqa: E402
    ARMENIAN_WORDS_TOP5000_FILE,
    CONTEXTO_WORDS_FILE,
    FAILED_TRANSLATIONS_TOP5000_FILE,
)


INPUT_FILE = CONTEXTO_WORDS_FILE
OUTPUT_FILE = ARMENIAN_WORDS_TOP5000_FILE
FAILED_FILE = FAILED_TRANSLATIONS_TOP5000_FILE

TOP_N = 5000
BATCH_SIZE = 50
MAX_WORKERS = 3
MAX_RETRIES = 3
SAVE_EVERY_BATCHES = 5


def is_single_armenian_word(text: str) -> bool:
    text = text.strip().lower()
    return re.fullmatch(r"[ա-ֆև]+", text) is not None


def load_json(path: Path, default):
    if not path.exists():
        return default

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def translate_batch(batch):
    words = [item["word"] for item in batch]

    for attempt in range(MAX_RETRIES):
        try:
            translator = GoogleTranslator(source="en", target="hy")
            translated_words = translator.translate_batch(words)

            return {
                "status": "success",
                "batch": batch,
                "translated_words": translated_words,
            }

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return {
                    "status": "failed",
                    "batch": batch,
                    "error": str(e),
                }

            time.sleep(2 * (attempt + 1))


def main():
    data = load_json(INPUT_FILE, [])

    data.sort(key=lambda x: x["frequency"], reverse=True)
    data = data[:TOP_N]

    armenian_map = {}
    failed = []

    batches = list(chunks(data, BATCH_SIZE))

    completed_batches = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(translate_batch, batch)
            for batch in batches
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Translating"):
            result = future.result()
            completed_batches += 1

            if result["status"] == "failed":
                for item in result["batch"]:
                    failed.append({
                        "en": item["word"],
                        "frequency": item["frequency"],
                        "reason": "batch_translation_error",
                        "error": result["error"],
                    })

            else:
                batch = result["batch"]
                translated_words = result["translated_words"]

                for item, hy_word in zip(batch, translated_words):
                    en_word = item["word"]
                    frequency = item["frequency"]

                    hy_word = hy_word.strip().lower()

                    if not is_single_armenian_word(hy_word):
                        failed.append({
                            "en": en_word,
                            "frequency": frequency,
                            "hy_raw": hy_word,
                            "reason": "not_single_armenian_word",
                        })
                        continue

                    if hy_word not in armenian_map:
                        armenian_map[hy_word] = {
                            "hy": hy_word,
                            "translations": [],
                        }

                    if len(armenian_map[hy_word]["translations"]) < 3:
                        armenian_map[hy_word]["translations"].append({
                            "en": en_word,
                            "frequency": frequency,
                        })
                    else:
                        failed.append({
                            "en": en_word,
                            "frequency": frequency,
                            "hy_raw": hy_word,
                            "reason": "duplicate_armenian_translation_more_than_3",
                        })

            if completed_batches % SAVE_EVERY_BATCHES == 0:
                output_data = list(armenian_map.values())
                output_data.sort(
                    key=lambda x: x["translations"][0]["frequency"],
                    reverse=True
                )

                save_json(OUTPUT_FILE, output_data)
                save_json(FAILED_FILE, failed)

    output_data = list(armenian_map.values())
    output_data.sort(
        key=lambda x: x["translations"][0]["frequency"],
        reverse=True
    )

    save_json(OUTPUT_FILE, output_data)
    save_json(FAILED_FILE, failed)

    print(f"Saved {len(output_data)} unique Armenian words to {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    print(f"Saved {len(failed)} failed/skipped words to {FAILED_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
