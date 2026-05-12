import json
import re
from pathlib import Path


def normalize_english_word(word: str) -> str:
    word = word.strip().lower()
    return re.sub(r"[^a-z]+", "", word)


def load_json(path: str | Path, default):
    path = Path(path)
    if not path.exists():
        return default

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_category_map(path: str | Path) -> dict[str, list[str]]:
    return load_json(path, {})


def build_category_lookup(category_map: dict[str, list[str]]) -> dict[str, str]:
    lookup = {}
    for category, words in category_map.items():
        for word in words:
            normalized = normalize_english_word(word)
            if normalized:
                lookup[normalized] = category
    return lookup


def category_for_english(word: str, category_lookup: dict[str, str]) -> str | None:
    return category_lookup.get(normalize_english_word(word))


def load_metadata(path: str | Path) -> dict[str, dict]:
    rows = load_json(path, [])
    return {row["hy"]: row for row in rows}
