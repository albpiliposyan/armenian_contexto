import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from armenian_contexto.paths import WORD_FREQ_FILE  # noqa: E402


with open(WORD_FREQ_FILE, "r", encoding="utf-8") as f:
    words = json.load(f)

top_words = [w[0] for w in words[:100]]

print(top_words)
