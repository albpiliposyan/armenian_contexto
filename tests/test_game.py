import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from armenian_contexto import ArmenianContextoEngine  # noqa: E402


engine = ArmenianContextoEngine()

target = "դպրոց"

guesses = [
    "ուսուցիչ",
    "աշակերտ",
    "գիրք",
    "մեքենա",
    "դպրոցը",
]

for guess in guesses:
    result = engine.get_rank(guess, target)
    print(result)

print("\nClosest words:")
for item in engine.closest_words(target, top_k=10):
    print(item)
