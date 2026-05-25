import argparse
import json
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from armenian_contexto import ArmenianContextoEngine  # noqa: E402
from armenian_contexto.paths import ARMENIAN_WORDS_TOP5000_FILE  # noqa: E402


DIFFICULTY_LEVELS = [
    {
        "name": "Easy",
        "range": (0, 400),
        "description": "very frequent everyday words",
        "preferred_targets": [
            "տուն",
            "դպրոց",
            "գիրք",
            "մարդ",
            "ջուր",
            "քաղաք",
            "ընտանիք",
            "երեխա",
        ],
    },
    {
        "name": "Medium",
        "range": (400, 1200),
        "description": "common words",
        "preferred_targets": [
            "մեքենա",
            "աշխատանք",
            "համալսարան",
            "երաժշտություն",
            "պատմություն",
            "հիվանդանոց",
        ],
    },
    {
        "name": "Hard",
        "range": (1200, 2500),
        "description": "less frequent words",
        "preferred_targets": [
            "կառավարություն",
            "հաստատություն",
            "կրթություն",
            "արվեստ",
            "հասարակություն",
        ],
    },
    {
        "name": "Expert",
        "range": (2500, None),
        "description": "lower-frequency vocabulary",
        "preferred_targets": [],
    },
]


def load_vocabulary():
    with open(ARMENIAN_WORDS_TOP5000_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def english_hint(item):
    translations = item.get("translations", [])
    if not translations:
        return "no English hint available"

    return ", ".join(translation["en"] for translation in translations[:3])


def choose_target(vocabulary, level, used_words, rng):
    by_word = {item["hy"]: item for item in vocabulary}
    preferred_candidates = [
        by_word[word] for word in level["preferred_targets"]
        if word in by_word and word not in used_words
    ]

    if preferred_candidates:
        return rng.choice(preferred_candidates)

    start, end = level["range"]
    candidates = vocabulary[start:end]
    candidates = [
        item for item in candidates
        if item["hy"] not in used_words and len(item["hy"]) >= 3
    ]

    if not candidates:
        return None

    return rng.choice(candidates)


def rank_label(rank):
    if rank <= 10:
        return "burning hot"
    if rank <= 50:
        return "very close"
    if rank <= 200:
        return "close"
    if rank <= 1000:
        return "warm"
    return "far"


def print_round_header(round_number, level):
    print()
    print(f"Round {round_number}: {level['name']} ({level['description']})")
    print("Guess the hidden Armenian word.")
    print("Commands: :hint, :closest, :reveal, :skip, :quit")
    print()


def print_guess_history(history):
    if not history:
        return

    print("\nBest guesses:")
    for item in sorted(history.values(), key=lambda row: row["rank"])[:5]:
        print(
            f"  {item['guess']:<18} "
            f"rank #{item['rank']:<5} "
            f"similarity {item['similarity']:<8} "
            f"{rank_label(item['rank'])}"
        )
    print()


def print_closest_words(engine, target, top_k=20):
    target_processed = engine.process_word(target)
    closest_words = [
        item for item in engine.closest_words(target, top_k=top_k + 1)
        if engine.process_word(item["word"]) != target_processed
    ][:top_k]

    print(f"\nTop {top_k} closest words:")
    for item in closest_words:
        print(f"  #{item['rank']:<2} {item['word']:<18} score {item['score']}")
    print()


def play_round(engine, target_item, round_number, level):
    target = target_item["hy"]
    attempts = 0
    history = {}

    print_round_header(round_number, level)

    while True:
        guess = input("Your guess: ").strip()

        if not guess:
            continue

        if guess == ":quit":
            return "quit"

        if guess == ":skip":
            print(f"Skipped. The word was: {target}")
            print(f"English hint was: {english_hint(target_item)}")
            return "skipped"

        if guess == ":reveal":
            print(f"Revealed. The word was: {target}")
            print(f"English hint was: {english_hint(target_item)}")
            return "revealed"

        if guess == ":hint":
            print(f"English hint: {english_hint(target_item)}")
            continue

        if guess == ":closest":
            print_closest_words(engine, target, top_k=20)
            continue

        processed_guess = engine.process_word(guess)
        if processed_guess in history:
            previous = history[processed_guess]
            print(
                f"Already guessed '{previous['guess']}'. "
                f"Rank #{previous['rank']} | "
                f"similarity {previous['similarity']} | "
                f"{rank_label(previous['rank'])}"
            )
            print_guess_history(history)
            continue

        attempts += 1
        result = engine.get_rank(guess, target)

        if result["is_correct"]:
            print(f"Correct: {target}")
            print(f"Solved in {attempts} guesses.")
            return "won"

        history[result["processed_guess"]] = result
        print(
            f"Rank #{result['rank']} | "
            f"similarity {result['similarity']} | "
            f"{rank_label(result['rank'])}"
        )
        print_guess_history(history)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play Armenian Contexto in the terminal.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set a random seed for repeatable target selection.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=len(DIFFICULTY_LEVELS),
        help="Number of rounds to play.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    vocabulary = load_vocabulary()
    engine = ArmenianContextoEngine()

    used_words = set()
    rounds_to_play = max(1, args.rounds)

    print("Armenian Contexto Terminal Game")
    print("A hidden Armenian word is selected. Guess words and follow the semantic rank.")
    print("Rank #1 means correct after Armenian normalization.")

    for round_index in range(rounds_to_play):
        level = DIFFICULTY_LEVELS[min(round_index, len(DIFFICULTY_LEVELS) - 1)]
        target_item = choose_target(vocabulary, level, used_words, rng)

        if target_item is None:
            print(f"No available target words for level: {level['name']}")
            break

        used_words.add(target_item["hy"])
        result = play_round(engine, target_item, round_index + 1, level)

        if result == "quit":
            print("Goodbye.")
            break

        if round_index < rounds_to_play - 1:
            print("Next round is harder.")


if __name__ == "__main__":
    main()
