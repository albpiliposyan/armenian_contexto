import argparse
import json
import random
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from armenian_contexto import ArmenianContextoEngine  # noqa: E402
from armenian_contexto.paths import ARMENIAN_WORDS_TOP5000_FILE  # noqa: E402


DIFFICULTY_LEVELS = {
    "Easy": {
        "slice": (0.00, 0.25),
        "description": "top-frequency everyday words",
    },
    "Medium": {
        "slice": (0.25, 0.50),
        "description": "common vocabulary",
    },
    "Hard": {
        "slice": (0.50, 0.75),
        "description": "less frequent vocabulary",
    },
    "Expert": {
        "slice": (0.75, 1.00),
        "description": "lowest-frequency vocabulary",
    },
    "Any": {
        "slice": (0.00, 1.00),
        "description": "all available target words",
    },
}

CLOSEST_HEADERS = ["Rank", "Word", "Score"]


@lru_cache(maxsize=1)
def get_engine() -> ArmenianContextoEngine:
    return ArmenianContextoEngine()


@lru_cache(maxsize=1)
def load_vocabulary() -> list[dict]:
    with open(ARMENIAN_WORDS_TOP5000_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def english_hint(item: dict) -> str:
    translations = item.get("translations", [])
    if not translations:
        return "No English hint available."

    return ", ".join(translation["en"] for translation in translations[:3])


def rank_label(rank: int) -> str:
    if rank <= 10:
        return "burning hot"
    if rank <= 50:
        return "very close"
    if rank <= 200:
        return "close"
    if rank <= 1000:
        return "warm"
    return "far"


def candidate_pool(difficulty: str) -> list[dict]:
    vocabulary = load_vocabulary()
    level = DIFFICULTY_LEVELS[difficulty]
    vocabulary = [item for item in vocabulary if len(item["hy"]) >= 3]

    start_ratio, end_ratio = level["slice"]
    start = int(len(vocabulary) * start_ratio)
    end = int(len(vocabulary) * end_ratio)
    candidates = vocabulary[start:end]

    if not candidates:
        candidates = vocabulary

    return candidates


def choose_target(difficulty: str) -> dict:
    return random.choice(candidate_pool(difficulty))


def format_status(state: dict) -> str:
    if not state:
        return "Start a new game."

    difficulty = state["difficulty"]
    description = DIFFICULTY_LEVELS[difficulty]["description"]
    attempts = state["attempts"]
    pool_size = state["pool_size"]
    return (
        f"### {difficulty}\n"
        f"{description}\n\n"
        f"Target pool: **{pool_size}** words\n\n"
        f"Attempts: **{attempts}**"
    )


def dataframe_data(headers: list[str], rows: list[list]) -> dict:
    return {
        "headers": headers,
        "data": rows,
    }


def history_markdown(state: dict | None) -> str:
    if not state or not state["history"]:
        return "### Guesses\nNo guesses yet."

    lines = [
        "### Guesses",
        "| Attempt | Guess | Rank | Score | Temperature |",
        "|---:|---|---:|---:|---|",
    ]
    for item in sorted(state["history"], key=lambda row: row["rank"]):
        lines.append(
            "| "
            f"{item['attempt']} | "
            f"{item['guess']} | "
            f"#{item['rank']} | "
            f"{item['score']} | "
            f"{rank_label(item['rank'])} |"
        )

    return "\n".join(lines)


def start_game(difficulty: str):
    pool = candidate_pool(difficulty)
    target_item = random.choice(pool)
    state = {
        "difficulty": difficulty,
        "target": target_item["hy"],
        "target_item": target_item,
        "pool_size": len(pool),
        "attempts": 0,
        "history": [],
        "guessed_words": {},
        "solved": False,
    }

    return (
        state,
        format_status(state),
        "New hidden Armenian word selected.",
        history_markdown(state),
        dataframe_data(CLOSEST_HEADERS, []),
        gr.update(value=""),
    )


def validate_active_game(state: dict | None) -> tuple[bool, str]:
    if not state:
        return False, "Start a new game first."
    if state.get("solved"):
        return False, "This round is finished. Start a new game."
    return True, ""


def submit_guess(guess: str, state: dict | None):
    is_active, message = validate_active_game(state)
    if not is_active:
        return state, message, history_markdown(state), gr.update(value="")

    guess = guess.strip()
    if not guess:
        return state, "Enter an Armenian word.", history_markdown(state), gr.update(value="")

    engine = get_engine()
    processed_guess = engine.process_word(guess)
    target = state["target"]

    if processed_guess in state["guessed_words"]:
        previous = state["guessed_words"][processed_guess]
        message = (
            f"Already guessed **{previous['guess']}** on attempt "
            f"**{previous['attempt']}**. "
            f"Rank **#{previous['rank']}**, score **{previous['score']}**."
        )
        return state, message, history_markdown(state), gr.update(value="")

    target_processed = engine.process_word(target)
    if processed_guess != target_processed and processed_guess not in engine.word_to_index:
        message = "This word is not in the current game vocabulary."
        return state, message, history_markdown(state), gr.update(value="")

    state["attempts"] += 1
    result = engine.get_rank(guess, target)
    history_item = {
        "attempt": state["attempts"],
        "guess": result["guess"],
        "rank": result["rank"],
        "score": result["score"],
    }
    state["history"].append(history_item)
    state["guessed_words"][result["processed_guess"]] = history_item

    if result["is_correct"]:
        state["solved"] = True
        message = (
            f"Correct: **{target}**\n\n"
            f"Solved in **{state['attempts']}** guesses."
        )
        return state, message, history_markdown(state), gr.update(value="")

    message = (
        f"Rank **#{result['rank']}** | "
        f"score **{result['score']}** | "
        f"{rank_label(result['rank'])}"
    )
    return state, message, history_markdown(state), gr.update(value="")


def show_hint(state: dict | None) -> str:
    if not state:
        return "Start a new game first."
    return f"English hint: **{english_hint(state['target_item'])}**"


def reveal_answer(state: dict | None):
    if not state:
        return state, "Start a new game first."

    state["solved"] = True
    return (
        state,
        f"Revealed: **{state['target']}**\n\nEnglish hint: **{english_hint(state['target_item'])}**",
    )


def show_closest(state: dict | None) -> dict:
    if not state:
        return dataframe_data(CLOSEST_HEADERS, [])

    engine = get_engine()
    target_processed = engine.process_word(state["target"])
    closest = [
        item for item in engine.closest_words(state["target"], top_k=25)
        if engine.process_word(item["word"]) != target_processed
    ][:20]

    rows = [
        [item["rank"], item["word"], item["score"]]
        for item in closest
    ]
    return dataframe_data(CLOSEST_HEADERS, rows)


def build_app():
    with gr.Blocks(
        title="Armenian Contexto",
        analytics_enabled=False,
    ) as demo:
        state = gr.State()

        with gr.Column():
            gr.Markdown(
                "# Armenian Contexto\n"
                "Guess the hidden Armenian word by semantic rank."
            )

            with gr.Row():
                difficulty = gr.Dropdown(
                    choices=list(DIFFICULTY_LEVELS),
                    value="Easy",
                    label="Difficulty",
                    interactive=True,
                )
                new_game_button = gr.Button("New Game", variant="primary")

            status = gr.Markdown()
            result = gr.Markdown(
                "Start a new game.",
            )

            with gr.Row():
                guess = gr.Textbox(
                    label="Guess",
                    placeholder="օրինակ՝ դպրոց",
                    scale=4,
                )
                guess_button = gr.Button("Guess", variant="primary", scale=1)

            with gr.Row():
                hint_button = gr.Button("Hint")
                closest_button = gr.Button("Show 20 Closest")
                reveal_button = gr.Button("Reveal")

            history = gr.Markdown(history_markdown(None))
            closest = gr.Dataframe(
                value=dataframe_data(CLOSEST_HEADERS, []),
                headers=CLOSEST_HEADERS,
                label="Closest Words",
                interactive=False,
                wrap=True,
            )

        demo.load(
            start_game,
            inputs=difficulty,
            outputs=[state, status, result, history, closest, guess],
        )
        new_game_button.click(
            start_game,
            inputs=difficulty,
            outputs=[state, status, result, history, closest, guess],
        )
        difficulty.change(
            start_game,
            inputs=difficulty,
            outputs=[state, status, result, history, closest, guess],
        )
        guess_button.click(
            submit_guess,
            inputs=[guess, state],
            outputs=[state, result, history, guess],
        ).then(
            format_status,
            inputs=state,
            outputs=status,
        )
        guess.submit(
            submit_guess,
            inputs=[guess, state],
            outputs=[state, result, history, guess],
        ).then(
            format_status,
            inputs=state,
            outputs=status,
        )
        hint_button.click(show_hint, inputs=state, outputs=result)
        reveal_button.click(
            reveal_answer,
            inputs=state,
            outputs=[state, result],
        ).then(
            format_status,
            inputs=state,
            outputs=status,
        )
        closest_button.click(show_closest, inputs=state, outputs=closest)

    return demo


demo = build_app()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Armenian Contexto Gradio app.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary public gradio.live link.",
    )
    parser.add_argument(
        "--server-name",
        default=None,
        help="Host/interface to bind. Use 0.0.0.0 on remote servers.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=None,
        help="Port to bind.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )
