__all__ = [
    "ArmenianContextoEngine",
    "normalize_text",
    "stem_armenian",
]


def __getattr__(name: str):
    if name in __all__:
        from .contexto_engine import (
            ArmenianContextoEngine,
            normalize_text,
            stem_armenian,
        )

        exports = {
            "ArmenianContextoEngine": ArmenianContextoEngine,
            "normalize_text": normalize_text,
            "stem_armenian": stem_armenian,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
