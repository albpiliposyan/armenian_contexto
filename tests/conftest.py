import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from armenian_contexto import ArmenianContextoEngine  # noqa: E402


@pytest.fixture(scope="session")
def engine():
    """Load the precomputed matrices once for the pytest session."""

    return ArmenianContextoEngine()
