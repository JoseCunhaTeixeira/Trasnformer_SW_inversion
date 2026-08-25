"""Filesystem locations used across the silex pipeline.

Replaces folders.py's hardcoded absolute paths (placeholders like
"/.../Silex/..." that were never meant to resolve on any machine but the
original author's) with env-driven defaults relative to the repo root.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True, frozen=True)
class Paths:
    """Root directories for the silex pipeline.

    Each defaults to a folder next to this repo, but can be overridden
    independently (e.g. to point `models` at a shared volume) via the
    matching environment variable.
    """

    input: Path
    """Raw and generated training data. SILEX_INPUT_DIR, default ./input"""

    models: Path
    """Trained checkpoints. SILEX_MODELS_DIR, default ./models"""

    output: Path
    """Inversion/evaluation outputs. SILEX_OUTPUT_DIR, default ./output"""

    @classmethod
    def from_env(cls) -> Paths:
        return cls(
            input=Path(os.environ.get("SILEX_INPUT_DIR", _REPO_ROOT / "input")),
            models=Path(os.environ.get("SILEX_MODELS_DIR", _REPO_ROOT / "models")),
            output=Path(os.environ.get("SILEX_OUTPUT_DIR", _REPO_ROOT / "output")),
        )
