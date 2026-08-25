"""Checkpoint save/load: a native `.keras` file + a trimmed JSON, replacing
transformer.py's `pickle.dump(self, f)` (which serialized the entire
stateful Transformer -- optimizer state, training-only code, hardcoded
paths baked into closures -- and needed sigpipe's experiments/silex/
export_legacy_model.py to migrate into something inference could load).

This writes exactly the schema sigpipe's SilexModel.load already reads
(min_freq, max_freq, d_freq, n_freqs, min_vel, max_vel, output_seq_length,
word_to_index, index_to_word, forbidden_tokens) directly, so a checkpoint
saved here needs no migration step to be usable by sigpipe's inference.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import keras


@dataclass(slots=True, frozen=True)
class CheckpointParams:
    min_freq: float
    max_freq: float
    d_freq: float
    n_freqs: int
    min_vel: float
    max_vel: float
    output_seq_length: int
    word_to_index: dict[str, int]
    index_to_word: dict[int, str]
    forbidden_tokens: list[list[int]]


def save_checkpoint(model: keras.Model, params: CheckpointParams, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save(out_dir / "silex.keras")

    # index_to_word's int keys stringify automatically via json.dump and are
    # cast back to int by sigpipe's SilexModel.load (json object keys are
    # always strings -- not lossy, just JSON's own restriction).
    (out_dir / "silex_params.json").write_text(json.dumps(asdict(params), indent=2))


def load_checkpoint(checkpoint_dir: Path) -> tuple[keras.Model, CheckpointParams]:
    model = cast("keras.Model", keras.saving.load_model(checkpoint_dir / "silex.keras"))

    raw = json.loads((checkpoint_dir / "silex_params.json").read_text())
    params = CheckpointParams(
        min_freq=raw["min_freq"],
        max_freq=raw["max_freq"],
        d_freq=raw["d_freq"],
        n_freqs=raw["n_freqs"],
        min_vel=raw["min_vel"],
        max_vel=raw["max_vel"],
        output_seq_length=raw["output_seq_length"],
        word_to_index=raw["word_to_index"],
        index_to_word={int(k): v for k, v in raw["index_to_word"].items()},
        forbidden_tokens=raw["forbidden_tokens"],
    )
    return model, params
