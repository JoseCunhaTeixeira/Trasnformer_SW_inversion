"""Checkpoint save/load: a native `.keras` file plus two trimmed JSON files
(silex_params.json, vocab.json), replacing transformer.py's
`pickle.dump(self, f)` (which serialized the entire stateful Transformer --
optimizer state, training-only code, hardcoded paths baked into closures --
and needed sigpipe's experiments/silex/export_legacy_model.py to migrate
into something inference could load).

silex_params.json holds normalization/training-band params plus the full
GenerationConfig used to build this checkpoint's training data (so a caller
can recover e.g. the `under_layers` substratum a forward-modeling comparison
needs, without it being silently implied). vocab.json holds the tokenizer
vocabulary and precomputed decode-grammar mask, split out since it's the
largest and least often needed standalone part of the artifact.

Together this is exactly the schema sigpipe's SilexModel.load reads
directly, so a checkpoint saved here needs no migration step to be usable
by sigpipe's inference.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import keras

from silex.generation import GenerationConfig, RockPhysicsConstants


@dataclass(slots=True, frozen=True)
class CheckpointParams:
    min_freq: float
    max_freq: float
    d_freq: float
    n_freqs: int
    min_vel: float
    max_vel: float
    output_seq_length: int
    generation_config: GenerationConfig
    """The full config used to generate this checkpoint's training data
    (soils/thicknesses/water-table/N sampling ranges, the `under_layers`
    substratum, dz, rock-physics constants, ...) -- see generation.py.
    `min_freq`/`max_freq`/`d_freq`/`n_freqs` above are the *training-band*
    slice of generation_config's own (typically wider) frequency range,
    matching whatever a training script filtered generated data down to."""


@dataclass(slots=True, frozen=True)
class VocabData:
    word_to_index: dict[str, int]
    index_to_word: dict[int, str]
    forbidden_tokens: list[list[int]]


def save_checkpoint(
    model: keras.Model,
    params: CheckpointParams,
    vocab: VocabData,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save(out_dir / "silex.keras")

    # index_to_word's int keys stringify automatically via json.dump and are
    # cast back to int by load_checkpoint/sigpipe's SilexModel.load (json
    # object keys are always strings -- not lossy, just JSON's restriction).
    (out_dir / "silex_params.json").write_text(json.dumps(asdict(params), indent=2))
    (out_dir / "vocab.json").write_text(json.dumps(asdict(vocab), indent=2))


def load_checkpoint(checkpoint_dir: Path) -> tuple[keras.Model, CheckpointParams, VocabData]:
    model = cast("keras.Model", keras.saving.load_model(checkpoint_dir / "silex.keras"))

    raw_params = json.loads((checkpoint_dir / "silex_params.json").read_text())
    raw_generation_config = dict(raw_params["generation_config"])
    generation_config = GenerationConfig(
        **{k: v for k, v in raw_generation_config.items() if k != "rock_physics"},
        rock_physics=RockPhysicsConstants(**raw_generation_config["rock_physics"]),
    )
    params = CheckpointParams(
        min_freq=raw_params["min_freq"],
        max_freq=raw_params["max_freq"],
        d_freq=raw_params["d_freq"],
        n_freqs=raw_params["n_freqs"],
        min_vel=raw_params["min_vel"],
        max_vel=raw_params["max_vel"],
        output_seq_length=raw_params["output_seq_length"],
        generation_config=generation_config,
    )

    raw_vocab = json.loads((checkpoint_dir / "vocab.json").read_text())
    vocab = VocabData(
        word_to_index=raw_vocab["word_to_index"],
        index_to_word={int(k): v for k, v in raw_vocab["index_to_word"].items()},
        forbidden_tokens=raw_vocab["forbidden_tokens"],
    )

    return model, params, vocab
