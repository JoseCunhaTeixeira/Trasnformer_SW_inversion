"""Output-sequence vocabulary and decode-time grammar mask.

A faithful, typed port of misc.py's make_vocab / make_index_representation /
make_allowed_tokens / make_forbidden_tokens. Verified (see tests) to
reproduce models/[202407170928]/[202407170928]_params.json's real
output_seq_format exactly, from the same data_params.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class VocabParams:
    """The subset of data-generation params that determine the output
    vocabulary and grammar -- soil names and the quantized WT/N/thickness
    values the model was trained to choose among."""

    soils: tuple[str, ...]
    max_n_layers: int
    water_table_depths: tuple[float, ...]
    n_values: tuple[int, ...]
    thicknesses: tuple[float, ...]


PAD = "[PAD]"
START = "[START]"
END = "[END]"
WT = "[WT]"


def build_vocab(params: VocabParams) -> list[str]:
    """[PAD, START, END, WT, SOIL1..N, THICKNESS1..N, N1..N, then every
    distinct WT/N/thickness value formatted to 2 decimals, then soil names]
    -- the exact layout the model's embedding indices were trained against."""
    vocab = [PAD, START, END, WT]

    for layer in range(params.max_n_layers):
        vocab.append(f"[SOIL{layer + 1}]")
    for layer in range(params.max_n_layers):
        vocab.append(f"[THICKNESS{layer + 1}]")
    for layer in range(params.max_n_layers):
        vocab.append(f"[N{layer + 1}]")

    numbers = sorted({*params.water_table_depths, *params.n_values, *params.thicknesses})
    for number in numbers:
        vocab.append(f"{number:.2f}")

    vocab.extend(params.soils)

    return vocab


def build_index(vocab: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    word_to_index = {word: i for i, word in enumerate(vocab)}
    index_to_word = {index: word for word, index in word_to_index.items()}
    return word_to_index, index_to_word


def build_allowed_tokens(params: VocabParams, word_to_index: dict[str, int]) -> list[list[int]]:
    """Per-decode-step id whitelist enforcing the grammar
    [WT] <v> ([SOILi] <soil> [THICKNESSi] <v> [Ni] <v>)* [END], one entry
    per step (WT keyword, WT value, then 6 entries per possible layer,
    then END)."""
    allowed_tokens = [
        [word_to_index[WT]],
        [word_to_index[f"{v:.2f}"] for v in params.water_table_depths],
    ]

    for layer in range(params.max_n_layers):
        if layer == 0:
            allowed_tokens.append([word_to_index[f"[SOIL{layer + 1}]"]])
        else:
            allowed_tokens.append([word_to_index[f"[SOIL{layer + 1}]"], word_to_index[END]])
        allowed_tokens.append([word_to_index[soil] for soil in params.soils])

        allowed_tokens.append([word_to_index[f"[THICKNESS{layer + 1}]"]])
        allowed_tokens.append([word_to_index[f"{v:.2f}"] for v in params.thicknesses])

        allowed_tokens.append([word_to_index[f"[N{layer + 1}]"]])
        allowed_tokens.append([word_to_index[f"{v:.2f}"] for v in params.n_values])

    allowed_tokens.append([word_to_index[END]])

    return allowed_tokens


def build_forbidden_tokens(
    params: VocabParams,
    word_to_index: dict[str, int],
    output_seq_length: int,
) -> list[list[int]]:
    """The complement of build_allowed_tokens per step, for output_seq_length
    - 1 decode steps -- what SilexModel._decode (sigpipe) and this repo's own
    decoding.py both mask logits with."""
    allowed_tokens = build_allowed_tokens(params, word_to_index)
    all_tokens = list(word_to_index.values())

    forbidden_tokens: list[list[int]] = []
    for step in range(output_seq_length - 1):
        step_forbidden = all_tokens.copy()
        for allowed in allowed_tokens[step]:
            step_forbidden.remove(allowed)
        forbidden_tokens.append(step_forbidden)

    return forbidden_tokens
