"""Greedy, grammar-masked autoregressive decode.

The same algorithm as sigpipe's SilexModel._decode (traced there against
the original Transformer.decode_seq_restrictive/RestrictiveSampler and
verified against real trained weights this same model produced) --
reimplemented directly rather than via keras-nlp's Sampler machinery, so
this module only needs `keras` itself, not keras-nlp's samplers submodule.
"""

from __future__ import annotations

from typing import Any

import keras
import numpy as np
from numpy.typing import NDArray


def decode(
    model: keras.Model,
    x: NDArray[np.float32],
    forbidden_tokens: list[list[int]],
    start_id: int,
    end_id: int,
    pad_id: int,
    output_seq_length: int,
) -> list[int]:
    """Decode one example. `x` must already be shaped (1, N_freqs, 1).

    forbidden_tokens[step] masks the logits used to fill prompt position
    step + 1, enforcing the [WT] <v> ([SOILi] <soil> [THICKNESSi] <v>
    [Ni] <v>)* [END] grammar (see vocab.build_forbidden_tokens).
    """
    prompt = np.full((1, output_seq_length), pad_id, dtype=np.int32)
    prompt[0, 0] = start_id

    decoded: list[int] = []
    for step in range(output_seq_length - 1):
        # keras's stubs leave Model.__call__/ops.convert_to_numpy's return types
        # too loose for pyright to narrow even with reportUnknown* off (same
        # gap sigpipe's SilexModel._decode hits) -- np.asarray(..., dtype=...)
        # below re-establishes a concrete type for everything downstream.
        logits: Any = model([x, prompt], training=False)
        raw_step_logits = keras.ops.convert_to_numpy(  # pyright: ignore[reportIndexIssue, reportOptionalSubscript]
            logits[:, step, :]
        )[0]
        step_logits = np.asarray(raw_step_logits, dtype=np.float32)
        step_logits[forbidden_tokens[step]] = -np.inf
        next_id = int(np.argmax(step_logits))
        decoded.append(next_id)
        if next_id == end_id:
            break
        prompt[0, step + 1] = next_id

    return decoded
