"""Model architecture: a Conv1D feature encoder feeding a Transformer
encoder-decoder that autoregressively emits the tokenized soil-profile
sequence. A typed port of transformer.py's Transformer.build_model, with
training/eval/decode/plotting split out into their own modules (training.py,
evaluation.py, decoding.py, checkpoint.py) instead of one stateful class.

See silex/__init__.py for the KERAS_BACKEND fallback + tensorflow stub this
package sets up before any of that -- needed for `import keras_nlp` (used
below for TransformerEncoder/TransformerDecoder/etc.) to work at all on
Python versions TensorFlow doesn't ship wheels for.
"""

from __future__ import annotations

from dataclasses import dataclass

import keras
from keras import layers

# Verified working at runtime (build_model() below actually runs) --
# pyright just can't follow whatever lazy-submodule mechanism keras-hub's
# keras_nlp compat shim uses for `.layers`.
from keras_nlp.layers import (  # pyright: ignore[reportMissingImports]
    SinePositionEncoding,
    TokenAndPositionEmbedding,
    TransformerDecoder,
    TransformerEncoder,
)


@dataclass(slots=True, frozen=True)
class ModelConfig:
    input_length: int
    """N_freqs -- length of the (normalized) input dispersion curve."""
    output_length: int
    """output_seq_format.length -- decoder sequence length."""
    vocab_size: int

    encoder_emb_dim: int = 64
    decoder_emb_dim: int = 64
    intermediate_dim: int = 256
    num_heads: int = 8
    encoder_n_layers: int = 4
    decoder_n_layers: int = 4
    learning_rate: float = 1e-3


def build_model(config: ModelConfig) -> keras.Model:
    feature_encoder_inputs = keras.Input(shape=(config.input_length, 1))

    x = layers.Conv1D(16, 3, activation="relu", padding="same")(feature_encoder_inputs)
    x = layers.Conv1D(16, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
    x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(config.encoder_emb_dim, 3, activation="relu", padding="same")(x)
    feature_encoder_outputs = layers.Conv1D(
        config.encoder_emb_dim, 3, activation="relu", padding="same"
    )(x)

    position_encoding = SinePositionEncoding()(feature_encoder_outputs)
    encoder_hidden = feature_encoder_outputs + position_encoding
    for _ in range(config.encoder_n_layers):
        encoder_hidden = TransformerEncoder(
            intermediate_dim=config.intermediate_dim, num_heads=config.num_heads
        )(inputs=encoder_hidden)
    encoder_outputs = encoder_hidden

    decoder_inputs = keras.Input(shape=(None,))
    decoder_hidden = TokenAndPositionEmbedding(
        vocabulary_size=config.vocab_size,
        sequence_length=config.output_length,
        embedding_dim=config.decoder_emb_dim,
    )(decoder_inputs)
    for _ in range(config.decoder_n_layers):
        decoder_hidden = TransformerDecoder(
            intermediate_dim=config.intermediate_dim, num_heads=config.num_heads
        )(decoder_sequence=decoder_hidden, encoder_sequence=encoder_outputs)
    decoder_outputs = layers.Dense(config.vocab_size, activation="softmax")(decoder_hidden)

    model = keras.Model([feature_encoder_inputs, decoder_inputs], decoder_outputs, name="silex")
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
