"""Train the Silex model from scratch on generated training data, and save
a checkpoint sigpipe's SilexModel.load can read directly (no migration
step needed).

Run from the repo root: python scripts/2_train.py
"""

import json
import logging

from silex.checkpoint import CheckpointParams, save_checkpoint
from silex.config import Paths
from silex.data import Dataset, load_dataset
from silex.model import ModelConfig, build_model
from silex.training import plot_history, train
from silex.vocab import VocabParams, build_forbidden_tokens, build_index, build_vocab

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

site = "Grand_Est"
min_freq = 15.0
max_freq = 50.0
n_val_samples = 100_000
n_test_samples = 50_000
epochs = 100
batch_size = 64

checkpoint_name = "latest"


def _slice(dataset: Dataset, start: int, stop: int) -> Dataset:
    return Dataset(
        x=dataset.x[start:stop],
        y=dataset.y[start:stop],
        min_vel=dataset.min_vel,
        max_vel=dataset.max_vel,
    )


def run() -> None:
    paths = Paths.from_env()
    data_dir = paths.input / "training_data" / site

    data_params = json.loads((data_dir / "params.json").read_text())
    freqs = [f for f in data_params["freqs"] if min_freq <= f <= max_freq]

    vocab_params = VocabParams(
        soils=tuple(data_params["soils"]),
        max_n_layers=data_params["max_N_layers"],
        water_table_depths=tuple(data_params["WTs"]),
        n_values=tuple(data_params["Ns"]),
        thicknesses=tuple(data_params["thicknesses"]),
    )
    vocab = build_vocab(vocab_params)
    word_to_index, index_to_word = build_index(vocab)
    output_seq_length = vocab_params.max_n_layers * 6 + 4
    forbidden_tokens = build_forbidden_tokens(vocab_params, word_to_index, output_seq_length)

    logger.info("Loading dataset from %s", data_dir)
    dataset = load_dataset(data_dir, word_to_index, vocab_params.max_n_layers, min_freq, max_freq)

    train_end = dataset.x.shape[0] - n_val_samples - n_test_samples
    train_data = _slice(dataset, 0, train_end)
    val_data = _slice(dataset, train_end, train_end + n_val_samples)
    logger.info(
        "Split: %d train, %d val, %d test (held out, unused here)",
        train_data.x.shape[0],
        val_data.x.shape[0],
        n_test_samples,
    )

    model = build_model(
        ModelConfig(
            input_length=len(freqs),
            output_length=output_seq_length,
            vocab_size=len(vocab),
        )
    )
    model.summary()

    result = train(model, train_data, val_data, epochs=epochs, batch_size=batch_size)

    checkpoint_params = CheckpointParams(
        min_freq=min_freq,
        max_freq=max_freq,
        d_freq=float(freqs[1] - freqs[0]),
        n_freqs=len(freqs),
        min_vel=dataset.min_vel,
        max_vel=dataset.max_vel,
        output_seq_length=output_seq_length,
        word_to_index=word_to_index,
        index_to_word=index_to_word,
        forbidden_tokens=forbidden_tokens,
    )
    out_dir = paths.models / checkpoint_name
    save_checkpoint(model, checkpoint_params, out_dir)
    plot_history(result).savefig(out_dir / "training_history.png", bbox_inches="tight")
    logger.info("Saved checkpoint to %s", out_dir)


if __name__ == "__main__":
    run()
