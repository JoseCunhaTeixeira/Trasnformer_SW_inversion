"""Continue training an existing checkpoint on (possibly new) data, and
save the result as a new checkpoint.

Run from the repo root: python scripts/3_retrain.py
"""

import json
import logging

from silex.checkpoint import load_checkpoint, save_checkpoint
from silex.config import Paths
from silex.data import Dataset, load_dataset
from silex.training import plot_history, train

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

site = "Grand_Est"
source_checkpoint = "latest"
target_checkpoint = "retrained"

n_test_samples = 50_000
epochs = 5
batch_size = 64
add_noise = True
"""The original retraining script added Gaussian noise to the input curves
-- kept, since retraining on noised data is presumably deliberate
robustness fine-tuning, not an oversight."""


def run() -> None:
    paths = Paths.from_env()
    data_dir = paths.input / "training_data" / site

    model, params = load_checkpoint(paths.models / source_checkpoint)
    logger.info("Loaded checkpoint %s", source_checkpoint)

    max_n_layers = json.loads((data_dir / "params.json").read_text())["max_N_layers"]

    dataset = load_dataset(
        data_dir,
        params.word_to_index,
        max_n_layers,
        params.min_freq,
        params.max_freq,
        min_vel=params.min_vel,
        max_vel=params.max_vel,
        noise=add_noise,
    )
    if dataset.min_vel != params.min_vel or dataset.max_vel != params.max_vel:
        raise ValueError("Dataset min/max velocity differs from the checkpoint's normalization.")

    train_end = dataset.x.shape[0] - n_test_samples
    train_data = Dataset(
        dataset.x[:train_end], dataset.y[:train_end], dataset.min_vel, dataset.max_vel
    )
    logger.info(
        "Retraining on %d samples (%d held out as test)", train_data.x.shape[0], n_test_samples
    )

    result = train(model, train_data, val_data=None, epochs=epochs, batch_size=batch_size)

    out_dir = paths.models / target_checkpoint
    save_checkpoint(model, params, out_dir)
    plot_history(result).savefig(out_dir / "training_history.png", bbox_inches="tight")
    logger.info("Saved retrained checkpoint to %s", out_dir)


if __name__ == "__main__":
    run()
