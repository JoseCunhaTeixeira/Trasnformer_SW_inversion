"""Evaluate a checkpoint on its held-out test split: per-token accuracy,
precision/recall/F1, and a confusion-matrix heatmap.

Run from the repo root: python scripts/4_evaluate.py
"""

import json
import logging

import matplotlib.pyplot as plt
import numpy as np

from silex.checkpoint import load_checkpoint
from silex.config import Paths
from silex.data import Dataset, load_dataset
from silex.evaluation import evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

site = "Grand_Est"
checkpoint_name = "latest"
n_test_samples = 50_000


def run() -> None:
    paths = Paths.from_env()
    data_dir = paths.input / "training_data" / site

    model, params, vocab = load_checkpoint(paths.models / checkpoint_name)
    logger.info("Loaded checkpoint %s", checkpoint_name)

    max_n_layers = json.loads((data_dir / "params.json").read_text())["max_N_layers"]

    dataset = load_dataset(
        data_dir,
        vocab.word_to_index,
        max_n_layers,
        params.min_freq,
        params.max_freq,
        min_vel=params.min_vel,
        max_vel=params.max_vel,
    )
    test_data = Dataset(
        x=dataset.x[-n_test_samples:],
        y=dataset.y[-n_test_samples:],
        min_vel=dataset.min_vel,
        max_vel=dataset.max_vel,
    )
    logger.info("Evaluating on %d test samples", test_data.x.shape[0])

    result = evaluate(model, params.output_seq_length, vocab, test_data)

    logger.info(
        "Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f",
        result.accuracy,
        result.precision,
        result.recall,
        result.f1,
    )

    out_dir = paths.output / checkpoint_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "accuracy": result.accuracy,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "per_word_precision": result.per_word_precision,
                "per_word_recall": result.per_word_recall,
                "per_word_f1": result.per_word_f1,
            },
            indent=2,
        )
    )

    fig, ax = plt.subplots(dpi=200, figsize=(10, 6))
    ax.scatter(range(len(result.accuracies)), result.accuracies, s=4)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    fig.savefig(out_dir / "accuracies.png", bbox_inches="tight")

    vocab_words = list(vocab.word_to_index.keys())
    fig, ax = plt.subplots(dpi=200, figsize=(12, 10))
    im = ax.imshow(np.log1p(result.confusion_matrix), cmap="Reds")
    ax.set_xticks(range(len(vocab_words)), vocab_words, rotation=90, fontsize=6)
    ax.set_yticks(range(len(vocab_words)), vocab_words, fontsize=6)
    ax.set_xlabel("Decoded")
    ax.set_ylabel("Target")
    fig.colorbar(im, ax=ax, label="log1p(count)")
    fig.savefig(out_dir / "confusion_matrix.png", bbox_inches="tight")

    logger.info("Saved evaluation results to %s", out_dir)


if __name__ == "__main__":
    run()
