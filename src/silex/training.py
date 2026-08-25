"""Training loop: teacher-forced fit on (curve, tokenized-sequence) pairs,
with optional validation + early stopping. A typed port of
transformer.py's Transformer.train/plot_training_history, minus the
pickle-everything save (see checkpoint.py) and print+ANSI logging."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import keras
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from matplotlib.figure import Figure

from silex.data import Dataset

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class TrainingResult:
    history: dict[str, list[float]]
    epochs_run: int


def train(
    model: keras.Model,
    train_data: Dataset,
    val_data: Dataset | None = None,
    epochs: int = 50,
    batch_size: int = 64,
    early_stopping_patience: int = 10,
    early_stopping_start_epoch: int = 10,
) -> TrainingResult:
    """Teacher-forced: the decoder is fed y[:, :-1] and scored against
    y[:, 1:] (predict-the-next-token), matching the sequence layout
    vocab.py/data.py build (leading [START], trailing [PAD])."""
    callbacks = []
    validation_data = None
    if val_data is not None:
        validation_data = (
            [val_data.x, val_data.y[:, :-1, ...]],
            val_data.y[:, 1:, ...],
        )
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                start_from_epoch=early_stopping_start_epoch,
                patience=early_stopping_patience,
            )
        )

    logger.info(
        "Training on %d samples%s for up to %d epochs (batch_size=%d)",
        train_data.x.shape[0],
        f", validating on {val_data.x.shape[0]}" if val_data is not None else "",
        epochs,
        batch_size,
    )

    history = model.fit(
        [train_data.x, train_data.y[:, :-1, ...]],
        train_data.y[:, 1:, ...],
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1,  # pyright: ignore[reportArgumentType]  # keras has always accepted int 0/1/2 here; the stub is just narrower than the runtime API
    )

    epochs_run = len(history.history["loss"])
    logger.info("Training finished after %d epochs", epochs_run)

    return TrainingResult(history=history.history, epochs_run=epochs_run)


def plot_history(result: TrainingResult) -> Figure:
    """Loss (solid) and accuracy (dashed) per epoch, train vs. validation
    if present."""
    fig, ax_loss = plt.subplots(figsize=(16, 10), dpi=150)

    ax_loss.plot(result.history["loss"], label="Training loss")
    if "val_loss" in result.history:
        ax_loss.plot(result.history["val_loss"], label="Validation loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Categorical crossentropy loss")

    ax_acc = ax_loss.twinx()
    ax_acc.plot(result.history["accuracy"], linestyle="--", label="Training accuracy")
    if "val_accuracy" in result.history:
        ax_acc.plot(result.history["val_accuracy"], linestyle="--", label="Validation accuracy")
    ax_acc.set_ylabel("Accuracy")

    lines = ax_loss.get_lines() + ax_acc.get_lines()
    labels = [str(line.get_label()) for line in lines]
    ax_loss.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    return fig
