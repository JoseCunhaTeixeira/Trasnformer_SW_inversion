"""Test-set evaluation: per-token accuracy, precision/recall/F1 per vocab
word, and the confusion matrix, decoding with the same algorithm as
decoding.py. A typed port of transformer.py's Transformer.evaluate, minus
the print+ANSI logging and the pickle-coupled save/plot side effects."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import keras
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from silex.checkpoint import VocabData
from silex.data import Dataset
from silex.decoding import decode

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    per_word_precision: dict[str, float]
    per_word_recall: dict[str, float]
    per_word_f1: dict[str, float]
    confusion_matrix: NDArray[np.int64]
    """(vocab_size, vocab_size), rows=target id, cols=decoded id."""
    accuracies: NDArray[np.float64]
    """Per-sample accuracy, same order as the evaluated dataset."""


def evaluate(
    model: keras.Model, output_seq_length: int, vocab_data: VocabData, test_data: Dataset
) -> EvaluationResult:
    vocab_size = len(vocab_data.index_to_word)
    vocab = [vocab_data.index_to_word[i] for i in range(vocab_size)]
    confusion_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    accuracies = np.zeros(test_data.x.shape[0])

    start_id = vocab_data.word_to_index["[START]"]
    end_id = vocab_data.word_to_index["[END]"]
    pad_id = vocab_data.word_to_index["[PAD]"]

    for i in tqdm(range(test_data.x.shape[0]), desc="Evaluating"):
        x = test_data.x[i : i + 1]
        target = test_data.y[i, 1:-1]  # drop the leading [START] and trailing [PAD]

        decoded = decode(
            model, x, vocab_data.forbidden_tokens, start_id, end_id, pad_id, output_seq_length
        )
        # decode() stops as soon as it emits [END]; pad back out to target's
        # fixed length so every position is compared, matching the original
        # Transformer.evaluate's always-fully-unrolled decode.
        decoded = decoded + [pad_id] * (len(target) - len(decoded))

        accuracies[i] = float(np.mean(np.array(decoded) == target))

        for decoded_id, target_id in zip(decoded, target, strict=True):
            confusion_matrix[target_id, decoded_id] += 1

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for i in range(vocab_size):
        true_positives = int(confusion_matrix[i, i])
        false_positives = int(confusion_matrix[:, i].sum()) - true_positives
        false_negatives = int(confusion_matrix[i, :].sum()) - true_positives

        precision = (
            math.nan
            if true_positives == 0 and false_positives == 0
            else true_positives / (true_positives + false_positives)
        )
        recall = (
            math.nan
            if true_positives == 0 and false_negatives == 0
            else true_positives / (true_positives + false_negatives)
        )
        f1 = (
            math.nan
            if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0
            else 2 * precision * recall / (precision + recall)
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return EvaluationResult(
        accuracy=float(np.nanmean(accuracies)),
        precision=float(np.nanmean(precisions)),
        recall=float(np.nanmean(recalls)),
        f1=float(np.nanmean(f1s)),
        per_word_precision=dict(zip(vocab, precisions, strict=True)),
        per_word_recall=dict(zip(vocab, recalls, strict=True)),
        per_word_f1=dict(zip(vocab, f1s, strict=True)),
        confusion_matrix=confusion_matrix,
        accuracies=accuracies,
    )
