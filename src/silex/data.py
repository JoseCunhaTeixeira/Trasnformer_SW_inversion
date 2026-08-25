"""Training-data loading: dispersion curves (X) and tokenized soil-profile
sequences (y). A typed port of misc.py's load_X / load_y / load_data /
resamp, same on-disk format (DCs.txt, GMs.txt, THKs.txt, WTs.txt, Ns.txt,
params.json produced by generation.py)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from silex.vocab import END, PAD, START, WT


def resample(
    freqs: NDArray[np.floating],
    values: NDArray[np.floating],
    target_freqs: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Linear interpolation onto `target_freqs`, extrapolating outside the
    observed range -- matches the model's own training-time preprocessing
    (sigpipe's SilexModel._preprocess relies on the same extrapolating
    behavior when inverting real field curves)."""
    interpolator = interp1d(
        freqs,
        values,
        fill_value="extrapolate",  # pyright: ignore[reportArgumentType]
    )
    return np.asarray(interpolator(target_freqs), dtype=np.float64)


@dataclass(slots=True, frozen=True)
class Dataset:
    x: NDArray[np.float32]
    """(N_samples, N_freqs, 1) -- min-max normalized dispersion curves."""
    y: NDArray[np.int32]
    """(N_samples, output_seq_length) -- tokenized soil-profile sequences."""
    min_vel: float
    max_vel: float


def _load_curves(
    data_dir: Path,
    min_freq: float,
    max_freq: float,
    min_vel: float | None,
    max_vel: float | None,
    noise: bool,
) -> tuple[NDArray[np.float32], float, float]:
    with (data_dir / "params.json").open(encoding="utf-8") as f:
        freqs = np.array(json.load(f)["freqs"])

    curves = np.loadtxt(data_dir / "DCs.txt")

    i_min = int(np.where(freqs == min_freq)[0][0])
    i_max = int(np.where(freqs == max_freq)[0][0])
    curves = curves[:, i_min : i_max + 1]

    if min_vel is None:
        min_vel = float(curves.min())
    if max_vel is None:
        max_vel = float(curves.max())

    normalized = (curves - min_vel) / (max_vel - min_vel)
    x = normalized.reshape(*normalized.shape, 1).astype(np.float32)

    if noise:
        x = x + np.random.default_rng().normal(loc=0.0, scale=0.01, size=x.shape).astype(np.float32)

    return x, min_vel, max_vel


def _load_sequences(
    data_dir: Path,
    word_to_index: dict[str, int],
    max_n_layers: int,
) -> NDArray[np.int32]:
    soils = np.loadtxt(data_dir / "GMs.txt", dtype=str)
    thicknesses = np.loadtxt(data_dir / "THKs.txt", dtype=str)
    water_tables = np.loadtxt(data_dir / "WTs.txt", dtype=str)
    n_values = np.loadtxt(data_dir / "Ns.txt", dtype=str)

    sequences: list[list[int]] = []
    for sample_soils, sample_thicknesses, sample_wt, sample_ns in zip(
        soils, thicknesses, water_tables, n_values, strict=True
    ):
        sequence = [
            word_to_index[START],
            word_to_index[WT],
            word_to_index[f"{float(sample_wt):.2f}"],
        ]

        n_layers = 0
        for soil, thickness, n in zip(sample_soils, sample_thicknesses, sample_ns, strict=True):
            if soil == "None" or thickness == "None":
                break
            sequence.append(word_to_index[f"[SOIL{n_layers + 1}]"])
            sequence.append(word_to_index[soil])
            sequence.append(word_to_index[f"[THICKNESS{n_layers + 1}]"])
            sequence.append(word_to_index[f"{float(thickness):.2f}"])
            sequence.append(word_to_index[f"[N{n_layers + 1}]"])
            sequence.append(word_to_index[f"{float(n):.2f}"])
            n_layers += 1

        sequence.append(word_to_index[END])
        sequence.extend([word_to_index[PAD]] * 6 * (max_n_layers - n_layers))
        sequence.append(word_to_index[PAD])

        sequences.append(sequence)

    return np.array(sequences, dtype=np.int32)


def load_dataset(
    data_dir: Path,
    word_to_index: dict[str, int],
    max_n_layers: int,
    min_freq: float,
    max_freq: float,
    min_vel: float | None = None,
    max_vel: float | None = None,
    noise: bool = False,
    n_samples: int | None = None,
) -> Dataset:
    x, resolved_min_vel, resolved_max_vel = _load_curves(
        data_dir, min_freq, max_freq, min_vel, max_vel, noise
    )
    y = _load_sequences(data_dir, word_to_index, max_n_layers)

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"X and y have different sample counts: {x.shape[0]} vs {y.shape[0]}")

    if n_samples is not None:
        x = x[:n_samples]
        y = y[:n_samples]

    return Dataset(x=x, y=y, min_vel=resolved_min_vel, max_vel=resolved_max_vel)
