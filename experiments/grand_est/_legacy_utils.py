"""Small standalone helpers from the original misc.py that the Grand_Est
site scripts still need, but that don't belong in the core silex package
(vocab.py/data.py cover misc.py's other functions -- these three are
QC/plotting-only). Light, typed port -- behavior unchanged."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d


def resample_legacy(
    f: NDArray[np.floating],
    v: NDArray[np.floating],
    axis_resamp: NDArray[np.floating] | None = None,
    type: str = "wavelength",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Port of misc.py's resamp: resample onto a frequency or wavelength
    axis (linear or log-spaced), extrapolating outside the observed range."""
    if axis_resamp is None:
        axis_resamp = np.arange(min(f), max(f), 0.1)

    if "wavelength" in type:
        w = v / f
        func_v = interp1d(w, v, fill_value="extrapolate")  # pyright: ignore[reportArgumentType]
        w_resamp = (
            axis_resamp if type == "wavelength" else np.geomspace(min(w), max(w), len(f))
        )
        v_resamp = func_v(w_resamp)
        return np.asarray(w_resamp, dtype=np.float64), np.asarray(v_resamp, dtype=np.float64)

    func_v = interp1d(f, v, fill_value="extrapolate")  # pyright: ignore[reportArgumentType]
    f_resamp = (
        axis_resamp
        if type == "frequency"
        else np.geomspace(min(axis_resamp), max(axis_resamp), len(axis_resamp))
    )
    v_resamp = func_v(f_resamp)
    return np.asarray(f_resamp, dtype=np.float64), np.asarray(v_resamp, dtype=np.float64)


def mode_filter_count(values: NDArray[np.floating]) -> int:
    """generic_filter callback: the most common value in the window, tied
    to the middle element on a tie (used to declump categorical soil/N
    maps -- see plot_inversion.py)."""
    counts = np.bincount(values.astype(int))
    if np.all(counts <= 1):
        return int(values[len(values) // 2])
    return int(counts.argmax())


def mode_filter_mean(values: NDArray[np.floating]) -> float:
    """generic_filter callback: nanmean of the window, silencing the
    all-NaN warning (used to smooth continuous physical-property maps)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mode = np.nanmean(values)
    return float(mode)
