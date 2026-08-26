"""Synthetic training-data generation: sample random soil profiles,
forward-model them (Santiludo rock physics + dispersion), and write
the (dispersion curve, tokenized profile) pairs data.py/vocab.py read.

A typed port of 1_run_data_generation.py's core loop. Concrete fixes along
the way: `n_models` is now a required parameter (the original had a
literal `N_models = None`, which crashed `while i < N_models` on the very
first iteration -- must have been a placeholder never filled in); the
dispersion computation now goes through santiludo's public
`compute_rock_physics`/`compute_seismic_forward` API instead of hand-rolling
calls into its low-level compiled modules and our own `gpdc` subprocess
call (santiludo now owns that, selectable via `GenerationConfig.backend`
-- `"disba"`, the default, is a pure-Python dependency of santiludo itself
and needs no external binary; `"gpdc"` is still available opt-in and needs
the Geopsy binary on PATH); and, as a side effect of that switch,
`_forward_model` no longer passes `frac` as a single
`config.frac * len(soils)` scalar into the Hertz-Mindlin frame computation
(santiludo's low-level `hertzMindlin` indexes `fracs` per layer -- this was
a pre-existing bug, confirmed against `invert_qc.py`'s own already-correct
`fracs = [0.3] * len(soil_types)`) -- santiludo's `Layer.frac` is now set
per layer instead.

Needs the optional `generation` extra (`uv sync --extra generation`),
which pulls in `santiludo @ git+https://github.com/JoseCunhaTeixeira/santiludo.git`
-- kept out of the base dependencies since it's a compiled Cython/C++
extension (building it needs a C++ toolchain; on Windows specifically,
MSVC via Visual Studio Build Tools' "Desktop development with C++"
workload -- without it the build fails with "Unable to find a compatible
Visual Studio installation", same as it would for any other
compiled-extension dependency). The santiludo import itself stays deferred
to inside `_forward_model()` so that importing this module doesn't require
the extra to be installed.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class RockPhysicsConstants:
    """Grain/fluid mechanical properties Santiludo's rock-physics forward
    model needs -- constant across all generated samples."""

    rho_water: float = 1000.0
    rho_air: float = 1.0
    k_water: float = 2.3e9
    k_air: float = 1.01e5
    gravity: float = 9.82

    mu_clay: float = 6.8
    mu_silt: float = 45.0
    mu_sand: float = 45.0
    k_clay: float = 25.0
    k_silt: float = 37.0
    k_sand: float = 37.0
    rho_clay: float = 2580.0
    rho_silt: float = 2600.0
    rho_sand: float = 2600.0

    pressure_model: int = 3
    """Santiludo's `kk`: 1=constant Pe, 2=Pe without suction, 3=Pe with
    suction (Solazzi et al. 2021) -- kept at the original's choice."""


@dataclass(slots=True, frozen=True)
class GenerationConfig:
    n_models: int
    """Required -- the original had `N_models = None`, which crashed on
    the very first loop iteration. No default here on purpose."""

    soils: tuple[str, ...] = ("clay", "loam", "silt", "sand")
    max_n_layers: int = 4
    layer_weights: tuple[float, ...] = (0.01, 0.05, 0.25, 0.69)
    """Probability of generating 1, 2, 3, 4 layers respectively."""

    min_thickness: float = 0.5
    d_thickness: float = 0.5
    max_depth: float = 20.0

    min_water_table: float = 1.0
    max_water_table: float = 10.0
    d_water_table: float = 1.0

    n_values: tuple[int, ...] = (6, 7, 8, 9, 10)
    frac: float = 0.3
    """Fraction of non-slipping grains per layer (Hertz-Mindlin)."""

    dz: float = 0.1
    """Depth sample interval [m]. Halved by 10x on repeated dispersion
    failures for a given sample, then reset for the next one."""
    min_dz: float = 0.001
    """Give up shrinking dz below this and skip the sample."""

    under_layers: str = "5 2000 1000 2000\n0 4000 2000 2500\n"
    """GPDC-format substratum layers under the generated soil column.
    Kept as this string (rather than a list of santiludo.UnderLayer) so
    that this module stays importable without santiludo installed, and so
    params.json's on-disk format (read by invert_qc.py) doesn't change --
    parsed into santiludo.UnderLayer inside _forward_model instead."""

    d_freq: float = 1.0
    min_freq: float = 5.0
    max_freq: float = 100.0
    n_modes: int = 1
    wave: str = "R"
    """Wave type: "R" for Rayleigh, "L" for Love."""
    frequency_domain: str = "frequency"
    """Sampling mode: "frequency" or "wavelength" (gpdc backend only --
    santiludo's disba backend only supports frequency sampling)."""

    backend: Literal["gpdc", "disba"] = "disba"
    """Dispersion-curve computation backend. "disba" (default) is a
    pure-Python dependency of santiludo itself -- no external binary
    needed. "gpdc" shells out to the Geopsy binary, which must be on
    PATH."""

    rock_physics: RockPhysicsConstants = field(default_factory=RockPhysicsConstants)


@dataclass(slots=True, frozen=True)
class GeneratedSample:
    dispersion_curve: NDArray[np.float64]
    soils: list[str]
    """Padded with "None" out to max_n_layers, matching the on-disk format
    data.py's _load_sequences reads (a real soil is never named "None")."""
    thicknesses: list[float | None]
    n_values: list[float | None]
    water_table_depth: float


def _generate_layers(
    config: GenerationConfig, rng: random.Random
) -> tuple[list[str], list[float], list[int]]:
    """Sample layer count/soils/thicknesses/N, never repeating the same
    soil in two consecutive layers (matches the original's `tmp`-list
    remove/append dance)."""
    n_layers = rng.choices(range(1, config.max_n_layers + 1), weights=config.layer_weights, k=1)[0]

    thicknesses = _generate_thicknesses(n_layers, config, rng)

    soils: list[str] = []
    n_values: list[int] = []
    available = list(config.soils)
    previous_soil: str | None = None
    for _ in range(n_layers):
        n_values.append(rng.choice(config.n_values))
        if previous_soil is not None:
            available.remove(previous_soil)
        soil = rng.choice(available)
        soils.append(soil)
        if previous_soil is not None:
            available.append(previous_soil)
        previous_soil = soil

    return soils, thicknesses, n_values


def _generate_thicknesses(
    n_layers: int, config: GenerationConfig, rng: random.Random
) -> list[float]:
    """Faithful port of the original misc.generate_numbers: pick n_layers-1
    values independently from [min_thickness, max_depth] (step
    d_thickness, with replacement), set the last layer to whatever makes
    the sum exactly max_depth, retry (all n_layers values, not just the
    last) if that's <= 0 or > max_depth. Kept as the original's exact
    sampling distribution rather than a redesigned one, per José wanting
    the data-generation numerics unchanged."""
    possible_values = np.arange(
        config.min_thickness, config.max_depth + config.d_thickness, config.d_thickness
    ).tolist()
    while True:
        thicknesses = rng.choices(possible_values, k=n_layers - 1)
        last = config.max_depth - sum(thicknesses)
        if last <= 0 or last > config.max_depth:
            continue
        thicknesses.append(last)
        rng.shuffle(thicknesses)
        return thicknesses


def _forward_model(
    soils: list[str],
    thicknesses: list[float],
    n_values: list[int],
    water_table_depth: float,
    dz: float,
    config: GenerationConfig,
) -> NDArray[np.float64]:
    """Santiludo rock physics -> dispersion curve, for one soil profile at
    one dz. Raises RuntimeError if the dispersion curve can't be resolved
    at this dz (the caller retries at a finer dz)."""
    # santiludo's compiled extension modules need a C++ toolchain to build
    # (see the module docstring) -- an environment without one still fails
    # to resolve this import even though it's a normal pyproject.toml
    # dependency, hence the ignore rather than that being a real error here.
    from santiludo import (  # pyright: ignore[reportMissingImports]
        DispersionConfig,
        FluidProperties,
        GrainProperties,
        Layer,
        UnderLayer,
        compute_rock_physics,
        compute_seismic_forward,
    )

    rp = config.rock_physics
    layers = [
        Layer(soiltype=soil, thickness=thickness, N=n, frac=config.frac)
        for soil, thickness, n in zip(soils, thicknesses, n_values, strict=True)
    ]
    rock_physics = compute_rock_physics(
        layers,
        WT=water_table_depth,
        dz=dz,
        kk=rp.pressure_model,
        grain_properties=GrainProperties(
            mu_clay=rp.mu_clay,
            mu_silt=rp.mu_silt,
            mu_sand=rp.mu_sand,
            k_clay=rp.k_clay,
            k_silt=rp.k_silt,
            k_sand=rp.k_sand,
            rho_clay=rp.rho_clay,
            rho_silt=rp.rho_silt,
            rho_sand=rp.rho_sand,
        ),
        fluid_properties=FluidProperties(
            rhow=rp.rho_water, rhoa=rp.rho_air, kw=rp.k_water, ka=rp.k_air
        ),
        g=rp.gravity,
    )

    # config.under_layers stays the GPDC-format string (see its docstring)
    # -- parsed into santiludo.UnderLayer here, where the deferred
    # santiludo import already lives.
    under_layers = tuple(
        UnderLayer(*(float(v) for v in line.split()))
        for line in config.under_layers.splitlines()
        if line.strip()
    )

    n_freqs = int((config.max_freq - config.min_freq) / config.d_freq) + 1
    dispersion = DispersionConfig(
        nf=n_freqs,
        df=config.d_freq,
        min_f=config.min_freq,
        n_modes=config.n_modes,
        wave=config.wave,
        mode=config.frequency_domain,
        backend=config.backend,
    )

    result = compute_seismic_forward(rock_physics, under_layers=under_layers, dispersion=dispersion)
    if not result.dispersion_data:
        raise RuntimeError("no dispersion mode resolved at this dz")
    return np.asarray(result.dispersion_data[0][:, 1], dtype=np.float64)


def generate_samples(config: GenerationConfig, seed: int | None = None) -> list[GeneratedSample]:
    # Fail fast rather than letting every sample burn through the dz-shrink
    # retry loop below for a cause (missing binary) more dz won't fix.
    if config.backend == "gpdc" and shutil.which("gpdc") is None:
        raise RuntimeError(
            "GenerationConfig.backend='gpdc' but the 'gpdc' executable was not "
            "found on PATH (https://www.geopsy.org). Install it, or use "
            "backend='disba' (the default) to avoid this dependency."
        )

    rng = random.Random(seed)
    samples: list[GeneratedSample] = []

    with tqdm(total=config.n_models, desc="Generating") as progress:
        while len(samples) < config.n_models:
            soils, thicknesses, n_values = _generate_layers(config, rng)
            water_table_depth = rng.choice(
                np.arange(
                    config.min_water_table,
                    config.max_water_table + config.d_water_table,
                    config.d_water_table,
                ).tolist()
            )

            dz = config.dz
            curve: NDArray[np.float64] | None = None
            while curve is None:
                try:
                    curve = _forward_model(
                        soils, thicknesses, n_values, water_table_depth, dz, config
                    )
                except RuntimeError:
                    dz /= 10
                    if dz < config.min_dz:
                        logger.warning(
                            "Skipping sample (dispersion kept failing down to dz=%s): "
                            "soils=%s thicknesses=%s n=%s wt=%s",
                            dz,
                            soils,
                            thicknesses,
                            n_values,
                            water_table_depth,
                        )
                        break

            if curve is None:
                continue

            padding = config.max_n_layers - len(soils)
            samples.append(
                GeneratedSample(
                    dispersion_curve=curve,
                    soils=[*soils, *(["None"] * padding)],
                    thicknesses=[*thicknesses, *([None] * padding)],
                    n_values=[*n_values, *([None] * padding)],
                    water_table_depth=water_table_depth,
                )
            )
            progress.update(1)

    return samples


def save_dataset(samples: list[GeneratedSample], config: GenerationConfig, out_dir: Path) -> None:
    """Writes DCs.txt/GMs.txt/THKs.txt/WTs.txt/Ns.txt/params.json in the
    exact format data.py's load_dataset and vocab.py's VocabParams read.
    Drops exact-duplicate dispersion curves first, same as the original."""
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = np.array([s.dispersion_curve for s in samples])
    _, unique_indices = np.unique(curves, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    if len(unique_indices) < len(samples):
        logger.info(
            "Dropping %d duplicate curves (%d -> %d)",
            len(samples) - len(unique_indices),
            len(samples),
            len(unique_indices),
        )
    unique_samples = [samples[i] for i in unique_indices]

    np.savetxt(
        out_dir / "DCs.txt", np.array([s.dispersion_curve for s in unique_samples]), fmt="%.3f"
    )
    np.savetxt(out_dir / "GMs.txt", np.array([s.soils for s in unique_samples]), fmt="%s")
    np.savetxt(out_dir / "THKs.txt", np.array([s.thicknesses for s in unique_samples]), fmt="%s")
    np.savetxt(
        out_dir / "WTs.txt", np.array([s.water_table_depth for s in unique_samples]), fmt="%s"
    )
    np.savetxt(out_dir / "Ns.txt", np.array([s.n_values for s in unique_samples]), fmt="%s")

    freqs = np.arange(config.min_freq, config.max_freq + config.d_freq, config.d_freq)
    np.savetxt(out_dir / "fs.txt", freqs, fmt="%s")

    all_curves = np.array([s.dispersion_curve for s in unique_samples])
    all_thicknesses = np.arange(
        config.min_thickness, config.max_depth + config.d_thickness, config.d_thickness
    )
    all_water_tables = np.arange(
        config.min_water_table, config.max_water_table + config.d_water_table, config.d_water_table
    )

    params = {
        "N_samples": len(unique_samples),
        "soils": list(config.soils),
        "max_N_layers": config.max_n_layers,
        "d_thickness": config.d_thickness,
        "min_thickness": config.min_thickness,
        "max_thickness": config.max_depth,
        "thicknesses": all_thicknesses.tolist(),
        "max_depth": config.max_depth,
        "d_WT": config.d_water_table,
        "min_WT": config.min_water_table,
        "max_WT": config.max_water_table,
        "WTs": all_water_tables.tolist(),
        "Ns": list(config.n_values),
        "frac": config.frac,
        "d_freq": config.d_freq,
        "min_freq": config.min_freq,
        "max_freq": config.max_freq,
        "freqs": freqs.tolist(),
        "N_freqs": all_curves.shape[1],
        "min_vel": float(all_curves.min()),
        "max_vel": float(all_curves.max()),
        "n_modes": config.n_modes,
        "under_layers": config.under_layers,
        "N_under_layers": config.under_layers.count("\n"),
        "dz": config.dz,
        "top_surface_level": config.dz,
    }
    (out_dir / "params.json").write_text(json.dumps(params, indent=2))
