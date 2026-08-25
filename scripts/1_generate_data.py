"""Generate synthetic training data: random soil profiles forward-modeled
(Santiludo rock physics + gpdc dispersion) into (dispersion curve, tokenized
profile) pairs, written under Paths.input/training_data/<site>/.

Needs Santiludo_layered built and installed (see repo README) and `gpdc`
(https://www.geopsy.org) on PATH.

Run from the repo root: python scripts/1_generate_data.py
"""

import logging

from silex.config import Paths
from silex.generation import GenerationConfig, generate_samples, save_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

site = "Grand_Est"

# SET THIS: the original script had a literal `N_models = None` here, which
# crashed immediately -- GenerationConfig.n_models has no default on
# purpose, so this can't silently happen again. The real trained checkpoint
# (models/[202407170928]) was generated from 1316446 *unique* samples after
# dedup, so start well above that if you want a comparable dataset size.
config = GenerationConfig(n_models=1_500_000)

seed = 0


def run() -> None:
    paths = Paths.from_env()
    samples = generate_samples(config, seed=seed)
    save_dataset(samples, config, paths.input / "training_data" / site)


if __name__ == "__main__":
    run()
