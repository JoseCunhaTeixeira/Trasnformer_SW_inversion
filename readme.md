# silex — Surface wave Inversion Lexicon

![Git_NLP_Passive](https://github.com/user-attachments/assets/a6935f75-98c6-4dba-aeee-8822e8e33ee0)

**silex** solves petrophysical inversion of surface-wave dispersion curves with a
Transformer inspired by neural machine translation and speech recognition: it infers
a textual sequence describing the propagating medium (soil layers, thicknesses,
compaction, water table) directly from an observed dispersion curve. 2D soil profiles
(lithofacies, petrophysical parameters, groundwater table height) are then built in
post-processing from the model's textual output.

This project is part of José Cunha Teixeira's PhD developments at SNCF Réseau,
Sorbonne Université and Mines Paris - PSL. Paper: [Cunha Teixeira et al.,
2025](https://doi.org/10.1029/2025GL114852).

## Citation

If you use this software, please cite the paper above (full metadata in
[CITATION.cff](CITATION.cff)):

```bibtex
@article{cunhateixeira2025silex,
  title   = {Neural Machine Translation of Seismic Ambient Noise for Soil Nature and Water Saturation Characterization},
  author  = {Cunha Teixeira, Jos\'{e} and Bodet, Ludovic and Rivi\`{e}re, Agn\`{e}s and Solazzi, Santiago G. and Hallier, Am\'{e}lie and Gesret, Alexandrine and El Janyani, Sanae and Dangeard, Marine and Dhemaied, Amine and Boisson Gaboriau, Jos\'{e}phine},
  journal = {Geophysical Research Letters},
  volume  = {52},
  number  = {13},
  year    = {2025},
  doi     = {10.1029/2025GL114852}
}
```

![Screenshot from 2024-07-08 15-18-04](https://github.com/JoseCunhaTeixeira/Trasnformer_SW_inversion/assets/148117375/074bd457-1acf-40c9-8ae7-ea47b9027de7)

![image](https://github.com/JoseCunhaTeixeira/Trasnformer_SW_inversion/assets/148117375/ba5a8a20-7c00-49be-9095-3487d9df0950)

![image](https://github.com/JoseCunhaTeixeira/Trasnformer_SW_inversion/assets/148117375/93cce05d-8dc7-4667-8e05-5a40477c80fa)

## Repository layout

```
src/silex/          Core package: config, vocab, data, model, decoding,
                     checkpoint, training, evaluation, generation
scripts/             Thin, directly-runnable entrypoints (data generation,
                     training, retraining, evaluation)
experiments/grand_est/  Site-specific QC and paper-figure scripts for the
                     Grand_Est field site -- light cleanup only, not held
                     to the same bar as src/silex/
models/              Trained checkpoints (tracked in git)
input/, output/      Training data and run outputs (gitignored)
```

A trained checkpoint for the model used in the paper already ships at
`models/[202407170928]/` (`silex.keras` + `silex_params.json`). This is also the
checkpoint format loaded by [sigpipe](https://github.com/JoseCunhaTeixeira/sigpipe)'s
inference-only `SilexModel` for production use -- a checkpoint saved here with
`silex.checkpoint.save_checkpoint` needs no conversion step to be usable there.

## Setup

Requires Python >=3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This installs the core package (`keras`, `keras-nlp`, `torch`, `numpy`, `scipy`,
`pandas`, `tqdm`, `matplotlib`) -- enough for training, retraining, and evaluation
on already-generated data.

Two optional extras cover the rest:

```bash
uv sync --extra generation      # synthetic training-data generation
uv sync --extra paper_figures   # experiments/grand_est/plot_paper_figures.py
```

`generation` pulls in [santiludo](https://github.com/JoseCunhaTeixeira/santiludo), a
compiled Cython/C++ rock-physics package. Building it needs a C++ toolchain -- on
Windows specifically, MSVC via Visual Studio Build Tools' "Desktop development with
C++" workload; without it `uv sync --extra generation` fails with "Unable to find a
compatible Visual Studio installation." Data generation also needs the `gpdc`
CLI binary from [Geopsy](https://www.geopsy.org) on `PATH`.

## Configuration

All filesystem locations are read from environment variables via `silex.config.Paths`,
each defaulting to a folder next to the repo root:

| Variable | Default | Contents |
|---|---|---|
| `SILEX_INPUT_DIR` | `./input` | Raw and generated training data |
| `SILEX_MODELS_DIR` | `./models` | Trained checkpoints |
| `SILEX_OUTPUT_DIR` | `./output` | Inversion/evaluation outputs |

## Usage

Run everything from the repo root.

```bash
python scripts/1_generate_data.py   # synthetic training data -> input/training_data/<site>/
python scripts/2_train.py           # train from scratch -> models/<checkpoint>/
python scripts/3_retrain.py         # continue training an existing checkpoint
python scripts/4_evaluate.py        # accuracy/precision/recall/F1 + confusion matrix
```

Each script is a thin entrypoint: open it and edit the module-level config (site,
frequency band, sample counts, epochs, checkpoint names) before running -- there's no
CLI argument parsing by design, to keep run configuration visible and typed.

For inference on real field dispersion curves, see
[sigpipe](https://github.com/JoseCunhaTeixeira/sigpipe)'s `SilexModel`, which loads the
checkpoint format this repo produces without any conversion step.

### Grand_Est site scripts

`experiments/grand_est/` holds the QC and paper-figure scripts specific to the
Grand_Est field site (hardcoded piezometer files, profile names, date ranges, panel
layouts). Run them the same way, e.g. `python experiments/grand_est/invert_qc.py`.

## Development

Type checking and linting run via `npx`/`uvx` rather than as project dependencies
(matching this project's other repos):

```bash
npx --yes pyright         # strict type checking (src/, scripts/, experiments/)
uvx ruff check .
uvx ruff format .
```

## License

MIT — see [LICENSE](LICENSE).
