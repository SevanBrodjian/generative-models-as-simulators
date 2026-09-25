# When Generative Models Act as Simulators

This repository holds the code, trained models and data for the paper *When Generative Models Act
as Simulators*. We ask when a generative model can act as a simulator, that is, when changing its
latent state produces the next-step outputs we get by editing a simulator's state. We test two kinds
of mapping between the model's latent state and the simulator's state, both learned after training
from observations paired with simulator states. Probes decode the state and are then used to steer
the model, and an inverse map writes the state directly. We compare them in Othello and in Rayworld,
our own continuous world of moving discs seen through partial 1D renderings.

![Overview of the evaluation](assets/teaser.png)

*We train a generator on observations, then set its latent state from a simulator state, either by
steering it with probes or by writing it with an inverse map.*

## Contributions

1. An evaluation framework of editability across diverse environments.
2. An investigation of two approaches for driving a generative model from simulators.
3. A demonstration that a model can be driven from simulator states using inverse maps when
   probe-derived editors fail, even when these maps predict latent states poorly.

## Setup

```bash
bash setup.sh
source .venv/bin/activate
```

This needs Python 3.12 or newer. The tables run on a CPU. Scoring, training and the figure scripts
need a CUDA GPU.

## Download

The trained models, fitted probes, scores and datasets are on
[Hugging Face](https://huggingface.co/datasets/AnonymousCompute/generative-models-as-simulators).
Download them into the repository root, where they fill `runs/` and `datasets/`. The core bundle
(2.9 GB) is all the tables and figures need.

```bash
pip install -U huggingface_hub
hf download AnonymousCompute/generative-models-as-simulators --repo-type dataset --local-dir . \
    --exclude ".gitattributes" --exclude "datasets/*/*/probe/*" \
    --exclude "runs/*/*__seed*/best_model.pt" --exclude "runs/*/*__seed*/probes/*"
```

Drop the last three `--exclude` flags to also fetch the probe corpora (9.4 GB) and the seed
replicates' weights and probes (3.5 GB), which rescoring needs.

## Reproduce the paper

`paper_tables.ipynb` builds the main-text tables and `appendix_tables.ipynb` every appendix table.

```bash
jupyter nbconvert --to notebook --execute notebooks/paper_tables.ipynb --output paper_tables.executed.ipynb
jupyter nbconvert --to notebook --execute notebooks/appendix_tables.ipynb --output appendix_tables.executed.ipynb
```

The figure scripts write to `outputs/figures/`.

```bash
python scripts/figures/qualitative_overview.py   # qualitative edits (main text)
python scripts/figures/qualitative_rayworld.py   # Rayworld edit grids (appendix)
python scripts/figures/qualitative_othello.py    # Othello edit grids (appendix)
python scripts/figures/editability_by_point.py   # editability across residual points (appendix)
python scripts/figures/history_rewrite.py        # history rewriting (appendix)
python scripts/figures/predictions.py            # predictions (appendix)
```

`notebooks/master_eval.ipynb` is the scorer that wrote every `scores.json`. On the downloaded runs it
has nothing left to do.

## Train from scratch

The training datasets are not shipped. The scripts regenerate every dataset from its recorded seeds,
train every model, fit the probes and run the analyses. [`scripts/README.md`](scripts/README.md) lists
the full pipeline in order, with a toy-size quick check.

## Variants

| Environment | Run ids |
|---|---|
| Othello | `othello/standard`, `othello/adjacent-flip`, `othello/adjacent-noflip`, `othello/standard-noflip` |
| Rayworld | `rayworld/standard`, `rayworld/blink`, `rayworld/128-ray`, `rayworld/16-ray`, `rayworld/8-ray`, `rayworld/5-ray`, `rayworld/smooth`, `rayworld/obs5`, `rayworld/8-ray-tokens` |

Each run lives in `runs/<run id>/`. The ten main-table variants also have seed replicates in
`runs/<run id>__seed{0,1,2}/`.

## Demos

```bash
python scripts/demos/demo.py --seed 8 --n-objects 4 --fixed-reflectivities   # an animated Rayworld scene
python scripts/demos/play.py                                                 # play Rayworld
```

## License

MIT (`LICENSE`). The vendored Othello engine and minGPT in `pim/environments/othello/vendor/` come from
Li et al.'s othello_world, whose minGPT derives from Karpathy's minGPT. Both keep their MIT licenses.

## Citation

```bibtex
@misc{anonymous2026simulators,
  title  = {When Generative Models Act as Simulators},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```
