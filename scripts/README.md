# Scripts

These scripts regenerate everything in the download: data, models, probes, scores and the analyses
the notebooks read. Run them from the repository root with the environment active. Every script
takes `--help`.

## From scratch

The training datasets are not shipped, and the scripts regenerate every dataset from its recorded
seeds. The eight Rayworld training corpora take about 2.1 TB, about 440 GB for each 128-ray instance
(`standard`, `blink`, `smooth` and `128-ray`). Scoring and probe fits memory-map residual stacks of up
to about 22 GB into `.scratch/`. Training on a GPU is not bitwise deterministic, so a retrained model
matches the shipped one statistically rather than exactly. The steps run in order.

**1. Data.** The 8-ray tokenizer needs the 8-ray corpus and splits, and runs before the edit
selections, which filter on its vocabulary.

```bash
for I in standard blink smooth obs5 128-ray 16-ray 8-ray 5-ray
do
    python scripts/build_rayworld_corpus.py --instance $I
    python scripts/generate_dataset.py --instance $I --role eval
    python scripts/generate_dataset.py --instance $I --role edits
    python scripts/generate_dataset.py --instance $I --role probe --size 120k
    python scripts/generate_dataset.py --instance $I --role probe --size 250k
done
python scripts/make_rayworld_tokens.py --instance 8-ray
for I in standard blink smooth obs5 128-ray 16-ray 8-ray 5-ray
do
    python scripts/make_edit_selection.py --instance $I
done
for I in standard adjacent-flip adjacent-noflip standard-noflip
do
    python scripts/make_othello_corpus.py --instance $I
    python scripts/make_othello_edits.py --instance $I
done
```

**2. Training.** Every main run trains for 780,000 steps and the seed replicates for 512,000. Each run
is evaluated at its lowest-validation-loss checkpoint, `best_model.pt`, validated every 5,000 steps.
Seeds 1 and 2 are their best checkpoint within 512,000 steps, and seed 0 is the main run's
checkpoint at exactly 512,000 steps.

```bash
for I in standard blink smooth obs5 128-ray 16-ray 8-ray 5-ray
do
    python scripts/train.py --env rayworld --instance $I --run rayworld/$I --steps 780000
done
python scripts/train.py --env rayworld --instance 8-ray --repr tokens --run rayworld/8-ray-tokens --steps 780000
for I in standard adjacent-flip adjacent-noflip standard-noflip
do
    python scripts/train.py --env othello --instance $I --run othello/$I --steps 780000
done
for R in rayworld/standard rayworld/blink rayworld/128-ray rayworld/16-ray rayworld/8-ray rayworld/5-ray \
         othello/standard othello/adjacent-flip othello/adjacent-noflip othello/standard-noflip
do
    python scripts/make_replicate_member.py --run $R --step 512000
    for K in 1 2
    do
        python scripts/train.py --env ${R%/*} --instance ${R#*/} --run ${R}__seed$K --seed $K \
            --steps 512000 --replicate-of $R
    done
done
```

**3. Categorical probes.** The scorer reads categorical probes and their floors only from the probe
cache, so they are fit first.

```bash
for T in appearance-fac appearance grid-6x5 grid-10x3 grid-16x8 pos@appearance
do
    python scripts/fit_probes.py --run rayworld/8-ray --target $T
done
for R in standard blink 128-ray 16-ray 5-ray 8-ray-tokens {128-ray,16-ray,8-ray,5-ray}__seed{0,1,2}
do
    python scripts/fit_probes.py --run rayworld/$R --target appearance-fac
done
for R in 128-ray 16-ray 8-ray 5-ray
do
    python scripts/fit_probes.py --run rayworld/$R --target appearance-fac --random-init
    python scripts/fit_probes.py --run rayworld/$R --target appearance-fac --observation
done
python scripts/fit_probes.py --run rayworld/8-ray-tokens --target appearance-fac --random-init
python scripts/fit_probes.py --run rayworld/8-ray --target appearance --random-init
python scripts/fit_probes.py --run rayworld/8-ray --target appearance --observation
```

**4. Scoring.** The scorer fits the remaining probes and inverse maps, scores every editor and writes
each run's `scores.json`. The prediction blocks follow.

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 notebooks/master_eval.ipynb --output master_eval.executed.ipynb
python scripts/score_prediction.py
```

**5. Analyses.** These write what the table notebooks read beyond `scores.json`.

```bash
python scripts/bayes_floor.py
python scripts/reachability_table.py
python scripts/two_flip_editability.py --run othello/adjacent-noflip
python scripts/two_flip_editability.py --run othello/standard
python scripts/two_flip_editability.py --run othello/standard-noflip --no-legal
python scripts/im_reconstruction.py
python scripts/othello_flip_rates.py
for R in rayworld/8-ray othello/standard othello/adjacent-flip__seed{0,1,2}
do
    python scripts/probe_refit_variance.py --run $R
done
```

Then rebuild the tables and figures as in the main README.

## Quick check

These commands run the data, training and probe-fit scripts at toy size, in about a minute on one
GPU. Run them in a fresh clone without the download, because they write small datasets where the full
ones go.

```bash
python scripts/make_othello_corpus.py --instance standard --splits train --n-train 5000
python scripts/make_othello_edits.py --instance standard
python scripts/train.py --env othello --instance standard --run othello/quick --steps 100 --limit 5000
python scripts/build_rayworld_corpus.py --instance 8-ray --shards 2 --shard-n 4000
python scripts/generate_dataset.py --instance 8-ray --role eval --n 500
python scripts/generate_dataset.py --instance 8-ray --role edits --n 1000
python scripts/generate_dataset.py --instance 8-ray --role probe --size 120k --n 2000
python scripts/generate_dataset.py --instance 8-ray --role probe --size 250k --n 2000
python scripts/make_rayworld_tokens.py --instance 8-ray
python scripts/make_edit_selection.py --instance 8-ray --pool 1000 --n 200
python scripts/train.py --env rayworld --instance 8-ray --run rayworld/quick --steps 100
python scripts/train.py --env rayworld --instance 8-ray --repr tokens --run rayworld/quick-tokens --steps 100
python scripts/fit_probes.py --run rayworld/quick --target appearance-fac --cat-n-seq 1500 --cat-epochs 2
```

## Rescoring a downloaded run

Delete the run's `scores.json` (and its `probes/` to refit them) and run `master_eval.ipynb` again
with the corpora bundle in place. A Rayworld run whose `probes/` you delete needs its step-3
`fit_probes.py` lines first. The corpora bundle holds the 250k Rayworld probe corpus only for 128-ray,
16-ray, 8-ray and 5-ray, so refitting the categorical probes of `standard` and `blink`, or the large
observation floors of `standard`, `blink`, `smooth` and `obs5`, first needs
`python scripts/generate_dataset.py --instance <instance> --role probe --size 250k`. Afterwards,
`scripts/score_prediction.py --runs <run id>` adds back the prediction block, and the three
qualitative figure scripts need `--recompute`. Refitting an Othello floor caches the probe labels
beside the probe split, about 1.4 GB per instance.

A rescore reproduces most values to about 1e-6. Some runs, the token model and several seed
replicates among them, were scored on a different GPU, and their GS values can move by up to about
0.02 in Edit Index. Every reported setting and every rounded table value stays the same.

Without the corpora bundle, the Othello figure scripts regenerate the Othello probe splits (5 MB) in
place, byte for byte, within about a minute.

## Reading `scores.json`

The tables choose each reported setting from a run's `arms` with `pim.metrics.selection.best_arm`: the
highest Edit Index among settings with Edit Fidelity of at least 0, or else the highest Edit Fidelity
(`best_arm_by_fidelity` for the fidelity-selected table). The `best` entries are each editor's top
arm without that cutoff, so they can differ from the tables. Edit Fidelity is `1 - fidelity_ratio`,
and Othello's reported Edit Index is `edit_index_symdiff`. In a Rayworld run, `bases` maps each block
to its scores.

| Term | Meaning |
|---|---|
| run id | `<env>/<variant>`, with `__seed<k>` for a seed replicate |
| instance | an environment configuration, with its data in `datasets/<env>/<instance>/` |
| block | one probe target's scores inside `scores.json` |
| arm | one editor setting (residual point and step size) with its scores |
| bench | the 1000 edit cases an instance is scored on |
| basis | the coordinates of a Rayworld regression target: `cartesian` is reported, `frustum` keys the categorical probes |
| floors | the observation and random-init decodability baselines, and the Bayes floor, in `runs/_baselines/<env>/<instance>/` |
| Transformer-L | the 8-block, width-512 minGPT model every run uses |

## Demos

`demos/demo.py` animates one scene, the 2D world beside its 1D ray observation. In `demos/play.py`,
W/A/S/D and the arrow keys move the two discs, R resets and Q quits. Both take
`--save outputs/<name>.gif` to write a GIF instead of opening a window.
