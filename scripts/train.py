#!/usr/bin/env python
"""Train a Transformer-L on an instance's training corpus into runs/<run id>/.

Rayworld trains on frames (MSE on the next frame) or, with --repr tokens, on the instance's frame
vocabulary; Othello trains on move tokens (cross-entropy). The recipe is pim.training.TrainConfig.

    python scripts/train.py --env rayworld --instance 8-ray --run rayworld/8-ray --steps 780000
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
import sys
from pathlib import Path

# cap CPU thread pools before torch loads; the loop is GPU-bound
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from pim.environments.layout import DEFAULT_INSTANCE  # noqa: E402
from pim.models import build as build_model  # noqa: E402
from pim.training import TrainConfig, rayworld_source, token_source, train  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RECIPE = {f.name: f.default for f in dataclasses.fields(TrainConfig) if f.name != "steps"}


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, choices=("rayworld", "othello"))
    p.add_argument("--instance", default=None, help="environment instance (default: standard)")
    p.add_argument("--run", required=True, help="run id <env>/<variant>; written to runs/<run>/")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--repr", choices=("frames", "tokens"), default=None,
                   help="Rayworld input: frames (default) or the instance's frame vocabulary")
    p.add_argument("--seed", type=int, default=RECIPE["seed"])
    p.add_argument("--replicate-of", default=None, metavar="RUN",
                   help="mark the run as a seed replicate of run id RUN (name it RUN__seed<k>)")
    p.add_argument("--resume", action="store_true",
                   help="continue from ckpt/latest.pt up to --steps")
    p.add_argument("--limit", type=int, default=None,
                   help="train on the first N sequences only (quick checks)")
    p.add_argument("--batch-size", type=int, default=RECIPE["batch_size"])
    p.add_argument("--lr", type=float, default=RECIPE["lr"])
    p.add_argument("--weight-decay", type=float, default=RECIPE["weight_decay"])
    p.add_argument("--grad-clip", type=float, default=RECIPE["grad_clip"])
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default=RECIPE["lr_schedule"])
    p.add_argument("--warmup-steps", type=int, default=RECIPE["warmup_steps"])
    p.add_argument("--ckpt-base", type=int, default=RECIPE["ckpt_base"])
    p.add_argument("--val-every", type=int, default=RECIPE["val_every"])
    return p.parse_args()


def _rel(p: Path) -> str:
    """Repo-relative path (datasets/ may be a symlink, so ``p`` is not resolved)."""
    return str(Path(p).relative_to(REPO))


def _rayworld_frames(inst: str):
    """(frame memmap, n_total, obs_dim, block, meta) of the instance's training corpus."""
    from pim.environments.rayworld import bigcorpus as bc

    bc.use_instance(inst)
    if not (bc.OUT / "corpus.json").exists():
        raise SystemExit(f"no training corpus in {_rel(bc.OUT)}; build it with "
                         f"python scripts/build_rayworld_corpus.py --instance {inst}")
    obs = bc.open_obs("r")
    return (obs, int(obs.shape[0]), bc.OBS_RES, bc.FRAMES - 1,
            {"instance": bc.INSTANCE, "corpus": _rel(bc.OUT)})


def _rayworld_tokens(inst: str, run_dir: Path):
    """(tokens, lengths, vocab size, block, meta) of the instance's frame vocabulary; vocab.npz
    is copied into the run directory."""
    from pim.environments.layout import tokens_dir
    from pim.environments.rayworld.tokens import load_tokens

    tdir = tokens_dir(inst)
    if not (tdir / "vocab.npz").exists():
        raise SystemExit(f"no frame vocabulary in {_rel(tdir)}; make it with "
                         f"python scripts/make_rayworld_tokens.py --instance {inst}")
    tok, ln, vocab, tmeta = load_tokens(tdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(tdir / "vocab.npz", run_dir / "vocab.npz")
    return (tok, ln, int(vocab.size), int(tmeta["n_frames"]) - 1,
            {"instance": inst, "repr": "tokens", "corpus": _rel(tdir / "train.i16"),
             "vocab": _rel(tdir / "vocab.npz"), "vocab_size": int(vocab.size)})


def _othello_tokens(inst: str, limit):
    """(tokens, lengths, vocab size, block, meta) of the instance's move corpus."""
    from pim.environments.layout import train_dir
    from pim.environments.othello import corpus as oc
    from pim.environments.othello.data import T_MODEL, VOCAB

    n = limit or oc.N_TRAIN_GAMES
    have = sorted(int(q.stem.split("_")[-1]) for q in train_dir("othello", inst).glob("train_*.npz")
                  if q.stem.split("_")[-1].isdigit())
    if not any(m >= n for m in have):
        hint = f", or train on the {have[-1]:,} games there with --limit {have[-1]}" if have else ""
        raise SystemExit(f"no train split of {n:,} games in {_rel(train_dir('othello', inst))}; make it "
                         f"with python scripts/make_othello_corpus.py --instance {inst} --splits train"
                         + ("" if n == oc.N_TRAIN_GAMES else f" --n-train {n}") + hint)
    paths = oc.build(n, only=("train",), instance=inst)
    tok, ln = oc.load(paths["train"])
    vocab, block = int(tok.max()) + 1, int(tok.shape[1]) - 1
    assert (vocab, block) == (VOCAB, T_MODEL), (vocab, block, VOCAB, T_MODEL)
    return tok, ln, vocab, block, {"instance": inst, **oc.rules_of(inst),
                                   "corpus": _rel(paths["train"])}


def main() -> None:
    a = _parse()
    cfg = TrainConfig(steps=a.steps, batch_size=a.batch_size, lr=a.lr,
                      weight_decay=a.weight_decay, grad_clip=a.grad_clip,
                      lr_schedule=a.lr_schedule, warmup_steps=a.warmup_steps,
                      ckpt_base=a.ckpt_base, val_every=a.val_every, seed=a.seed)
    inst = a.instance or DEFAULT_INSTANCE[a.env]
    run_dir = REPO / "runs" / a.run
    repr_ = a.repr or ("tokens" if a.env == "othello" else "frames")
    if a.env == "othello" and repr_ == "frames":
        raise SystemExit("Othello observations are move tokens (--repr tokens)")

    if repr_ == "frames":
        obs, n_total, dim, block, meta = _rayworld_frames(inst)
        arch, mc = "transformer_l", {"obs_res": dim, "block_size": block}
        try:
            source = rayworld_source(obs, n_total=n_total, batch_size=a.batch_size, seed=a.seed,
                                     device=DEV, limit=a.limit, meta=meta)
        except ValueError as e:     # too few sequences for one training block
            raise SystemExit(f"{e}; the corpus holds {n_total:,} (build_rayworld_corpus.py "
                             f"--shards / --shard-n)") from None
    else:
        tok, ln, dim, block, meta = (_rayworld_tokens(inst, run_dir) if a.env == "rayworld"
                                     else _othello_tokens(inst, a.limit))
        arch, mc = "transformer_l_tokens", {"vocab": dim, "block_size": block}
        source = token_source(tok, ln, block=block, env=a.env, batch_size=a.batch_size,
                              seed=a.seed, device=DEV, limit=a.limit, meta=meta)

    model = build_model(arch, mc)
    extra = ({"replicate": {"of": a.replicate_of, "seed": a.seed, "steps": a.steps}}
             if a.replicate_of else None)
    train(model, source, cfg, run_dir, arch=arch, model_config=mc, device=DEV, resume=a.resume,
          extra_config=extra)


if __name__ == "__main__":
    main()
