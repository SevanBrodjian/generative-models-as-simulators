#!/usr/bin/env python
"""Write a Rayworld instance's edit selection: the cases every editor is scored on (edits/selection.json).

The rule: the first --n cases of the edits split whose two clean renders at the edit frame differ
on at least --min-rays rays; where the instance has a frame vocabulary, both frames and the whole
context must also be in it, so the frame and token models are scored on the same cases.

    python scripts/make_edit_selection.py --instance 8-ray
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pim.environments import layout  # noqa: E402
from pim.environments.rayworld import bench as rwb  # noqa: E402
from pim.environments.rayworld.bench import EF  # noqa: E402

POOL = {"blink": 6000, "5-ray": 6000}         # the released selections' pools; 4000 elsewhere


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--n", type=int, default=1000, help="cases selected")
    ap.add_argument("--pool", type=int, default=None, help="cases scanned, in split order (default 4000; blink, 5-ray 6000)")
    ap.add_argument("--min-rays", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="overwrite an existing selection")
    a = ap.parse_args()
    a.pool = a.pool or POOL.get(a.instance, 4000)
    out = layout.edits_selection("rayworld", a.instance)
    if out.exists() and not a.force:
        raise SystemExit(f"{out.relative_to(REPO)} exists (--force to overwrite)")

    arr = rwb.bench_arrays(n=a.pool, target="full", basis_name="frustum", instance=a.instance,
                           use_selection=False)
    pre_f, post_f = arr["zones"].gt_unedited, arr["clean"][:, EF]
    nray = (np.abs(pre_f - post_f) > 1e-6).sum(1)
    ok = nray >= a.min_rays
    stats = {"pool": int(len(ok)), "pool_identical_frame": int((nray == 0).sum()),
             "pool_ge_min_rays": int(ok.sum())}
    vocab_p = layout.tokens_dir(a.instance) / "vocab.npz"
    if vocab_p.exists():
        from pim.environments.rayworld.tokens import UNK, FrameVocab, encode

        vocab = FrameVocab.load(vocab_p)
        pre, post = encode(pre_f, vocab).astype(int), encode(post_f, vocab).astype(int)
        tok = encode(arr["obs"][:, :EF], vocab).astype(int)
        in_vocab = (pre != UNK) & (post != UNK) & (tok != UNK).all(1)
        stats["pool_in_vocab"] = int(in_vocab.sum())
        ok &= in_vocab
    sel = np.where(ok)[0][: a.n]
    if len(sel) < a.n:
        raise SystemExit(f"only {len(sel)} valid cases in a pool of {a.pool}; raise --pool")
    i = np.arange(len(pre_f))
    eo = arr["edit_object"]
    tele = np.linalg.norm(arr["pos"][i, EF, eo] - arr["pos"][i, EF - 1, eo], axis=-1)
    res = {"instance": a.instance, "n": int(len(sel)), "pool": a.pool, "min_rays": a.min_rays,
           "vocab": str(vocab_p.relative_to(REPO)) if vocab_p.exists() else None,
           "rule": f"the FIRST {a.n} cases in split order whose two clean renders differ on >= {a.min_rays} rays"
                   + (", both frames and the context in the token vocabulary" if vocab_p.exists() else ""),
           "select": sel.tolist(),
           "stats": {**stats, "scanned": int(sel[-1]) + 1,
                     "teleport_mean_selected": float(tele[sel].mean()),
                     "teleport_mean_pool": float(tele.mean()),
                     "rays_changed_selected": {int(k): int(v) for k, v in
                                               zip(*np.unique(nray[sel], return_counts=True))}}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "select"}, indent=1))
    print(f"-> {out.relative_to(REPO)}  ({len(sel)} cases)")


if __name__ == "__main__":
    main()
