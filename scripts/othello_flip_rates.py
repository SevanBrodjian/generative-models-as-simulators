#!/usr/bin/env python
"""Tokens flipped per move in an instance's held-out test games, under the instance's own rules.

Writes runs/_baselines/othello/<instance>/corpus_stats.json. CPU only, about a minute per instance.

    python scripts/othello_flip_rates.py [--instance standard adjacent-flip]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pim.environments.layout import baselines_dir  # noqa: E402
from pim.environments.othello import corpus as oc  # noqa: E402
from pim.environments.othello.counterfactual import flips_per_move  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", nargs="+", default=list(oc.INSTANCES), choices=sorted(oc.INSTANCES),
                    help="default: all four")
    ap.add_argument("--n-games", type=int, default=oc.TEST_N, help="test games replayed")
    a = ap.parse_args()
    for inst in a.instance:
        path = oc.build(log=lambda s: None, only=("test",), instance=inst)["test"]
        tok, ln = oc.load(path)
        tok, ln = tok[: a.n_games], ln[: a.n_games]
        res = {"instance": inst, "split": "test", "rules": oc.rules_of(inst),
               **flips_per_move(tok, ln, oc.rules_of(inst)), "game_length_mean": float(ln.mean())}
        out = baselines_dir("othello", inst) / "corpus_stats.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(res, indent=1) + "\n")
        os.replace(tmp, out)
        print(f"{inst}: {res['flips_per_move']:.4f} tokens flipped per move "
              f"({res['n_games']:,} games) -> {out.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
