#!/usr/bin/env python
"""Generate an Othello instance's splits (disjoint index ranges of one seed) and verify them.

Splits: train (20M games), test, probe, probe_large, edits; each lands in datasets/othello/<instance>/.

    python scripts/make_othello_corpus.py --instance adjacent-flip --splits train,test,probe,probe_large,edits
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pim.environments.othello import corpus as oc  # noqa: E402

SPLITS = ("train", "test", "probe", "probe_large", "edits")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance", default="standard", choices=sorted(oc.INSTANCES))
    p.add_argument("--splits", default=",".join(SPLITS), help="comma-separated subset of " + ",".join(SPLITS))
    p.add_argument("--n-train", type=int, default=oc.N_TRAIN_GAMES,
                   help="games in the train split (a smaller value is a prefix of the full split)")
    a = p.parse_args()
    only = tuple(s for s in a.splits.split(",") if s)
    if set(only) - set(SPLITS):
        raise SystemExit(f"unknown split(s) {sorted(set(only) - set(SPLITS))}")
    print(f"{a.instance}: generating {', '.join(only)}", flush=True)
    paths = oc.build(a.n_train, only=only, instance=a.instance)
    oc.verify_splits(paths)
    for name, path in paths.items():
        print(f"  {name}: {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
