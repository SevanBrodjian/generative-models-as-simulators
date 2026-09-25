#!/usr/bin/env python
"""Synthesize an Othello instance's edit cases: one token recolored after a fixed number of moves.

Cases are cut from the instance's own edits games (generated if absent) by
pim.environments.othello.bench.synthesize_cases, and written to edits/cases_<n>.pkl + .json.

    python scripts/make_othello_edits.py --instance standard-noflip
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pim.environments.othello import corpus as oc  # noqa: E402
from pim.environments.othello.bench import cases_path, synthesize_cases  # noqa: E402
from pim.environments.othello.data import canonical_vocab  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", required=True, choices=sorted(oc.INSTANCES))
    ap.add_argument("--n", type=int, default=1000, help="cases")
    ap.add_argument("--length", type=int, default=20, help="moves in every case's prefix")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.length < 1:
        raise SystemExit("--length must be a positive number of moves")
    t0 = time.time()
    tok, ln = oc.load(oc.build(only=("edits",), instance=a.instance, log=lambda s: None)["edits"])
    itos = {v: k for k, v in canonical_vocab().items()}
    hist = [[int(itos[int(t)]) for t in row[:L]] for row, L in zip(tok, ln)]
    cases, manifest = synthesize_cases(hist, a.n, {a.length: a.n}, seed=a.seed,
                                       **oc.rules_of(a.instance))
    out = cases_path(a.instance)
    if out.name != f"cases_{a.n}.pkl":
        out = out.with_name(f"cases_{a.n}.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(cases, f)
    manifest.update({"instance": a.instance,
                     "source": f"{a.instance} EDITS split ({len(hist)} games, "
                               f"index range [{oc.EDITS_LO}, {oc.EDITS_LO + oc.EDITS_N}))",
                     "prefix_length": a.length,
                     "recipe": "one occupied non-center token recolored; rejected if the legal set is "
                               "unchanged or empty (bench.synthesize_cases)",
                     "minutes": round((time.time() - t0) / 60, 1)})
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {len(cases)} cases -> {out.relative_to(REPO)}  [{manifest['minutes']} min]")


if __name__ == "__main__":
    main()
