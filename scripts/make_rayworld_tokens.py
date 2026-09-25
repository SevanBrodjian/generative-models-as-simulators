#!/usr/bin/env python
"""Tokenize a Rayworld instance: every distinct frame becomes one token (pim.environments.rayworld.tokens).

Needs the instance's training corpus and its eval/probe/edits splits (they join the vocabulary);
writes datasets/rayworld/<instance>/tokens/ (train.i16, vocab.npz, meta.json).

    python scripts/make_rayworld_tokens.py --instance 8-ray
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pim.environments.layout import instance_root  # noqa: E402
from pim.environments.rayworld.tokens import tokenize_instance  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance", default="8-ray")
    p.add_argument("--chunk", type=int, default=250_000, help="sequences per pass (memory)")
    a = p.parse_args()
    tokenize_instance(instance_root("rayworld", a.instance), chunk=a.chunk)


if __name__ == "__main__":
    main()
