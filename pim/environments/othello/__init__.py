"""Othello: next-move prediction over synthetic games (Li et al.'s generator, uniform over legal moves).

``vendor/`` is Li et al.'s engine and minGPT (MIT); ``corpus`` holds the four instances and their
splits, ``bench`` the edit cases, ``arms`` the gates, probes and editor arms."""

from pim.environments.othello.bench import (
    Benchmark,
    benchmark_from_cases,
    case_targets,
    load_benchmark,
)
from pim.environments.othello.data import (
    CENTER,
    N_CLASSES,
    N_TILES,
    ProbeData,
    board_probs,
    canonical_vocab,
    flatten_rows,
    harvest_point,
    tokens_and_labels,
)

__all__ = [
    "CENTER",
    "N_CLASSES",
    "N_TILES",
    "ProbeData",
    "board_probs",
    "canonical_vocab",
    "flatten_rows",
    "harvest_point",
    "tokens_and_labels",
    "Benchmark",
    "benchmark_from_cases",
    "case_targets",
    "load_benchmark",
]
