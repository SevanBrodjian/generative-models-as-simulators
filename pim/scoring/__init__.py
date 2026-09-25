"""The scorer behind ``notebooks/master_eval.ipynb``: wiring only, every number is a call into
``pim.probes``, ``pim.editors``, ``pim.metrics`` or ``pim.environments``.

``runs`` finds the runs, ``blocks`` defines a ``scores.json`` block, ``rayworld`` / ``othello``
score one run, ``baselines`` writes the decodability floors, ``driver`` scores what is missing.
"""
from pim.scoring.baselines import score_all_baselines
from pim.scoring.driver import EVAL_VERSION_BY_ENV, eval_version, score_all
from pim.scoring.runs import scan_runs
from pim.scoring.summary import print_summaries

__all__ = ["scan_runs", "score_all_baselines", "score_all", "print_summaries", "eval_version",
           "EVAL_VERSION_BY_ENV"]
