"""Metrics, arrays in and numbers out: Probe Skill (``decodability``), the Edit Index and Edit Fidelity
(``edit_index``, with the frame construction in ``zone_editability`` and the distribution construction
in ``set_editability``), the reported-arm rules (``selection``), seed spread (``replicates``) and
prediction loss against the Bayes floor (``prediction``).
"""

from pim.metrics.decodability import (
    insample_gap_from_stats,
    probe_skill_from_stats,
    probe_skill_regression,
    r2,
)
from pim.metrics.edit_index import FIDELITY_GUARD, edit_index_per_case, fidelity, fidelity_ratio_from, masked_rmse_per_case
from pim.metrics.zone_editability import (
    DIFF_EPS,
    EditZones,
    build_edit_zones,
    edit_index,
    edit_index_by_step,
    edit_scorecard,
    fidelity_ratio,
    sim_config_from,
    zone_rmse,
)
from pim.metrics.set_editability import (
    N_TILES,
    edit_index_legal,
    li_error,
    move_fidelity_ratio,
    move_rmse,
    move_scorecard,
    uniform_over_legal,
)

__all__ = [
    "FIDELITY_GUARD",
    "fidelity",
    # the formulas
    "edit_index_per_case",
    "fidelity_ratio_from",
    "masked_rmse_per_case",
    # decodability
    "probe_skill_regression",
    "probe_skill_from_stats",
    "insample_gap_from_stats",
    "r2",
    # rayworld editability
    "DIFF_EPS",
    "EditZones",
    "build_edit_zones",
    "edit_index",
    "edit_index_by_step",
    "edit_scorecard",
    "fidelity_ratio",
    "sim_config_from",
    "zone_rmse",
    # othello editability
    "N_TILES",
    "edit_index_legal",
    "li_error",
    "move_scorecard",
    "move_rmse",
    "move_fidelity_ratio",
    "uniform_over_legal",
]
