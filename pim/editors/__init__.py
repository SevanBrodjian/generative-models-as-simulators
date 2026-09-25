"""Editors that write a target state into the residual stream: PI (``pinv``), GS (``grad_steer``),
and IM with its retrieval control IM-NN (``inverse``). Editors write; they never score."""

from pim.editors.grad_steer import EditSpec, build_edit_spec, make_intervention_hook
from pim.editors.inverse import inverse_overwrite, retrieval_overwrite
from pim.editors.pinv import (
    decompose_hidden,
    inject_state,
    pinv_step,
    readout_error,
)

__all__ = [
    # PI
    "pinv_step",
    "inject_state",
    "decompose_hidden",
    "readout_error",
    # GS
    "EditSpec",
    "build_edit_spec",
    "make_intervention_hook",
    # IM
    "inverse_overwrite",
    "retrieval_overwrite",
]
