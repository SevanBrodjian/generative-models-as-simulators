"""The residual-point count and the free-run loop shared by every rollout.

Residual point ``ell`` is the stream after ``ell`` blocks: point 0 is the embedding and
point ``n_layers`` the final stream before the output LayerNorm, ``n_layers + 1`` points in all.
"""

from __future__ import annotations

import torch


def n_points(model) -> int:
    """Residual points a model exposes: n_layers + 1."""
    return model.n_layers + 1


def free_run(model, pred0: torch.Tensor, state1, steps: int) -> torch.Tensor:
    """(B, steps, out): ``pred0`` followed by ``steps - 1`` unedited ``predict_step`` calls from ``state1``.

    ``pred0`` is the first prediction (edited or not) and ``state1`` the carried state after it,
    so any effect of an edit on later steps has to travel through the predicted frames.
    """
    out, s = [pred0], state1
    for _ in range(steps - 1):
        p, s = model.predict_step(s)
        out.append(p)
    return torch.stack(out, 1)
