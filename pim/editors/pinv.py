"""PI: pseudoinverse injection, the minimum-norm write that makes a linear probe read the target.

A ``WorldStateProbe`` reads ``y = (A z + b) * y_std + y_mean`` with ``z = (h - x_mean) / x_std``; PI
solves in ``z`` with the output affine included, so the write is minimum-norm in activation SD units.
"""

from __future__ import annotations

import torch

from pim.probes.base import WorldStateProbe


def decompose_hidden(
    h: torch.Tensor, A: torch.Tensor, A_pinv: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split h into its row-space (probe-controlled) and null-space (probe-invariant) parts."""
    Ah = h @ A.T  # (..., d_out)
    h_parallel = Ah @ A_pinv.T  # (..., d_in)
    h_perp = h - h_parallel
    return h_parallel, h_perp


def inject_state(
    h: torch.Tensor,
    target: torch.Tensor,
    A: torch.Tensor,
    A_pinv: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """h' with ``A h' + b = target`` (exactly when A has full row rank), keeping h's null-space part."""
    _, h_perp = decompose_hidden(h, A, A_pinv)
    h_new_parallel = (target - b) @ A_pinv.T
    return h_new_parallel + h_perp


@torch.no_grad()
def pinv_step(h0: torch.Tensor, target: torch.Tensor, probe: WorldStateProbe,
              dims=None) -> torch.Tensor:
    """The alpha = 1 write Δh such that the linear probe reads ``target`` (raw units, or flattened
    ``(B, d_out * n_classes)`` logits) at ``h0 + Δh``. ``dims`` solves on a subset of the outputs
    only (pseudoinverse of the sub-matrix), leaving the other read-outs free."""
    A, b = probe.net.weight.detach(), probe.net.bias.detach()
    idx = None if dims is None else list(dims)
    z0 = (h0 - probe.x_mean) / probe.x_std
    # a classification probe has no output affine: its logit target is already in net units
    tgt_net = (target if probe.n_classes is not None
               else (target - probe.y_mean) / probe.y_std)
    if idx is not None:
        A, b, tgt_net = A[idx], b[idx], tgt_net[..., idx]
    z1 = inject_state(z0, tgt_net, A, torch.linalg.pinv(A), b)
    return (z1 - z0) * probe.x_std


@torch.no_grad()
def swap_class_logits(logits: torch.Tensor, tile: torch.Tensor, cls_a: torch.Tensor,
                      cls_b: torch.Tensor) -> torch.Tensor:
    """The categorical PI target: ``logits`` (B, d_out, C) with the scores of classes ``cls_a`` and
    ``cls_b`` exchanged at ``tile`` (all (B,) long), as a copy. Othello swaps two colors at the edited
    square; a Rayworld categorical target swaps empty and the disc's class at the old and new cells."""
    out = logits.clone()
    ar = torch.arange(out.shape[0], device=out.device)
    sel = out[ar, tile]                                  # (B, C), an indexed copy
    a, b = sel[ar, cls_a].clone(), sel[ar, cls_b].clone()
    sel[ar, cls_a], sel[ar, cls_b] = b, a
    out[ar, tile] = sel
    return out


@torch.no_grad()
def readout_error(h: torch.Tensor, target: torch.Tensor, probe: WorldStateProbe,
                  dims=None) -> float:
    """Mean ‖probe(h) - target‖ in the probe's output units, over ``dims`` if given: did the write land?"""
    err = probe(h) - target
    if dims is not None:
        err = err[..., list(dims)]
    return float(err.norm(dim=-1).mean())
