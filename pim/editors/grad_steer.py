"""GS: gradient steering through the MLP probe, layer by layer (Li et al. §4.1 and Appendix G).

At the last position of every residual point from a starting layer: ``n_steps`` Adam steps on the
activation at learning rate ``alpha`` times the point's activation scale (median per-dimension SD),
minimizing the probe loss toward the target, with the other read-outs held at weight ``beta``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pim.probes.base import WorldStateProbe


@dataclass
class EditSpec:
    """What the write must achieve, in the probe's output space."""

    values: torch.Tensor  # (B, d_out) new values on changed dims, the pre-edit reading elsewhere
    weight: torch.Tensor  # (B, d_out) 1 on changed dims, beta on held dims
    n_classes: int | None = None

    def loss(self, pred: torch.Tensor) -> torch.Tensor:
        if self.n_classes is None:
            return (self.weight * (pred - self.values) ** 2).sum(1).mean()
        # classification: per-tile cross-entropy against B', weighted, then averaged
        ce = torch.nn.functional.cross_entropy(
            pred.reshape(-1, self.n_classes),
            self.values.reshape(-1).long(),
            reduction="none",
        ).reshape(self.values.shape)
        return (self.weight * ce).mean()


def build_edit_spec(
    probe: WorldStateProbe,
    x0: torch.Tensor,
    change_mask: np.ndarray | torch.Tensor,
    target_values: torch.Tensor,
    *,
    beta: float = 1.0,
) -> EditSpec:
    """The GS objective: ``target_values`` on the dims in ``change_mask`` (both (B, d_out)), the probe's
    pre-edit reading elsewhere, held with weight ``beta``."""
    with torch.no_grad():
        base = probe(x0)  # the pre-edit reading B
        if probe.n_classes is not None:
            base = base.argmax(-1)
    cm = (
        change_mask
        if torch.is_tensor(change_mask)
        else torch.tensor(change_mask, device=x0.device)
    )
    cm = cm.bool()
    values = torch.where(cm, target_values.to(base.dtype), base)
    ones = torch.ones(base.shape, device=base.device, dtype=torch.float32)
    weight = torch.where(cm, ones, torch.full_like(ones, beta))
    return EditSpec(values=values, weight=weight, n_classes=probe.n_classes)


def _descend(
    probe: WorldStateProbe,
    x: torch.Tensor,
    spec: EditSpec,
    lr: float,
    n_steps: int,
) -> torch.Tensor:
    """``n_steps`` Adam steps on the activation at learning rate ``lr``, minimizing ``spec.loss``."""
    v = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        opt = torch.optim.Adam([v], lr=lr)
        for _ in range(n_steps):
            loss = spec.loss(probe(v))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return v.detach()


def make_intervention_hook(
    probes: dict[int, WorldStateProbe],
    specs: dict[int, EditSpec],
    start_layer: int,
    *,
    alpha: float = 0.05,
    n_steps: int = 100,
    record: dict | None = None,
):
    """GS as a ``_run`` hook: ``n_steps`` Adam steps at learning rate ``alpha * probe.act_scale`` on
    the last position at every residual point ``>= start_layer`` that has a probe; earlier points
    pass through. ``record`` collects per-point diagnostics."""

    def hook(layer: int, x: torch.Tensor) -> torch.Tensor:
        if layer < start_layer or layer not in probes:
            return x
        probe, spec = probes[layer], specs[layer]
        cur = x[:, -1]
        # alpha is relative to this point's activation scale (WorldStateProbe.act_scale)
        new = _descend(probe, cur, spec, alpha * probe.act_scale, n_steps)
        if record is not None:
            with torch.no_grad():
                rec = record.setdefault(layer, {})
                # the objective before and after; unlike hit_target it does not saturate
                rec["edit_loss_before"] = float(spec.loss(probe(cur)))
                rec["edit_loss_after"] = float(spec.loss(probe(new)))
                if spec.n_classes is None:
                    rec["readout_err_before"] = float(
                        torch.sqrt(
                            (spec.weight * (probe(cur) - spec.values) ** 2).sum(1).mean()
                        )
                    )
                    rec["readout_err_after"] = float(
                        torch.sqrt(
                            (spec.weight * (probe(new) - spec.values) ** 2).sum(1).mean()
                        )
                    )
                else:
                    # classification diagnostics: tiles read as the requested board B'
                    tgt = spec.values.long()
                    chg = spec.weight >= 1.0
                    for tag, act in (("before", cur), ("after", new)):
                        hat = probe(act).argmax(-1)
                        rec[f"readout_err_{tag}"] = float((hat != tgt).float().mean())
                        rec[f"hit_target_{tag}"] = float(
                            (hat == tgt)[chg].float().mean()
                        )
                        rec[f"hold_rest_{tag}"] = float(
                            (hat == tgt)[~chg].float().mean()
                        )
                rec["delta_norm"] = float((new - cur).norm(dim=1).mean())
                rec["x_norm"] = float(cur.norm(dim=1).mean())
        out = x.clone()
        out[:, -1] = new
        return out

    return hook
