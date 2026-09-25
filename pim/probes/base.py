"""The probe body shared by the linear and MLP-128 probes, the residual harvest, and the fit.

The probe shapes follow Li et al. (arXiv:2210.13382): a linear map (their §3.1) or one hidden
layer (§3.2); the two differ only in the middle map ``net``. Both standardize their input and
output inside the module, so a probe is a function of the raw activation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pim.metrics.decodability import r2

# The probe-fit recipe; ``pim.probes.baselines`` and the Othello probe grid read these.
CANONICAL_HIDDEN = 128            # MLP-128: Li et al.'s §3.2 width
FIT_EPOCHS, FIT_LR, FIT_BATCH = 200, 1e-3, 4096


class WorldStateProbe(nn.Module):
    """A linear probe (``hidden=None``) or an MLP with one hidden layer, on the raw activation.

    ``n_classes=None``: regression to ``d_out`` values in raw units. ``n_classes=C``: C-way
    classification of ``d_out`` tiles, ``forward`` returning ``(B, d_out, C)`` logits."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden: int | None = CANONICAL_HIDDEN,
        *,
        x_mean=None,
        x_std=None,
        y_mean=None,
        y_std=None,
        n_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.d_in, self.d_out, self.hidden = d_in, d_out, hidden
        self.n_classes = n_classes
        n_final = d_out if n_classes is None else d_out * n_classes
        if hidden is None:  # linear probe
            self.net = nn.Linear(d_in, n_final)
        else:  # one hidden layer: W1 ReLU(W2 z)
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, n_final)
            )
        z = torch.zeros(d_in)
        o = torch.ones(d_in)
        self.register_buffer("x_mean", z.clone() if x_mean is None else x_mean)
        self.register_buffer("x_std", o.clone() if x_std is None else x_std)
        zo, oo = torch.zeros(d_out), torch.ones(d_out)
        self.register_buffer("y_mean", zo.clone() if y_mean is None else y_mean)
        self.register_buffer("y_std", oo.clone() if y_std is None else y_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, d_in) raw activation -> (B, d_out) values, or (B, d_out, n_classes) logits."""
        z = (x - self.x_mean) / self.x_std
        out = self.net(z)
        if self.n_classes is not None:
            return out.reshape(*out.shape[:-1], self.d_out, self.n_classes)
        return out * self.y_std + self.y_mean

    @property
    def kind(self) -> str:
        base = "linear" if self.hidden is None else f"mlp-{self.hidden}"
        return base if self.n_classes is None else f"{base}-{self.n_classes}way"

    @property
    def act_scale(self) -> float:
        """Median per-dimension std of the activations the probe was fit on.

        Residual points differ in scale by more than an order of magnitude, so GS step sizes
        are relative to this and one ``alpha`` means the same at every point.
        """
        return float(self.x_std.median())



@torch.no_grad()
def collect_residuals(model, obs: np.ndarray, batch: int = 128,
                      memmap: str | Path | None = None,
                      points: "list[int] | None" = None) -> np.ndarray:
    """(n_points, N, T, d_model): the residual stream at every residual point and position.

    ``points`` keeps only these residual points, in the given order. ``memmap`` backs the
    array with a file instead of RAM (a 30k-sequence stack is about 22 GB).
    """
    dev = next(model.parameters()).device
    # preallocated and filled in place, so the peak is one copy of the stack
    out: np.ndarray | None = None
    for i in range(0, len(obs), batch):
        o = torch.from_numpy(obs[i : i + batch]).to(dev)
        # float frames for a frame model, integer ids for a token model
        o = o.float() if np.issubdtype(obs.dtype, np.floating) else o.long()
        tokens = model.embed(o)
        _, resids = model._run(tokens, want_resid=True)
        stack = torch.stack(resids, 0)
        if points is not None:
            stack = stack[list(points)]
        chunk = stack.float().cpu().numpy()
        if out is None:
            shape = (chunk.shape[0], len(obs), *chunk.shape[2:])
            out = (np.lib.format.open_memmap(str(memmap), mode="w+", dtype=np.float32,
                                             shape=shape)
                   if memmap is not None else np.empty(shape, dtype=np.float32))
        out[:, i : i + chunk.shape[1]] = chunk
    if memmap is not None:
        out.flush()
    return out



def fit_probe(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    hidden: int | None = CANONICAL_HIDDEN,
    epochs: int = FIT_EPOCHS,
    lr: float = FIT_LR,
    batch: int = FIT_BATCH,
    device: str = "cuda",
    seed: int = 0,
    n_classes: int | None = None,
) -> tuple[WorldStateProbe, dict]:
    """Fit one probe (``hidden=None``: linear). Returns ``(probe, stats)``. Regression: squared error in
    standardized target space, the linear probe in closed form. Classification (``n_classes``): integer
    labels ``(N, d_out)``, per-tile cross-entropy by Adam, quality as an error rate in percent."""
    torch.manual_seed(seed)
    xm, xs = x_tr.mean(0), x_tr.std(0)
    # Floor the per-dim scale: near-constant dims would make gradient steering through the
    # standardized probe ill-conditioned.
    xs = np.maximum(xs, 1e-2 * np.median(xs)) + 1e-8
    if n_classes is None:
        ym, ys = y_tr.mean(0), y_tr.std(0) + 1e-6
    else:  # logits carry no target affine
        ym = np.zeros(y_tr.shape[1], np.float32)
        ys = np.ones(y_tr.shape[1], np.float32)

    probe = WorldStateProbe(
        x_tr.shape[1],
        y_tr.shape[1],
        hidden,
        x_mean=torch.tensor(xm, dtype=torch.float32),
        x_std=torch.tensor(xs, dtype=torch.float32),
        y_mean=torch.tensor(ym, dtype=torch.float32),
        y_std=torch.tensor(ys, dtype=torch.float32),
        n_classes=n_classes,
    ).to(device)

    if n_classes is not None:
        # per-tile cross-entropy
        xt = torch.tensor(x_tr, dtype=torch.float32, device=device)
        yt = torch.tensor(y_tr, dtype=torch.long, device=device)
        opt = torch.optim.Adam(probe.parameters(), lr=lr)
        n = len(xt)
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, batch):
                idx = perm[i : i + batch]
                logits = probe(xt[idx])  # (b, d_out, C)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, n_classes), yt[idx].reshape(-1)
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
        probe.eval()

        def _pred(x):
            out = []
            with torch.no_grad():
                for i in range(0, len(x), 8192):
                    xb = torch.tensor(x[i : i + 8192], dtype=torch.float32, device=device)
                    out.append(probe(xb).argmax(-1).cpu().numpy())
            return np.concatenate(out, 0)

        hat_te, hat_tr = _pred(x_te), _pred(x_tr)
        per_tile_err = (hat_te != y_te).mean(0)
        stats = {
            "error_rate": float((hat_te != y_te).mean() * 100.0),
            "error_rate_insample": float((hat_tr != y_tr).mean() * 100.0),
            "accuracy": float((hat_te == y_te).mean() * 100.0),
            "per_tile_error_rate": (per_tile_err * 100.0).tolist(),
            # the trivial predictor: the single most common class pooled over all cells of the train split
            "majority_class_error_rate": float(
                (1.0 - np.bincount(y_tr.reshape(-1), minlength=n_classes).max()
                 / y_tr.size) * 100.0
            ),
            "n_train_rows": int(len(x_tr)),
            "n_test_rows": int(len(x_te)),
            "kind": probe.kind,
        }
        return probe, stats

    if hidden is None:
        # closed-form least squares
        zx = (x_tr - xm) / xs
        zy = (y_tr - ym) / ys
        A = np.concatenate([zx, np.ones((len(zx), 1), dtype=np.float32)], 1)
        W, *_ = np.linalg.lstsq(A, zy, rcond=None)
        with torch.no_grad():
            probe.net.weight.copy_(torch.tensor(W[:-1].T, dtype=torch.float32))
            probe.net.bias.copy_(torch.tensor(W[-1], dtype=torch.float32))
    else:
        xt = torch.tensor(x_tr, dtype=torch.float32, device=device)
        yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
        opt = torch.optim.Adam(probe.parameters(), lr=lr)
        n = len(xt)
        # Loss in standardized target space so every output dim counts equally; a raw-units
        # loss would weight position over velocity by the ratio of their variances.
        ys_t = probe.y_std.detach()
        for ep in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, batch):
                idx = perm[i : i + batch]
                loss = (((probe(xt[idx]) - yt[idx]) / ys_t) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

    probe.eval()
    with torch.no_grad():
        pr_te = (
            probe(torch.tensor(x_te, dtype=torch.float32, device=device)).cpu().numpy()
        )
        pr_tr = (
            probe(torch.tensor(x_tr, dtype=torch.float32, device=device)).cpu().numpy()
        )
    stats = {
        "r2": r2(pr_te, y_te, ym),
        "r2_insample": r2(pr_tr, y_tr, ym),
        "rmse": float(np.sqrt(((pr_te - y_te) ** 2).mean())),
        # per-dim R²: NaN where a target dim is constant on the held-out split
        "per_dim_r2": [
            (r2(pr_te[:, [j]], y_te[:, [j]], ym[[j]])
             if float(((y_te[:, j] - ym[j]) ** 2).sum()) > 0 else float("nan"))
            for j in range(y_te.shape[1])
        ],
        "kind": probe.kind,
    }
    return probe, stats

