"""The inverse map (IM): g from environment state to the residual at one point, E[h | state].

g mirrors the MLP-128 probe (same body, width, recipe and held-out split by sequence) with
state as input and residual as output; one g per residual point. ``RetrievalBank`` is its
nearest-neighbor control (IM-NN): the mean residual of the k training frames nearest in state.
"""

from __future__ import annotations

import numpy as np
import torch

from pim.probes.base import FIT_BATCH, FIT_EPOCHS, FIT_LR, WorldStateProbe, fit_probe
from pim.probes.mlp import CANONICAL_HIDDEN

INVERSE_HIDDEN = CANONICAL_HIDDEN     # the MLP probe's width
INVERSE_EPOCHS = FIT_EPOCHS
INVERSE_SEED = 0
RETRIEVAL_K = 10
R2_ROWS = 20_000                       # held-out rows used for RetrievalBank.r2


def fit_inverse_map(s_tr, h_tr, s_te, h_te, *, hidden: int = INVERSE_HIDDEN,
                    epochs: int = INVERSE_EPOCHS, seed: int = INVERSE_SEED,
                    device: str = "cuda") -> tuple[WorldStateProbe, dict]:
    """Fit g: state (N, m) -> residual (N, d) with the probe body (standardized-target regression;
    ``forward`` returns raw residuals). Returns the frozen map and its held-out stats (``r2``)."""
    with torch.enable_grad():            # callers may hold no_grad; the fit needs gradients
        g, st = fit_probe(s_tr, h_tr, s_te, h_te, hidden=hidden, epochs=epochs, lr=FIT_LR,
                          batch=FIT_BATCH, device=device, seed=seed, n_classes=None)
    g.eval()
    for p in g.parameters():
        p.requires_grad_(False)
    return g, st


# The categorical inverse map. On a categorical Rayworld target, g's input is the target's labels
# (one-hot per tile) followed by the discs' Cartesian velocity, so writing g(s) sets where the
# discs are without erasing how they move. It is fit on the categorical probes' corpus with the
# residual stack streamed from disk as the target.

CATEGORICAL_STATE = "onehot-labels+cartesian-velocity"      # stored in the cache key and scores.json


def encode_categorical_state(labels: torch.Tensor, extra: torch.Tensor | None, n_classes: int) -> torch.Tensor:
    """(R, n_tiles) labels [+ (R, m) floats] -> (R, n_tiles * n_classes + m): one-hot per tile,
    continuous values appended raw (the map standardizes them)."""
    oh = torch.nn.functional.one_hot(labels.long(), n_classes).reshape(len(labels), -1).float()
    return oh if extra is None else torch.cat([oh, extra.float()], dim=1)


class CategoricalState:
    """Row source for the categorical inverse map's input, with ``baselines.MemmapRows``'s
    ``build(seq, frame)`` surface. Holds ``labels`` (N, T, n_tiles) and ``extra`` (N, T, m)
    and builds the one-hot rows per minibatch."""

    kind = "categorical_state"

    def __init__(self, labels, n_classes: int, extra=None, device="cuda") -> None:
        self.device = torch.device(device)
        self.labels = torch.as_tensor(labels).long().to(self.device)
        self.extra = None if extra is None else torch.as_tensor(extra).float().to(self.device)
        self.n, self.T, self.n_tiles = self.labels.shape
        self.n_classes = int(n_classes)
        self.m = 0 if self.extra is None else int(self.extra.shape[-1])
        self.dim = self.n_tiles * self.n_classes + self.m

    def build(self, seq: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        return encode_categorical_state(self.labels[seq, frame],
                                        None if self.extra is None else self.extra[seq, frame], self.n_classes)

    def moments(self, tr_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Input standardization: one-hot columns stay 0/1 (dividing by a rare class's small sd would
        inflate it); the continuous columns are standardized on the train sequences."""
        xm, xs = torch.zeros(self.dim, device=self.device), torch.ones(self.dim, device=self.device)
        if self.m:
            v = self.extra[tr_seq].reshape(-1, self.m)
            xm[-self.m:], xs[-self.m:] = v.mean(0), v.std(0).clamp_min(1e-6)
        return xm, xs


def fit_inverse_map_stream(state, H, tr_seq, te_seq, *, hidden: int = INVERSE_HIDDEN,
                           epochs: int = INVERSE_EPOCHS, batch: int = FIT_BATCH,
                           seed: int = INVERSE_SEED, log=None) -> tuple[WorldStateProbe, dict]:
    """Streamed ``fit_inverse_map``: input rows from ``state`` (``CategoricalState``), residual targets from
    ``H`` (``baselines.MemmapRows``), minibatches by sequence. Stats: ``r2`` / ``r2_insample`` (against
    the train mean, pooled over residual dimensions) and ``rmse``."""
    from pim.probes.baselines import _moments, _row_index

    torch.manual_seed(seed)
    dev = H.device
    tr_seq = torch.as_tensor(tr_seq, device=dev).sort().values
    te_seq = torch.as_tensor(te_seq, device=dev).sort().values
    s_tr, f_tr = _row_index(tr_seq, H.T, dev)
    s_te, f_te = _row_index(te_seq, H.T, dev)
    ym, ys = _moments(H, s_tr, f_tr)
    xm, xs = state.moments(tr_seq)
    g = WorldStateProbe(state.dim, H.dim, hidden, x_mean=xm, x_std=xs, y_mean=ym, y_std=ys,
                        n_classes=None).to(dev)
    with torch.enable_grad():            # callers may hold no_grad; the fit needs gradients
        opt = torch.optim.Adam(g.parameters(), lr=FIT_LR)
        ys_t = g.y_std.detach()
        bseq = max(1, batch // H.T)
        for ep in range(epochs):
            perm = tr_seq[torch.randperm(len(tr_seq), device=dev)]
            for i in range(0, len(perm), bseq):
                s_b, f_b = _row_index(perm[i:i + bseq], H.T, dev)
                loss = (((g(state.build(s_b, f_b)) - H.build(s_b, f_b)) / ys_t) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            if log and (ep + 1) % 10 == 0:
                log(f"      epoch {ep + 1}/{epochs} loss {float(loss.detach()):.5f}")
    g.eval()
    for p in g.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def _sse(s, f, chunk=8192):
        sse = sst = 0.0
        n = 0
        for i in range(0, len(s), chunk):
            h = H.build(s[i:i + chunk], f[i:i + chunk]).double()
            pr = g(state.build(s[i:i + chunk], f[i:i + chunk])).double()
            sse += float(((pr - h) ** 2).sum())
            sst += float(((h - ym.double()) ** 2).sum())
            n += h.numel()
        return sse, sst, n

    sse_te, sst_te, n_te = _sse(s_te, f_te)
    sse_tr, sst_tr, _ = _sse(s_tr, f_tr)
    if sst_te <= 0:
        raise ValueError("trivial predictor has zero error: the residual is constant on this split")
    return g, {"r2": 1.0 - sse_te / sst_te, "r2_insample": 1.0 - sse_tr / sst_tr,
               "rmse": float(np.sqrt(sse_te / n_te)), "kind": "inverse_map_stream",
               "state": CATEGORICAL_STATE, "d_in": int(state.dim), "rows_train": int(len(s_tr))}


class RetrievalBank:
    """IM-NN: the mean residual of the k nearest training states at one point. ``metric="euclidean"``:
    standardized state distance; ``"onehot"``: Hamming distance over tiles. The residual bank stays
    float32: token-model residuals reach about 1e5, beyond float16's range."""

    def __init__(self, states: torch.Tensor, resid: torch.Tensor, *,
                 metric: str = "euclidean", k: int = RETRIEVAL_K) -> None:
        if metric not in ("euclidean", "onehot"):
            raise ValueError(f"metric must be 'euclidean' or 'onehot', got {metric!r}")
        self.k, self.metric = int(k), metric
        if not torch.isfinite(resid).all():
            raise ValueError("retrieval bank got non-finite residuals")
        self.H = resid.float()
        if metric == "euclidean":
            self.mu = states.float().mean(0)
            self.sd = states.float().std(0).clamp_min(1e-6)
            self.A = (states.float() - self.mu) / self.sd
            self.a2 = (self.A * self.A).sum(1)
        else:
            self.A = states.half()

    @torch.no_grad()
    def mean(self, s_query: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        """(n, d): the mean residual of the k nearest training states to each query."""
        out = torch.zeros(len(s_query), self.H.shape[1], device=self.H.device)
        for i in range(0, len(s_query), chunk):
            q = s_query[i:i + chunk]
            if self.metric == "euclidean":
                Q = (q.float() - self.mu) / self.sd
                d2 = self.a2[None, :] - 2 * Q @ self.A.T + (Q * Q).sum(1)[:, None]
                idx = d2.topk(self.k, dim=1, largest=False).indices
            else:
                idx = (q.half() @ self.A.T).topk(self.k, dim=1, largest=True).indices
            out[i:i + chunk] = self.H[idx].float().mean(1)
        return out

    @torch.no_grad()
    def r2(self, s_te: torch.Tensor, h_te: torch.Tensor, max_rows: int = R2_ROWS) -> float:
        """Held-out R² of the retrieval mean as a predictor of the residual, the same statistic as
        the inverse map's ``r2``. The held-out set is subsampled to ``max_rows`` (seed 0)."""
        from pim.metrics.decodability import r2

        if len(s_te) > max_rows:
            idx = torch.from_numpy(
                np.random.default_rng(0).choice(len(s_te), max_rows, replace=False)
            ).to(s_te.device)
            s_te, h_te = s_te[idx], h_te[idx]
        pred = self.mean(s_te).cpu().numpy()
        return float(r2(pred, h_te.float().cpu().numpy(), self.H.float().mean(0).cpu().numpy()))
