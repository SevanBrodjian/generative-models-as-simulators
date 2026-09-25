"""Decodability baselines: the observation baseline, its streamed probe fit, and the random-init model.

The observation baseline fits the same probes to the causal observation history instead of the
residual stream; rows are built per minibatch because the history is ``T x R`` wide.
"""

from __future__ import annotations

import numpy as np
import torch

from pim.metrics.decodability import r2
from pim.probes.base import FIT_BATCH, FIT_EPOCHS, FIT_LR, WorldStateProbe


class CausalHistory:
    """The causal history ``obs[s, 0..t]`` as one feature row per ``(s, t)``: (N, T, R) frames (``"dense"``)
    or (N, T) ids expanded to R = vocab (``"one_hot"``). ``align="left"``: frame j in block j, later frames
    zeroed; ``"right"``: frame t-k in block k, so fixed-lag reads are linear."""

    def __init__(self, src: torch.Tensor, kind: str = "dense", vocab: int | None = None,
                 align: str = "left"):
        if align not in ("left", "right"):
            raise ValueError(f"align must be left|right, got {align!r}")
        self.src, self.kind, self.align = src, kind, align
        self.device = src.device
        self.n, self.T = src.shape[0], src.shape[1]
        self.R = int(src.shape[2]) if kind == "dense" else int(vocab)
        self.dim = self.T * self.R
        self._ar = torch.arange(self.T, device=src.device)

    def build(self, seq: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        """(B, T*R) features for the given (sequence, frame) rows."""
        if self.align == "right":
            idx = frame[:, None] - self._ar[None, :]                 # source frame per block
            valid = idx >= 0
            g = self.src[seq[:, None].expand_as(idx), idx.clamp_min(0)]
            if self.kind == "dense":
                x = g.float()
            else:
                x = torch.zeros(len(seq), self.T, self.R, device=self.src.device)
                x.scatter_(2, g.unsqueeze(-1).long(), 1.0)
            x = x * valid.unsqueeze(-1).to(x.dtype)
            return x.reshape(len(seq), self.dim)
        if self.kind == "dense":
            x = self.src[seq]                                   # (B, T, R)
        else:
            x = torch.zeros(len(seq), self.T, self.R, device=self.src.device)
            x.scatter_(2, self.src[seq].unsqueeze(-1).long(), 1.0)
        # causal mask: every frame after the probed one is zeroed
        x = x * (self._ar[None, :] <= frame[:, None]).unsqueeze(-1).to(x.dtype)
        return x.reshape(len(seq), self.dim)


class MemmapRows:
    """Rows of an on-disk residual stack ``(N, T, d)`` (``collect_residuals(memmap=..., points=[ell])``),
    served per minibatch for the categorical probes and the categorical inverse map."""

    kind = "memmap"

    def __init__(self, arr, device="cuda"):
        self.mm = arr
        self.n, self.T, self.dim = arr.shape
        self.device = torch.device(device)

    def build(self, seq: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        s, f = seq.cpu().numpy(), frame.cpu().numpy()
        return torch.from_numpy(np.ascontiguousarray(self.mm[s, f])).to(self.device)


def _row_index(seq: torch.Tensor, T: int, device, mask=None):
    """The (sequence, frame) rows to fit on, as two flat index tensors.

    ``mask`` (N, T) drops invalid rows (Othello's padding after a game ends); Rayworld passes None.
    """
    s = seq.repeat_interleave(T)
    f = torch.arange(T, device=device).repeat(len(seq))
    if mask is not None:
        keep = mask[s, f]
        s, f = s[keep], f[keep]
    return s, f


def _moments(hist, s, f, chunk: int = 4096):
    """Streamed train-set mean/std of the features (never materializes the matrix)."""
    tot = torch.zeros(hist.dim, dtype=torch.float64, device=hist.device)
    sq = torch.zeros_like(tot)
    for i in range(0, len(s), chunk):
        x = hist.build(s[i : i + chunk], f[i : i + chunk]).double()
        tot += x.sum(0)
        sq += (x * x).sum(0)
    n = float(len(s))
    mean = tot / n
    var = (sq / n - mean * mean).clamp_min(0)
    return mean.float(), var.sqrt().float().clamp_min(1e-6)


@torch.no_grad()
def _predict(probe, hist, s, f, chunk: int = 8192, classify: bool = False):
    out = []
    for i in range(0, len(s), chunk):
        p = probe(hist.build(s[i : i + chunk], f[i : i + chunk]))
        out.append((p.argmax(-1) if classify else p).cpu().numpy())
    return np.concatenate(out, 0)


def fit_probe_stream(hist, y: torch.Tensor, tr_seq, te_seq, *,
                     hidden: int | None, n_classes: int | None = None,
                     row_mask: torch.Tensor | None = None, seed: int = 0,
                     epochs: int = FIT_EPOCHS, batch: int = FIT_BATCH, log=None):
    """Fit one probe on a streamed row source (``CausalHistory`` or ``MemmapRows``) with ``y`` (N, T, d_out).
    Same probe, standardization, loss, optimizer and stats as ``base.fit_probe``; minibatches are whole
    sequences so a disk-backed source reads contiguously. Returns ``(probe, stats)``."""
    torch.manual_seed(seed)
    dev = hist.device
    tr_seq = torch.as_tensor(tr_seq, device=dev).sort().values   # contiguous reads
    te_seq = torch.as_tensor(te_seq, device=dev).sort().values
    d_out = y.shape[-1]

    s_tr, f_tr = _row_index(tr_seq, hist.T, dev, row_mask)
    s_te, f_te = _row_index(te_seq, hist.T, dev, row_mask)
    n = len(s_tr)

    xm, xs = _moments(hist, s_tr, f_tr)
    if n_classes is None:
        y_tr = y[s_tr, f_tr].float()
        ym, ys = y_tr.mean(0), y_tr.std(0).clamp_min(1e-6)
    else:                                    # classification: the y affine is meaningless
        ym, ys = torch.zeros(d_out, device=dev), torch.ones(d_out, device=dev)

    probe = WorldStateProbe(hist.dim, d_out, hidden, x_mean=xm, x_std=xs,
                            y_mean=ym, y_std=ys, n_classes=n_classes).to(dev)

    if hidden is None and n_classes is None:
        # Closed-form least squares via streamed normal equations: z has zero train mean, so
        # the intercept is 0 and (ZᵀZ) W = Zᵀw gives the minimum-norm lstsq solution.
        ZtZ = torch.zeros(hist.dim, hist.dim, dtype=torch.float64, device=dev)
        Ztw = torch.zeros(hist.dim, d_out, dtype=torch.float64, device=dev)
        for i in range(0, n, 8192):
            sl = slice(i, i + 8192)
            z = (hist.build(s_tr[sl], f_tr[sl]) - xm) / xs
            w = (y[s_tr[sl], f_tr[sl]].float() - ym) / ys
            ZtZ += (z.T @ z).double()        # matmul in fp32, accumulate in fp64
            Ztw += (z.T @ w).double()
        # pinv of the symmetric normal matrix is also minimum-norm when the design is rank
        # deficient (a one-hot history); CUDA's default lstsq driver assumes full rank.
        W = (torch.linalg.pinv(ZtZ, hermitian=True) @ Ztw).float()
        with torch.no_grad():
            probe.net.weight.copy_(W.T)
            probe.net.bias.zero_()
    else:
        opt = torch.optim.Adam(probe.parameters(), lr=FIT_LR)
        ys_t = probe.y_std.detach()
        bseq = max(1, batch // hist.T)                 # sequences per minibatch
        for ep in range(epochs):
            perm = tr_seq[torch.randperm(len(tr_seq), device=dev)]
            for i in range(0, len(perm), bseq):
                s_b, f_b = _row_index(perm[i : i + bseq], hist.T, dev, row_mask)
                if len(s_b) == 0:
                    continue
                x = hist.build(s_b, f_b)
                tgt = y[s_b, f_b]
                if n_classes is None:
                    # loss in standardized target space, as in base.fit_probe
                    loss = (((probe(x) - tgt.float()) / ys_t) ** 2).mean()
                else:
                    loss = torch.nn.functional.cross_entropy(
                        probe(x).reshape(-1, n_classes), tgt.reshape(-1).long())
                opt.zero_grad()
                loss.backward()
                opt.step()
            if log and (ep + 1) % 50 == 0:
                log(f"      epoch {ep + 1}/{epochs} loss {float(loss.detach()):.5f}")
    probe.eval()

    classify = n_classes is not None
    if classify:
        # Streamed error counts (never the rows x tiles label matrices): error rate =
        # mismatches / (rows x tiles), majority class from the train split's class counts.
        def _errors(s, f, chunk=8192):
            n_rows, err, per_tile, counts = 0, 0, None, torch.zeros(n_classes, dtype=torch.long, device=dev)
            for i in range(0, len(s), chunk):
                hat = probe(hist.build(s[i:i + chunk], f[i:i + chunk])).argmax(-1)
                g = y[s[i:i + chunk], f[i:i + chunk]].long()
                miss = (hat != g)
                err += int(miss.sum())
                per_tile = miss.sum(0) if per_tile is None else per_tile + miss.sum(0)
                counts += torch.bincount(g.reshape(-1), minlength=n_classes)
                n_rows += len(g)
            return n_rows, err, per_tile.cpu().numpy(), counts.cpu().numpy()
        n_te, err_te, tile_te, _ = _errors(s_te, f_te)
        n_tr, err_tr, _, cnt_tr = _errors(s_tr, f_tr)
        stats = {
            "error_rate": float(err_te / (n_te * d_out) * 100.0),
            "error_rate_insample": float(err_tr / (n_tr * d_out) * 100.0),
            "accuracy": float(100.0 - err_te / (n_te * d_out) * 100.0),
            "per_tile_error_rate": (tile_te / n_te * 100.0).tolist(),
            # the trivial predictor: the single most common class pooled over all cells of the train split
            "majority_class_error_rate": float((1.0 - cnt_tr.max() / cnt_tr.sum()) * 100.0),
        }
    else:
        hat_te = _predict(probe, hist, s_te, f_te)
        hat_tr = _predict(probe, hist, s_tr, f_tr)
        g_te = y[s_te, f_te].cpu().numpy()
        g_tr = y[s_tr, f_tr].cpu().numpy()
        ymn = ym.cpu().numpy()
        stats = {
            "r2": r2(hat_te, g_te, ymn),
            "r2_insample": r2(hat_tr, g_tr, ymn),
            "rmse": float(np.sqrt(((hat_te - g_te) ** 2).mean())),
            "per_dim_r2": [r2(hat_te[:, [j]], g_te[:, [j]], ymn[[j]])
                           for j in range(d_out)],
        }
    stats.update({"kind": probe.kind, "n_train_rows": int(n),
                  "n_test_rows": int(len(s_te)), "d_in": hist.dim})
    return probe, stats


def random_init_model(arch: str, model_config: dict, seed: int = 0, device: str = "cpu"):
    """An untrained, seeded model of the same architecture: the random-init baseline.

    Seeding makes its fingerprint, and so its probe-cache keys, reproducible.
    """
    from pim.models.registry import build

    torch.manual_seed(seed)
    return build(arch, model_config).to(device).eval()


fit_baseline_probe = fit_probe_stream   # the observation baseline's name for the same fit
