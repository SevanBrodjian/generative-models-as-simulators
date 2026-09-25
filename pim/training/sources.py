"""Data sources for the training loop.

``rayworld_source`` streams a flat (N, T, R) frame memmap through ``BlockStream`` (the last 10%
of the pool is validation). ``token_source`` holds a token corpus on the device and samples it
with a seeded generator. Both treat ``limit`` as a prefix of the pool.
"""

from __future__ import annotations

import functools

import numpy as np
import torch
import torch.nn.functional as F

from pim.training.stream import BlockStream
from pim.training.train import DataSource, ce_next_move, mse_next_obs, xy_tokens, IGNORE


def rayworld_source(obs, *, n_total: int, batch_size: int, seed: int,
                    device: str = "cuda", block: int = 2_048,
                    val_fraction: float = 0.1, val_batches: int = 64,
                    limit: int | None = None, meta: dict | None = None) -> DataSource:
    """Stream a flat (N, T, R) float32 memmap; validation is the last ``val_fraction`` of the pool
    (of the first ``limit`` sequences when ``limit`` is set)."""
    n_total = min(limit, n_total) if limit else n_total
    n_val = max(block * 2, int(val_fraction * n_total))
    n_train = n_total - n_val
    if n_train < block:
        raise ValueError(f"{n_total:,} sequences leave {max(n_train, 0):,} for training after {n_val:,} "
                         f"for validation, fewer than one {block:,}-sequence block; use at least "
                         f"{3 * block:,} (--limit)")
    tr = BlockStream(obs, 0, n_train, batch_size, block, seed).batches()
    va_src = BlockStream(obs, n_train, n_total, batch_size, block, seed + 1, shuffle=False)

    def batches():
        while True:
            yield next(tr).to(device, non_blocking=True)

    @torch.no_grad()
    def validate(model) -> float:
        g, tot = va_src.batches(), 0.0
        for _ in range(val_batches):
            x = next(g).to(device, non_blocking=True)
            tot += mse_next_obs(model, x).item()
        return tot / val_batches

    return DataSource(batches=batches(), loss_fn=mse_next_obs, validate=validate,
                      steps_per_epoch=n_train / batch_size,
                      meta={"env": "rayworld", "n_total": n_total, "n_train": n_train,
                            "n_val": n_val, "objective": "mse", **(meta or {})})


def token_source(tok_np: np.ndarray, ln_np: np.ndarray, *, block: int, env: str,
                 batch_size: int, seed: int, device: str = "cuda", val_fraction: float = 0.1,
                 limit: int | None = None, objective: str = "ce",
                 meta: dict | None = None) -> DataSource:
    """A token corpus on the device, split into train / validation by a seeded permutation.

    ``block`` is the model's input length (59 for Othello, T - 1 for the Rayworld token model).
    The objective is padded cross-entropy on the next token (``objective="ce"``).
    """
    losses = {"ce": ce_next_move}
    if objective not in losses:
        raise ValueError(f"unknown objective {objective!r}; one of {sorted(losses)}")
    loss_fn = functools.partial(losses[objective], block=block)
    if limit:
        tok_np, ln_np = tok_np[:limit], ln_np[:limit]
    if not tok_np.flags.writeable:          # a read-only memmap -> one resident copy
        tok_np = np.array(tok_np)
    n = len(tok_np)
    cut = int((1 - val_fraction) * n)
    perm = np.random.default_rng(seed).permutation(n)
    tok = torch.from_numpy(np.ascontiguousarray(tok_np)).to(device)
    ln = torch.from_numpy(np.ascontiguousarray(ln_np)).to(device)
    tr_i = torch.from_numpy(perm[:cut]).to(device)
    va_i = torch.from_numpy(perm[cut:]).to(device)
    gen = torch.Generator(device=device).manual_seed(seed)

    def batches():
        while True:
            idx = tr_i[torch.randint(len(tr_i), (batch_size,), device=device, generator=gen)]
            yield (tok[idx], ln[idx])

    def skip(n: int) -> None:
        """Advance the stream by n batches without gathering them (one generator draw per batch,
        as ``batches`` makes), so a resumed run sees batch n+1 next."""
        for _ in range(n):
            torch.randint(len(tr_i), (batch_size,), device=device, generator=gen)

    @torch.no_grad()
    def validate(model) -> float:
        tot, cnt = 0.0, 0
        for i in range(0, len(va_i), 1024):
            idx = va_i[i: i + 1024]
            x, y = xy_tokens(tok[idx], ln[idx], block)
            lg = model.logits(x)
            m = y != IGNORE
            tot += F.cross_entropy(lg[m], y[m], reduction="sum").item()
            cnt += int(m.sum())
        return tot / max(cnt, 1)

    return DataSource(batches=batches(), loss_fn=loss_fn, validate=validate,
                      steps_per_epoch=len(tr_i) / batch_size,
                      meta={"env": env, "n_total": n, "n_train": int(cut),
                            "n_val": int(n - cut), "objective": objective, "block": block,
                            **(meta or {})},
                      skip=skip)
