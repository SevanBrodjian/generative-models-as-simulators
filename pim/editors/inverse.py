"""IM: overwrite the residual with the inverse map's E[h | target state]; IM-NN uses the retrieval bank.

The caller supplies the target state in the coordinates g was fit in (the full state before the
dynamics step on Rayworld, the mine / theirs board on Othello). Pairs with ``pim.probes.inverse``.
"""

from __future__ import annotations

import torch

from pim.probes.inverse import RetrievalBank


@torch.no_grad()
def inverse_overwrite(g, s_post: torch.Tensor) -> torch.Tensor:
    """(n, d): the residual written for target state ``s_post``, g(s_post) = E[h | s_post]."""
    return g(s_post)


@torch.no_grad()
def retrieval_overwrite(bank: RetrievalBank, s_post: torch.Tensor) -> torch.Tensor:
    """(n, d): the mean residual of the k training states nearest ``s_post`` (IM-NN)."""
    return bank.mean(s_post)
