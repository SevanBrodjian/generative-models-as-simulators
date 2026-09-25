"""Transformer-L: the minGPT body of Li et al. with a frame-regression head or a token head.

``TransformerL(obs_res=R)``: frames in through ``Linear(R, d)``, next frame out through ``Linear(d, R)``.
``TransformerL(vocab=V)`` (``TransformerLTokens``): minGPT unchanged, token ids in, next-token logits out.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from pim.environments.othello.vendor.mingpt_model import GPT, GPTConfig
from pim.models.protocol import free_run

_NO_ROLLOUT = "A token model has no rollout; every measurement on it is at the edit step."


class ArchState(NamedTuple):
    """The carried state of a frame model: the observed frames, left-aligned."""

    obs: torch.Tensor  # (B, T, obs_res), T <= block_size


class TransformerL(nn.Module):
    """minGPT with a linear frame input and regression head (``obs_res``) or its token embedding and
    categorical head (``vocab``). Use ``eval()`` for analysis (minGPT has dropout). The carried state
    is left-aligned because minGPT's position embeddings are absolute."""

    def __init__(self, obs_res: int | None = None, block_size: int = 39, n_layer: int = 8,
                 n_head: int = 8, n_embd: int = 512, dropout: float = 0.1, *,
                 vocab: int | None = None) -> None:
        super().__init__()
        if (obs_res is None) == (vocab is None):
            raise ValueError("give exactly one of obs_res (frame model) or vocab (token model)")
        frames = obs_res is not None
        self.input = "linear" if frames else "embedding"
        self.head = "regression" if frames else "categorical"
        self.obs_res, self.vocab = obs_res, vocab

        # Parameter names, registration order and init are part of every probe-cache key:
        # gpt.* is registered first, then encoder.* and decoder.* on a frame model.
        cfg = GPTConfig(vocab_size=vocab if vocab is not None else 1, block_size=block_size,
                        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                        embd_pdrop=dropout, resid_pdrop=dropout, attn_pdrop=dropout)
        self.gpt = GPT(cfg)
        self.cfg = cfg
        self.n_layers = n_layer
        self.probe_layer = n_layer
        if frames:
            # minGPT's unused token layers are replaced, not wrapped, so the state dict names
            # exactly the parameters in use
            self.gpt.tok_emb = nn.Identity()
            self.encoder = nn.Linear(obs_res, n_embd)
            self.gpt.head = nn.Identity()
            # `add_module` refuses the name because the `decoder` property exists
            self._modules["decoder"] = nn.Linear(n_embd, obs_res)

    @property
    def emits(self) -> str:
        """``"frame"`` (regression head) or ``"distribution"`` (categorical head)."""
        return "frame" if self.head == "regression" else "distribution"

    @property
    def norm_out(self) -> nn.Module:
        return self.gpt.ln_f

    @property
    def decoder(self) -> nn.Module:
        """The output head; the prediction is ``decoder(norm_out(h))``."""
        return self._modules["decoder"] if self.head == "regression" else self.gpt.head

    def embed(self, inp: torch.Tensor) -> torch.Tensor:
        """Residual point 0: frames ``(B, T, obs_res)`` through the linear encoder, or ids
        ``(B, T)`` through the token embedding, plus minGPT's position embedding and dropout."""
        t = inp.shape[1]
        x = self.encoder(inp) if self.input == "linear" else self.gpt.tok_emb(inp)
        return self.gpt.drop(x + self.gpt.pos_emb[:, :t, :])

    def _run(self, tokens, edit=None, want_resid=False):
        """The block stack. ``edit`` is ``(layer, vector)``, which overwrites the stream at that
        residual point at the last position, or a callable ``fn(layer, x) -> x`` that fires at
        every residual point 0..n_layers. ``edit=None`` matches ``GPT.forward`` exactly."""
        x = tokens
        hook = edit if callable(edit) else None
        resids = [x] if want_resid else None
        for i, blk in enumerate(self.gpt.blocks):
            if hook is not None:
                x = hook(i, x)
                if want_resid:
                    resids[i] = x
            elif edit is not None and edit[0] == i:
                x = x.clone()
                x[:, -1] = edit[1]
                if want_resid:
                    resids[i] = x
            x = blk(x)
            if want_resid:
                resids.append(x)
        if hook is not None:
            x = hook(self.n_layers, x)
            if want_resid:
                resids[-1] = x
        elif edit is not None and edit[0] == self.n_layers:
            x = x.clone()
            x[:, -1] = edit[1]
            if want_resid:
                resids[-1] = x
        return x, resids

    @torch.no_grad()
    def residual_stack(self, inp: torch.Tensor, edit=None) -> torch.Tensor:
        """(n_layers+1, B, T, n_embd): the stream at every residual point."""
        _, resids = self._run(self.embed(inp), edit=edit, want_resid=True)
        return torch.stack(resids, 0)

    @property
    def state_span(self) -> int:
        """Positions the model attends over: the whole history up to ``block_size``."""
        return self.cfg.block_size

    def forward(self, inp: torch.Tensor, edit=None) -> torch.Tensor:
        """The head at every position: ``(B, T, obs_res)`` next frames or ``(B, T, vocab)`` logits."""
        h, _ = self._run(self.embed(inp), edit=edit)
        return self.decoder(self.norm_out(h))

    def logits(self, inp: torch.Tensor, edit=None) -> torch.Tensor:
        """``forward`` under its categorical name (``ce_next_move`` calls this)."""
        return self.forward(inp, edit=edit)

    def decode(self, state_or_input, edit=None) -> torch.Tensor:
        """The head at the last position: ``(B, obs_res)`` or ``(B, vocab)``, from an
        ``ArchState`` or the raw input tensor.

        Differs from the last position of ``forward`` by about 1e-6 on GPU (matmul kernel choice).
        """
        inp = state_or_input.obs if isinstance(state_or_input, ArchState) else state_or_input
        h, _ = self._run(self.embed(inp), edit=edit)
        return self.decoder(self.norm_out(h[:, -1]))

    def _needs_frames(self) -> None:
        if self.emits != "frame":
            raise NotImplementedError(_NO_ROLLOUT)

    def state_from_obs(self, frames: torch.Tensor) -> ArchState:
        """(B, T, obs_res) frames observed so far -> the carried state."""
        self._needs_frames()
        return ArchState(frames[:, -self.state_span :].contiguous())

    def advance(self, state: ArchState, obs_t: torch.Tensor) -> ArchState:
        self._needs_frames()
        buf = torch.cat([state.obs, obs_t[:, None, :]], dim=1)
        return ArchState(buf[:, -self.state_span :])

    def flat_state(self, state: ArchState) -> torch.Tensor:
        """(B, n_embd): the stream at ``probe_layer``, last position."""
        self._needs_frames()
        _, resids = self._run(self.embed(state.obs), want_resid=True)
        return resids[self.probe_layer][:, -1]

    def decode_with_edit(self, state, layer: int, resid: torch.Tensor) -> torch.Tensor:
        return self.decode(state, edit=(layer, resid))

    def predict_step(self, state: ArchState):
        self._needs_frames()
        pred = self.decode(state)
        return pred, self.advance(state, pred)

    @torch.no_grad()
    def rollout_with_edit(self, state: ArchState, layer: int, resid: torch.Tensor, steps: int):
        """Free run whose first prediction is made with ``resid`` written at ``layer``; every
        later step is recomputed without the edit (``free_run``)."""
        self._needs_frames()
        pred = self.decode_with_edit(state, layer, resid)
        return free_run(self, pred, self.advance(state, pred), steps)


class TransformerLTokens(TransformerL):
    """The token preset: minGPT unchanged, token embedding in, next-token logits out."""

    def __init__(self, vocab: int = 61, block_size: int = 59, n_layer: int = 8,
                 n_head: int = 8, n_embd: int = 512, dropout: float = 0.1) -> None:
        super().__init__(vocab=vocab, block_size=block_size, n_layer=n_layer, n_head=n_head,
                         n_embd=n_embd, dropout=dropout)
