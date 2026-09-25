"""Model registry: architecture name to builder, and the checkpoint loader.

A checkpoint written by ``pim.training`` names its architecture in ``ckpt["arch"]``. A
checkpoint without that key whose state-dict keys all start with ``gpt.`` and include
``gpt.tok_emb.weight`` is a bare minGPT and loads as ``transformer_l_tokens``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from pim.models.transformer_l import TransformerL, TransformerLTokens


def _build_l(cfg: dict):
    keep = {k: cfg[k] for k in ("obs_res", "vocab", "block_size", "n_layer", "n_head",
                                "n_embd", "dropout") if k in cfg}
    return TransformerL(**keep)


def _build_l_tokens(cfg: dict):
    keep = {k: cfg[k] for k in ("vocab", "block_size", "n_layer", "n_head",
                                "n_embd", "dropout") if k in cfg}
    return TransformerLTokens(**keep)


BUILDERS = {
    "transformer_l": _build_l,
    "transformer_l_tokens": _build_l_tokens,
}


def build(arch: str, model_config: dict):
    """Instantiate a registered architecture from its config dict."""
    if arch not in BUILDERS:
        raise KeyError(f"unknown arch {arch!r}; registered: {sorted(BUILDERS)}")
    return BUILDERS[arch](model_config)


@dataclass
class CheckpointInfo:
    arch: str
    val_loss: float
    model_config: dict
    train_config: dict
    run_dir: Path
    ckpt: dict  # the raw checkpoint dict, for fields not modeled here


def _infer_arch(ckpt: dict) -> str:
    if "arch" in ckpt:
        return ckpt["arch"]
    mc = ckpt.get("model_config") or {}
    sd = ckpt.get("model_state") or {}
    if sd and all(k.startswith("gpt.") for k in sd) and "gpt.tok_emb.weight" in sd:
        return "transformer_l_tokens"
    raise ValueError(
        f"cannot identify architecture: no 'arch' key and the state dict is not a bare minGPT "
        f"(model_config keys: {sorted(mc)})"
    )


def load_checkpoint(path: str | Path, device: str = "cpu"):
    """Load a checkpoint as (model in eval mode with ``requires_grad_(False)``, CheckpointInfo).

    Editors that need gradients take them with respect to activations, never weights.
    """
    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = _infer_arch(ckpt)
    mc = dict(ckpt.get("model_config") or {})
    model = build(arch, mc).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, CheckpointInfo(
        arch=arch,
        # intermediate checkpoints carry val_loss=None before the first validation pass
        val_loss=float(ckpt["val_loss"]) if ckpt.get("val_loss") is not None else float("nan"),
        model_config=mc,
        train_config=ckpt.get("train_config", {}),
        run_dir=path.parent,
        ckpt=ckpt,
    )
