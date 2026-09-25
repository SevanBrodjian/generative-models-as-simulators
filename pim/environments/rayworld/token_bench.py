"""Editability of the Rayworld token model, scored like Othello's distribution models.

The legal sets are the frames the unedited and edited worlds render at the edit frame; also
reported is the mean-frame readout (ray-zone Edit Index of the expected frame) and its guard."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pim.editors.grad_steer import build_edit_spec, make_intervention_hook
from pim.editors.pinv import pinv_step, readout_error
from pim.environments.rayworld import bench as rwb
from pim.environments.rayworld.tokens import UNK, FrameVocab, encode
from pim.metrics.edit_index import fidelity_ratio_from
from pim.metrics.zone_editability import edit_index as zone_edit_index
from pim.metrics.zone_editability import zone_rmse
from pim.metrics.set_editability import move_fidelity_ci95, move_fidelity_ratio, move_scorecard

DEV, EF = rwb.DEV, rwb.EF


@dataclass
class TokenBench:
    tokens: np.ndarray        # (n, EF) int64, frames 0..EF-1: the model's context
    pre_tok: np.ndarray       # (n,) the frame the unedited world renders at EF
    post_tok: np.ndarray      # (n,) the frame the edited world renders at EF
    keep: np.ndarray          # (n,) bool: pre != post and both in the vocabulary
    tgt: torch.Tensor         # (n, d_out) the probe target the edit asks for
    change_mask: torch.Tensor  # (n, d_out) bool, the edited object's dims only
    out_dims: list[int]
    zones: object             # ray zones (for the expected-frame bridge)
    vocab: FrameVocab
    n: int
    # as on `bench.Bench`
    kind: str = "regression"
    cells: dict | None = None
    moves: dict | None = None
    selection: dict | None = None

    @property
    def legal_pre(self) -> list[list[int]]:
        return [[int(t)] if k else [] for t, k in zip(self.pre_tok, self.keep)]

    @property
    def legal_post(self) -> list[list[int]]:
        return [[int(t)] if k else [] for t, k in zip(self.post_tok, self.keep)]


selection_path = rwb.selection_path        # one case list per instance, shared with bench.py


def load_token_bench(vocab: FrameVocab, n: int = 192, target: str = "pos",
                     basis_name: str = "cartesian", *,
                     select: "np.ndarray | None" = None, use_selection: bool = True,
                     instance: str | None = None) -> TokenBench:
    """The edit cases as tokens: the context, the two worlds' frames at EF, the targets.
    Case selection as in ``bench.bench_arrays`` (the instance's ``selection.json`` by default,
    which keeps cases whose two worlds differ on at least two rays)."""
    a = rwb.bench_arrays(n, target, basis_name, select=select,
                         use_selection=use_selection, instance=instance)
    tokens = encode(a["obs"][:, :EF], vocab).astype(np.int64)
    post = encode(a["clean"][:, EF], vocab).astype(np.int64)
    pre = encode(a["zones"].gt_unedited, vocab).astype(np.int64)
    keep = (pre != post) & (pre != UNK) & (post != UNK) & (tokens != UNK).all(1)
    tgt = torch.from_numpy(a["y"]).to(DEV)
    tgt = tgt.long() if a["kind"] == "classification" else tgt.float()
    _long = lambda d: (None if d is None                                   # noqa: E731
                       else {k: torch.from_numpy(v).long().to(DEV) for k, v in d.items()})
    cells, moves = _long(a["cells"]), _long(a["moves"])
    return TokenBench(tokens, pre, post, keep, tgt,
                      torch.from_numpy(a["change_mask"]).to(DEV), a["out_dims"], a["zones"],
                      vocab, a["n"], kind=a["kind"], cells=cells, moves=moves, selection=a["selection"])


def frame_probs(outputs: torch.Tensor) -> torch.Tensor:
    """(B, V) head logits to (B, V) next-frame probabilities."""
    return torch.softmax(outputs.float(), -1)


@torch.no_grad()
def probs_at_edit(model, tb: TokenBench, hook=None) -> np.ndarray:
    """(n, V) the model's next-frame distribution after the context, under an edit hook."""
    idx = torch.from_numpy(tb.tokens).to(DEV)
    out = model.decode(idx, edit=hook)
    return frame_probs(out).cpu().numpy()


def expected_frame(probs: np.ndarray, vocab: FrameVocab) -> np.ndarray:
    """(n, R) sum_k p_k * frame_k over the real frames (the UNK mass renormalized away)."""
    p = probs[:, 1:]
    p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
    return p @ vocab.frames[1:]


def scorecard(probs: np.ndarray, tb: TokenBench, uns: np.ndarray | None = None) -> dict:
    """Every number an arm reports: Othello's ``move_scorecard`` on frame sets, plus the
    mean-frame readout."""
    c = move_scorecard(probs, tb.legal_pre, tb.legal_post)
    out = {"edit_index": c["edit_index_union"],           # the frame-set construction
           "edit_index_symdiff": c["edit_index_symdiff"],
           "li_error_vs_post": c["li_error_vs_post"], "li_error_vs_pre": c["li_error_vs_pre"],
           "p_post": c["legal_mass"],
           "p_pre": float(np.mean([probs[i, L].sum() for i, L in enumerate(tb.legal_pre) if L])),
           "n_scored": c["n_scored"],
           "edit_index_per_case": c["edit_index_union_per_case"],
           "zone_edit_index_expected": float(zone_edit_index(
               expected_frame(probs, tb.vocab)[tb.keep], _mask_zones(tb.zones, tb.keep)))}
    # mean-frame guard: whole-frame RMSE to the edited world, relative to the unsteered frame's
    zk = _mask_zones(tb.zones, tb.keep)
    allm = np.ones_like(zk.target)
    out["mean_frame_rmse"] = zone_rmse(expected_frame(probs, tb.vocab)[tb.keep], zk.gt_edited, allm)
    if uns is not None:
        out["fidelity_ratio"] = move_fidelity_ratio(probs, uns, tb.legal_post)
        out.update(move_fidelity_ci95(probs, uns, tb.legal_post))
        out["fidelity_ratio_expected"] = fidelity_ratio_from(
            out["mean_frame_rmse"], zone_rmse(expected_frame(uns, tb.vocab)[tb.keep], zk.gt_edited, allm))
    return out


def _mask_zones(zones, keep):
    """The zones restricted to the kept cases (``edit_index`` reads the per-case masks)."""
    import dataclasses
    kw = {}
    for f in dataclasses.fields(zones):
        v = getattr(zones, f.name)
        kw[f.name] = v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(keep) else v
    return dataclasses.replace(zones, **kw)


@torch.no_grad()
def unsteered(model, tb: TokenBench) -> tuple[np.ndarray, dict]:
    probs = probs_at_edit(model, tb)
    c = scorecard(probs, tb)
    c["fidelity_ratio"] = 1.0
    c["fidelity_ratio_expected"] = 1.0
    return probs, c


@torch.no_grad()
def residuals_last(model, tb: TokenBench) -> dict[int, torch.Tensor]:
    """{point: (n, d)} the residual stream at the LAST context position, every point."""
    rs = model.residual_stack(torch.from_numpy(tb.tokens).to(DEV))
    return {ell: rs[ell][:, -1] for ell in range(len(rs))}


def _write_hook(ell: int, h: torch.Tensor):
    def hook(layer, x):
        if layer != ell:
            return x
        out = x.clone()
        out[:, -1] = h
        return out
    return hook


def pinv_arm(model, tb: TokenBench, probes: dict, alphas, uns: np.ndarray,
             dims: str = "all") -> list[dict]:
    """PI at one residual point, tried at every point, step size swept: ``arms.pinv_arm``
    on tokens."""
    from pim.environments.rayworld.arms import pinv_target, readout_landed

    idx = rwb.dim_idx(dims)
    x0 = residuals_last(model, tb)
    recs = []
    for ell, (probe, _) in probes.items():
        h0 = x0[ell]
        tgt = pinv_target(probe, h0, tb)
        step = pinv_step(h0, tgt, probe, dims=idx)
        if tb.kind == "classification":
            before = {"readout_landed_before": readout_landed(h0, probe, tb)}
            after = lambda h: {"readout_landed": readout_landed(h, probe, tb)}  # noqa: E731
        else:
            before = {"readout_err_before": readout_error(h0, tgt, probe, dims=idx)}
            after = lambda h: {"readout_err_after": readout_error(h, tgt, probe, dims=idx)}  # noqa: E731
        for a in alphas:
            h = h0 + a * step
            probs = probs_at_edit(model, tb, hook=_write_hook(ell, h))
            recs.append({"editor": "PI[zspace]", "point": int(ell), "alpha": float(a),
                         "dims": dims,
                         "write_ratio": float((a * step).norm(dim=1).div(h0.norm(dim=1)).mean()),
                         **before, **after(h),
                         **scorecard(probs, tb, uns)})
    return recs


def grad_steer_arm(model, tb: TokenBench, probes: dict, start_layers, alphas,
                   uns: np.ndarray, n_steps: int = 100, beta: float = 0.2,
                   dims: str = "all") -> list[dict]:
    """GS from each start layer and every point after it: ``arms.grad_steer_arm`` on tokens."""
    rwb.dim_idx(dims)
    cm = tb.change_mask
    x0 = residuals_last(model, tb)
    recs = []
    for ls in start_layers:
        pts = {e: probes[e][0] for e in probes if e >= ls}
        for a in alphas:
            specs = {e: build_edit_spec(pr, x0[e], cm, tb.tgt, beta=beta) for e, pr in pts.items()}
            rec: dict = {}
            hook = make_intervention_hook(pts, specs, ls, alpha=a, n_steps=n_steps, record=rec)
            probs = probs_at_edit(model, tb, hook=hook)
            recs.append({"editor": f"GS@L{ls}", "point": int(ls), "alpha": float(a), "dims": dims,
                         "write_ratio": float(np.mean(
                             [d["delta_norm"] / d["x_norm"] for d in rec.values()
                              if isinstance(d, dict) and d.get("x_norm", 0) > 0] or [np.nan])),
                         **scorecard(probs, tb, uns)})
    return recs


def token_encoder(vocab: FrameVocab):
    """The ``fit_probes(encoder=...)`` hook for a frames-as-tokens model, plus its cache tag."""
    def enc(obs: np.ndarray) -> np.ndarray:
        return encode(obs, vocab)
    tag = f"tokens:V{vocab.size}:{int(vocab.counts.sum())}"
    return enc, tag


@torch.no_grad()
def inverse_arms(model, tbenches: dict, arrays: dict, vocab: FrameVocab, *, basis_name: str,
                 target: str = "full", uns: dict | None = None, **fit_kw) -> tuple[dict, dict]:
    """``arms.inverse_arms`` on the token model. ``arrays`` ({key: bench_arrays(...)}) supplies
    the world state a TokenBench lacks; ``uns`` ({key: unsteered probs}) attaches the guard."""
    from pim.editors.inverse import inverse_overwrite, retrieval_overwrite
    from pim.environments.rayworld.arms import N_OBJ, categorical_target, iter_inverse_maps
    from pim.probes.inverse import encode_categorical_state

    cat = categorical_target(target) is not None
    enc, tag = token_encoder(vocab)
    S, LAB = {}, {}
    for key, tb in tbenches.items():
        a = arrays[key]
        if cat != (a["kind"] == "classification"):
            raise ValueError(f"{key}: a {a['kind']} bench cannot be written through the inverse map of target "
                             f"{target!r} — the map must invert the state the block's own probes read")
        _, s_post = rwb.full_state_pair(a["pos"], a["vel"], a["edit_object"], a["sim"],
                                        "cartesian" if cat else basis_name)
        assert len(s_post) == tb.n, f"{key}: bench_arrays and TokenBench disagree on n"
        S[key] = torch.from_numpy(s_post).to(DEV)
        if cat:
            LAB[key] = torch.from_numpy(np.asarray(a["y"])).long().to(DEV)
    H0 = {key: residuals_last(model, tb) for key, tb in tbenches.items()}
    arms = {key: [] for key in tbenches}
    stats = {"g_r2": [], "g_rmse": [], "nn_r2": [], "g_r2_insample": []}
    for ell, g, bank, st in iter_inverse_maps(model, basis_name=basis_name, target=target, encoder=enc,
                                              encoder_tag=tag, **fit_kw):
        stats["g_r2"].append(float(st["r2"]))
        stats["g_rmse"].append(float(st["rmse"]))
        stats["nn_r2"].append(float(st["nn_r2"]))
        stats["g_r2_insample"].append(float(st.get("r2_insample", float("nan"))))
        for key, tb in tbenches.items():
            h0 = H0[key][ell]
            s_post = (encode_categorical_state(LAB[key], S[key][:, 2 * N_OBJ:], st["n_classes"]) if cat else S[key])
            for editor, h_new in ((("IM", inverse_overwrite(g, s_post)),) if bank is None else
                                  (("IM", inverse_overwrite(g, s_post)),
                                   ("IM-NN", retrieval_overwrite(bank, s_post)))):
                probs = probs_at_edit(model, tb, hook=_write_hook(ell, h_new))
                rec = {"editor": editor, "point": ell, "alpha": 1.0, "dims": "all",
                       "write_ratio": float((h_new - h0).norm(dim=1).div(h0.norm(dim=1)).mean()),
                       "g_r2": float(st["r2"]), **scorecard(probs, tb, None if uns is None else uns.get(key))}
                if editor == "IM-NN":
                    rec["k"] = int(bank.k)
                arms[key].append(rec)
    return arms, stats
