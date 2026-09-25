"""Othello held-out gates, probe grid, and editor arms (PI, GS, IM) on the edit bench.

Probes come from ``pim.probes``, editors from ``pim.editors``, scorecards from
``pim.metrics.set_editability``. Every measurement is a single forward pass (no rollout).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pim.editors.grad_steer import build_edit_spec, make_intervention_hook
from pim.editors.pinv import pinv_step, swap_class_logits
from pim.environments.othello.bench import Benchmark
from pim.environments.othello.data import (
    N_CLASSES, N_TILES, T_MODEL, board_probs, canonical_vocab, flatten_rows, move_probs)
from pim.environments.othello.vendor.othello import OthelloBoardState
from pim.metrics.decodability import probe_skill_from_stats
from pim.metrics.set_editability import move_scorecard
from pim.probes.base import CANONICAL_HIDDEN, FIT_BATCH, FIT_EPOCHS, FIT_LR, fit_probe
from pim.probes.cache import ProbeCache

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK = T_MODEL


def _require_cache_dir(cache_dir) -> Path:
    if cache_dir is None:
        raise ValueError("cache_dir is required: every fitted probe is stored in a named directory "
                         "(the run's probes/)")
    return Path(cache_dir)


# ── held-out gates ───────────────────────────────────────────────────────────


def legal_sets(tokens: np.ndarray, lengths: np.ndarray, flip: bool = True,
               placement: str = "enclosure") -> list[list[list[int]]]:
    """Per game, per position, the legal moves (board squares) after that position's move."""
    itos = {v: k for k, v in canonical_vocab().items()}
    out = []
    for row, L in zip(tokens, lengths):
        b = OthelloBoardState(flip=flip, placement=placement)
        per = []
        for t in range(int(L)):
            b.umpire(itos[int(row[t])])
            per.append(sorted(b.get_valid_moves()))
        out.append(per)
    return out


@torch.no_grad()
def gates(model, tokens: np.ndarray, lengths: np.ndarray, batch: int = 512,
          log=print, flip: bool = True, placement: str = "enclosure") -> dict:
    """Held-out next-move quality: legal mass, top-1 legality and accuracy, cross-entropy, and
    the exact Bayes values (``bayes_ce = E[log |legal|]``, ``bayes_top1 = E[1/|legal|]``)."""
    stoi = canonical_vocab()
    legal = legal_sets(tokens, lengths, flip, placement)
    mass, hit1, acc1, ce, bce, btop1, n = 0.0, 0, 0, 0.0, 0.0, 0.0, 0
    for i in range(0, len(tokens), batch):
        tk = torch.from_numpy(tokens[i: i + batch]).long().to(DEV)
        lg = model.logits(tk[:, :BLOCK])
        p = move_probs(lg).cpu().numpy()                   # (B, T, 60), pad output dropped
        am = p.argmax(-1) + 1                              # back into token space
        for r in range(len(tk)):
            L = int(lengths[i + r])
            for t in range(L - 1):                         # position t predicts move t+1
                lm = legal[i + r][t]
                if not lm:
                    continue
                toks = [stoi[s] for s in lm]
                mass += float(p[r, t, [k - 1 for k in toks]].sum())
                hit1 += int(am[r, t] in toks)
                acc1 += int(am[r, t] == int(tokens[i + r, t + 1]))
                ce += -float(np.log(max(p[r, t, int(tokens[i + r, t + 1]) - 1], 1e-12)))
                bce += float(np.log(len(lm)))
                btop1 += 1.0 / len(lm)
                n += 1
        if log and i % (batch * 4) == 0:
            log(f"    gates {i + len(tk):,}/{len(tokens):,}")
    return {"legal_mass": mass / n, "top1_legal": hit1 / n, "top1_acc": acc1 / n,
            "ce": ce / n, "bayes_ce": bce / n, "bayes_top1": btop1 / n,
            "n_positions": n, "n_games": len(tokens)}


# ── probes over residual points ──────────────────────────────────────────────


def _split(n_seq: int, seq_of_row: np.ndarray, how: str, holdout: float, seed: int):
    """(train rows, test rows) with whole games held out (``how="sequence"``)."""
    if how != "sequence":
        raise ValueError(f"only the 'sequence' split is supported, got {how!r}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_seq)
    is_tr = np.zeros(n_seq, bool)
    is_tr[order[: int((1 - holdout) * n_seq)]] = True
    tr_mask = is_tr[seq_of_row]
    return np.where(tr_mask)[0], np.where(~tr_mask)[0]


def observation_probes(data, family: str = "linear", target: str = "mine",
                       holdout: float = 0.2, seed: int = 0, cache_dir=None,
                       cache: bool = True, log=print, epochs: int | None = None,
                       align: str = "left") -> tuple:
    """The observation floor: the probes fitted to the causal move history instead of a model's
    residual stream, with the same games, target, seeded split by game, and padding mask as
    ``fit_probe_grid``. The board is a deterministic function of the moves, so a low value means
    the probe cannot compute the rules from raw moves, not that the information is missing.

    Returns ``(probe, stats)``.
    """
    import torch as _t

    from pim.probes.baselines import CausalHistory, fit_baseline_probe
    from pim.probes.mlp import CANONICAL_HIDDEN

    if target not in ("state", "mine"):
        raise ValueError(f"target must be 'state' or 'mine', got {target!r}")
    store = ProbeCache(_require_cache_dir(cache_dir))
    n_seq = int(len(data.tokens))
    vocab = len(canonical_vocab())
    extra = {} if epochs is None else {"epochs": int(epochs)}
    if align != "left":                     # the cache key carries align only when it is not "left"
        extra["align"] = align
    fname, prov = store.key(None, kind="othello_observation", target=target,
                            family=family, holdout=holdout, seed=seed, n_seq=n_seq,
                            n_rows=int(data.mask.sum()), vocab=vocab, **extra)
    if cache:
        hit = store.load(fname, prov, device=DEV)
        if hit is not None:
            if log:
                log(f"    obs-baseline cache HIT  {fname}")
            return hit
    # the same permutation as _split: identical held-out games
    order = np.random.default_rng(seed).permutation(n_seq)
    cut = int((1 - holdout) * n_seq)
    tr, te = order[:cut], order[cut:]

    y = data.mine if target == "mine" else data.labels
    y_t = _t.from_numpy(y.astype("int64")).to(DEV)
    hist = CausalHistory(_t.from_numpy(data.tokens).to(DEV), kind="one_hot", vocab=vocab, align=align)
    out = fit_baseline_probe(
        hist, y_t, tr, te,
        hidden=None if family == "linear" else CANONICAL_HIDDEN,
        n_classes=3,
        row_mask=_t.from_numpy(data.mask).to(DEV), seed=seed, log=log,
        **{k: v for k, v in extra.items() if k != "align"})
    if log:
        st = out[1]
        log(f"    obs baseline [{target}/{family}]: skill {probe_skill_from_stats(st):+.4f} "
            f"(d_in {st['d_in']})")
    if cache:
        store.store(fname, prov, out)
    return out


@dataclass
class ProbeGrid:
    probes: dict  # (target, family, split, point) -> WorldStateProbe
    stats: list


def fit_probe_grid(model, data, *, targets=("mine",),
                   families=("linear", "mlp"), splits=("sequence",),
                   holdout: float = 0.2, epochs: int = FIT_EPOCHS, batch: int = FIT_BATCH,
                   lr: float = FIT_LR, seed: int = 0, log=print,
                   cache: bool = True, cache_dir=None) -> ProbeGrid:
    """One probe per (target, family, split, residual point), cached under the model
    fingerprint. ``family`` "linear" or "mlp" (MLP-128); ``targets`` "mine" (mine/theirs) or
    "state" (absolute color); whole games held out."""
    from pim.environments.othello.data import harvest_point

    store = ProbeCache(_require_cache_dir(cache_dir))
    n_points = model.n_layers + 1
    fname, prov = store.key(
        model, kind="othello_grid", targets=list(targets), families=list(families),
        splits=list(splits), holdout=holdout, epochs=epochs, batch=batch, lr=lr,
        seed=seed, n_seq=int(len(data.tokens)), n_rows=int(data.mask.sum()),
        n_points=n_points)
    if cache:
        blob = store.load(fname, prov, device=DEV)
        if blob is not None:
            if log:
                log(f"  probe grid cache HIT ({fname})")
            return ProbeGrid(blob["probes"], blob["stats"])

    seq_of_row, _ = flatten_rows(data)
    ys = {t: flatten_rows(data, t)[1].astype(np.int64) for t in targets}
    idx = {s: _split(len(data.tokens), seq_of_row, s, holdout, seed) for s in splits}
    hidden = {"linear": None, "mlp": CANONICAL_HIDDEN}

    probes, stats = {}, []
    for point in range(n_points):
        acts = harvest_point(model, data.tokens, point)
        x = acts[data.mask]
        del acts
        for target in targets:
            y = ys[target]
            for split in splits:
                tr, te = idx[split]
                for fam in families:
                    probe, st = fit_probe(x[tr], y[tr], x[te], y[te],
                                          hidden=hidden[fam], epochs=epochs,
                                          batch=batch, lr=lr, device=DEV, seed=seed,
                                          n_classes=N_CLASSES)
                    st |= {"target": target, "family": fam, "split": split,
                           "point": point}
                    probes[(target, fam, split, point)] = probe
                    stats.append(st)
                    if log:
                        log(f"  point {point}  {target:11s}  {split:8s}  {fam:6s}  "
                            f"skill {probe_skill_from_stats(st):+.4f}")
        del x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if cache:
        store.store(fname, prov, {"probes": probes, "stats": stats})
    return ProbeGrid(probes, stats)


# ── the editor arms (single forward pass, over the bench) ────────────────────


@torch.no_grad()
def unsteered_probs(model, bench: Benchmark) -> np.ndarray:
    """(n_cases, 64) next-move distributions without intervention (the Edit Fidelity reference)."""
    probs = np.zeros((bench.n_cases, N_TILES), np.float32)
    for toks, ids in zip(bench.tokens, bench.case_ids):
        idx = torch.from_numpy(toks).to(DEV)
        probs[ids] = board_probs(model.decode(idx))
    return probs


def unsteered(model, bench: Benchmark) -> dict:
    """The scorecard of the unedited model."""
    return move_scorecard(unsteered_probs(model, bench), bench.legal_pre, bench.legal_post)


@torch.no_grad()
def linear_arm(model, bench: Benchmark, probes: dict, tgt_lab, cur_lab, *,
               alpha: float, points, second=None) -> tuple[np.ndarray, dict]:
    """PI through the classification probes at the residual points in ``points``: ``pinv_step``
    in z-space toward the probe's own read-out with the tile's current and target class scores
    swapped, scaled by ``alpha``.

    ``second``: ``(squares, cur_labels, tgt_labels)``, one per case, a second tile flipped in the
    same write. None is the single-tile edit.
    """
    if any(probes[q].n_classes is None for q in points):
        raise ValueError("linear_arm needs classification probes")
    probs = np.zeros((bench.n_cases, N_TILES), np.float32)
    ratios = []
    for toks, ids in zip(bench.tokens, bench.case_ids):
        idx = torch.from_numpy(toks).to(DEV)
        bsz = len(ids)
        sq = torch.from_numpy(bench.pos_int[ids]).to(DEV)
        td = torch.from_numpy(tgt_lab[ids]).to(DEV)
        cd = torch.from_numpy(cur_lab[ids]).to(DEV)
        if second is not None:
            sq2, cd2, td2 = (torch.from_numpy(np.asarray(a, dtype=np.int64)[ids]).to(DEV) for a in second)
        rec = []

        def hook(layer, x, _rec=rec):
            if layer not in points:
                return x
            p = probes[layer]
            cur = x[:, -1]
            lg = swap_class_logits(p(cur), sq, cd, td)   # (B, N_TILES, N_CLASSES)
            if second is not None:
                lg = swap_class_logits(lg, sq2, cd2, td2)
            delta = alpha * pinv_step(cur, lg.view(bsz, -1), p)
            _rec.append(float((delta.norm(dim=1) / cur.norm(dim=1)).mean()))
            out = x.clone()
            out[:, -1] = cur + delta
            return out

        probs[ids] = board_probs(model.decode(idx, edit=hook))
        ratios.append(np.mean(rec) if rec else 0.0)
    card = move_scorecard(probs, bench.legal_pre, bench.legal_post)
    card["write_ratio"] = float(np.mean(ratios))
    return probs, card


def grad_steer_arm(model, bench: Benchmark, probes: dict, start_layer: int, *,
                   alpha: float, n_steps: int, beta: float,
                   target_labels=None, second=None) -> tuple[np.ndarray, dict]:
    """GS through the classification probes, starting at residual point ``start_layer``.

    ``target_labels`` must be in the frame of the probes: the mine/theirs targets
    (``case_targets(bench)[1]``) for the "mine" probes; None means ``bench.new_class``
    (absolute color, for "state" probes). ``second``: ``(squares, target_labels)``, one per case,
    a second tile the descent must also move. None is the single-tile edit.
    """
    if next(iter(probes.values())).n_classes is None:
        raise ValueError("grad_steer_arm needs classification probes")
    n_points = model.n_layers + 1
    probs = np.zeros((bench.n_cases, N_TILES), np.float32)
    for toks, ids in zip(bench.tokens, bench.case_ids):
        idx = torch.from_numpy(toks).to(DEV)
        bsz = len(ids)
        with torch.no_grad():
            rs = model.residual_stack(idx)
        x0 = {ell: rs[ell][:, -1] for ell in range(n_points)}
        cm = np.zeros((bsz, N_TILES), bool)
        cm[np.arange(bsz), bench.pos_int[ids]] = True
        lab = bench.new_class if target_labels is None else np.asarray(target_labels)
        sq_t = torch.from_numpy(bench.pos_int[ids]).to(DEV)
        tv = torch.zeros(bsz, N_TILES, dtype=torch.long, device=DEV)
        tv[torch.arange(bsz), sq_t] = torch.from_numpy(lab[ids]).to(DEV)
        if second is not None:
            sq2 = np.asarray(second[0], dtype=np.int64)[ids]
            cm[np.arange(bsz), sq2] = True
            tv[torch.arange(bsz), torch.from_numpy(sq2).to(DEV)] = torch.from_numpy(np.asarray(second[1], dtype=np.int64)[ids]).to(DEV)
        specs = {ell: build_edit_spec(probes[ell], x0[ell], cm, tv, beta=beta)
                 for ell in range(n_points)}
        hook = make_intervention_hook(probes, specs, start_layer, alpha=alpha, n_steps=n_steps)
        with torch.no_grad():
            probs[ids] = board_probs(model.decode(idx, edit=hook))
        del rs, x0, specs
    return probs, move_scorecard(probs, bench.legal_pre, bench.legal_post)


# ── IM: the inverse-map editor ───────────────────────────────────────────────


@torch.no_grad()
def inverse_arms(model, bench: Benchmark, data, *, rules: dict, cache_dir, n_games: int,
                 seed: int = 0, k: int | None = None, points=None,
                 uns_probs: np.ndarray | None = None, log=print,
                 return_probs: bool = False, post_boards: np.ndarray | None = None):
    """IM and IM-NN at each residual point in ``points`` (default: all).

    IM writes h' = g(post-edit board) at the last position, where g maps the one-hot
    mine/theirs board (64 x 3) to the residual stream, fitted on ``data``'s games with the same
    seeded split by game as ``fit_probe_grid`` and cached in ``cache_dir``. IM-NN writes the mean
    residual of the k training boards nearest the target board (Hamming). ``rules`` =
    ``corpus.rules_of(instance)``, to replay the bench histories.

    Returns ``(records, {"g_r2", "g_rmse", "nn_r2"})``, plus ``{(editor, point): (n_cases, 64)
    probabilities}`` with ``return_probs``. ``post_boards``: (n_cases, 64) post-edit boards in the
    mover's frame that replace the single-tile flip; ``bench.legal_post`` must describe them.
    """
    from pim.editors.inverse import inverse_overwrite, retrieval_overwrite
    from pim.environments.othello.bench import case_targets
    from pim.environments.othello.data import harvest_point, tokens_and_labels
    from pim.metrics.set_editability import move_fidelity_ci95, move_fidelity_ratio
    from pim.probes.inverse import (INVERSE_EPOCHS, INVERSE_HIDDEN, RETRIEVAL_K, RetrievalBank,
                                    fit_inverse_map)

    store = ProbeCache(_require_cache_dir(cache_dir))
    seq_of_row, states = flatten_rows(data, "mine")                          # (rows,), (rows, 64) in {0,1,2}
    n_seq = int(data.mask.shape[0])
    tr_idx, te_idx = _split(n_seq, seq_of_row, "sequence", 0.2, int(seed))
    onehot = lambda st: np.eye(N_CLASSES, dtype=np.float32)[st].reshape(len(st), -1)   # noqa: E731
    X_all = onehot(states)
    # the bench's pre- and post-edit boards in the mover's frame
    itos = {v: kk for kk, v in canonical_vocab().items()}
    n_cases = bench.n_cases
    hist = [None] * n_cases
    for toks, ids in zip(bench.tokens, bench.case_ids):
        for row, i in zip(toks, ids):
            hist[i] = [itos[int(t)] for t in row]
    bd = tokens_and_labels([hist[i] for i in range(n_cases)], **rules)
    cur_lab, tgt_lab = case_targets(bench)
    s_pre = np.stack([bd.mine[i, len(hist[i]) - 1] for i in range(n_cases)])
    s_post = s_pre.copy()
    s_post[np.arange(n_cases), bench.pos_int] = tgt_lab
    assert (s_pre[np.arange(n_cases), bench.pos_int] == cur_lab).all(), "pre-edit board disagrees with the bench"
    if post_boards is not None:
        post_boards = np.asarray(post_boards)
        assert post_boards.shape == s_post.shape, f"post_boards must be {s_post.shape}, got {post_boards.shape}"
        s_post = post_boards.astype(s_post.dtype)
    Xpost_t = torch.from_numpy(onehot(s_post)).to(DEV)
    X_tr_t = torch.from_numpy(X_all[tr_idx]).to(DEV)
    recs, stats, probs_by = [], {"g_r2": [], "g_rmse": [], "nn_r2": []}, {}
    for ell in (points if points is not None else range(model.n_layers + 1)):
        fname, prov = store.key(model, kind="inverse_map", target="mine-onehot", n_seq=n_seq,
                                split="sequence", seed=int(seed), hidden=INVERSE_HIDDEN,
                                epochs=INVERSE_EPOCHS, point=int(ell), n_games=int(n_games))
        acts = harvest_point(model, data.tokens, ell)
        H = acts[data.mask]
        del acts
        hit = store.load(fname, prov, device=DEV)
        if hit is not None:
            g, st = hit["g"].to(DEV), hit["stats"]
        else:
            g, st = fit_inverse_map(X_all[tr_idx], H[tr_idx], X_all[te_idx], H[te_idx], seed=int(seed), device=DEV)
            store.store(fname, prov, {"g": g, "stats": st})
            if log:
                log(f"    inverse map point {ell}: held-out R² {st['r2']:+.3f}  WROTE {fname}")
        bank = RetrievalBank(X_tr_t, torch.from_numpy(H[tr_idx]).to(DEV), metric="onehot",
                             k=RETRIEVAL_K if k is None else int(k))
        # IM-NN's held-out R² on g's held-out games
        stats["nn_r2"].append(bank.r2(torch.from_numpy(X_all[te_idx]).to(DEV), torch.from_numpy(H[te_idx]).to(DEV)))
        del H
        stats["g_r2"].append(float(st["r2"]))
        stats["g_rmse"].append(float(st["rmse"]))
        for editor, h_new_all in (("IM", inverse_overwrite(g, Xpost_t)),
                                  ("IM-NN", retrieval_overwrite(bank, Xpost_t))):
            probs = np.zeros((n_cases, N_TILES), np.float32)
            ratios = []
            for toks, ids in zip(bench.tokens, bench.case_ids):
                idx = torch.from_numpy(toks).to(DEV)
                h_new = h_new_all[torch.as_tensor(ids, device=DEV)]

                def hook(layer, x, _h=h_new):
                    if layer != ell:
                        return x
                    cur = x[:, -1]
                    ratios.append(float(((_h - cur).norm(dim=1) / cur.norm(dim=1)).mean()))
                    out = x.clone()
                    out[:, -1] = _h
                    return out
                probs[ids] = board_probs(model.decode(idx, edit=hook))
            card = move_scorecard(probs, bench.legal_pre, bench.legal_post)
            rec = {"editor": editor, "point": int(ell), "alpha": 1.0, "g_r2": float(st["r2"]),
                   "write_ratio": float(np.mean(ratios)) if ratios else None,
                   **{kk: v for kk, v in card.items() if isinstance(v, (int, float))}}
            if uns_probs is not None:
                rec["fidelity_ratio"] = move_fidelity_ratio(probs, uns_probs, bench.legal_post)
                rec.update(move_fidelity_ci95(probs, uns_probs, bench.legal_post))
            if editor == "IM-NN":
                rec["k"] = int(bank.k)
            recs.append(rec)
            if return_probs:
                probs_by[(editor, int(ell))] = probs.copy()
        del bank
        torch.cuda.empty_cache()
    return (recs, stats, probs_by) if return_probs else (recs, stats)
