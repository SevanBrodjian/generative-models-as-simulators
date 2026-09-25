"""Rayworld editor arms: probes over residual points, rollouts, and the PI, GS and IM arms.

Wiring over ``pim.probes``, ``pim.editors`` and ``pim.metrics``; the edit cases come from
``bench.py``."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from pim.editors.grad_steer import build_edit_spec, make_intervention_hook
from pim.editors.pinv import pinv_step, readout_error, swap_class_logits
from pim.environments.rayworld.bench import (
    DEV, EF, K_ROLL, N_OBJ, SEED, Bench, _to_basis, dim_idx)
from pim.environments.rayworld.grid_target import categorical_target, snapped_target
from pim.metrics.zone_editability import edit_scorecard, fidelity_ci95, fidelity_ratio
from pim.models.protocol import free_run
from pim.probes.base import FIT_BATCH, FIT_EPOCHS, collect_residuals
from pim.probes.cache import ProbeCache
from pim.probes.linear import fit_linear
from pim.probes.mlp import CANONICAL_HIDDEN, fit_mlp

__all__ = ["fit_probes", "observation_probes", "probe_recipe", "GRID_PROBE_RECIPE",
           "as_activations", "score", "unsteered", "unsteered_rollout", "pinv_rollout",
           "grad_steer_rollout", "pinv_arm", "grad_steer_arm", "iter_inverse_maps",
           "inverse_arms", "collect_residuals", "fidelity_ratio",
           "Bench", "DEV", "EF", "K_ROLL", "N_OBJ", "SEED"]


def _require_cache_dir(cache_dir) -> Path:
    if cache_dir is None:
        raise ValueError("cache_dir is required: every fitted probe is stored in a named "
                         "directory (a run's probes/ directory for scoring)")
    return Path(cache_dir)


def _scratch_dir() -> Path:
    """Where residual stacks are memory-mapped during a fit: ``.scratch/`` at the repository
    root. They reach tens of GB, too large for a RAM-backed system temp directory."""
    d = Path(__file__).resolve().parents[3] / ".scratch"
    d.mkdir(exist_ok=True)
    return d


# ── probes over residual points, cached ──────────────────────────────────────

# the fit recipe of every categorical-target probe; part of their cache keys
GRID_PROBE_RECIPE = {"probe_size": "250k", "n_seq": 200_000, "epochs": 50}


def probe_recipe(target: str, instance, n_seq: int = 30_000, *,
                 cat_n_seq: int = GRID_PROBE_RECIPE["n_seq"],
                 cat_epochs: int = GRID_PROBE_RECIPE["epochs"]) -> dict:
    """``fit_probes`` keyword arguments selecting the probe corpus and fit length for ``target``:
    the 120k corpus at ``n_seq`` for a regression target, the 250k corpus at ``cat_n_seq`` for
    ``cat_epochs`` epochs for a categorical one. ``instance`` is an instance name such as ``"8-ray"``."""
    inst = Path(str(instance)).name
    if categorical_target(target) is not None:
        return {"probe": {"instance": inst, "size": GRID_PROBE_RECIPE["probe_size"]},
                "n_seq": int(cat_n_seq), "epochs": int(cat_epochs)}
    return {"probe": {"instance": inst, "size": "120k"}, "n_seq": n_seq, "epochs": None}


def _probe_instance(probe) -> tuple[str, str]:
    """(instance, size) of a probe corpus; ``probe`` is ``{"instance", "size"}`` or a pair, None
    for the default instance's 120k corpus."""
    from pim.environments import layout

    if probe is not None:
        return (probe["instance"], probe["size"]) if isinstance(probe, dict) else tuple(probe)
    return layout.DEFAULT_INSTANCE["rayworld"], "120k"


def _probe_corpus(probe) -> tuple[Path, Path, dict]:
    """(h5 file, manifest, cache-key fields) of a probe corpus (``probe`` as in ``_probe_instance``)."""
    from pim.environments import layout

    inst, size = _probe_instance(probe)
    data, key_split = layout.probe_key("rayworld", inst, size)
    return (layout.probe_file("rayworld", inst, size), layout.probe_manifest("rayworld", inst, size),
            {"split": key_split, "data": data})


def _corpus_sim(probe) -> dict:
    """The probe corpus's sim config, from its manifest; without the probe corpora, from the eval
    split's manifest (every split of an instance is generated with the same sim)."""
    from pim.environments import layout

    inst, size = _probe_instance(probe)
    manifest = layout.probe_manifest("rayworld", inst, size)
    if not manifest.exists():
        manifest = layout.eval_manifest("rayworld", inst)
    return json.loads(manifest.read_text())["sim"]


def _targets(target: str, pos: np.ndarray, vel: np.ndarray, sim: dict, basis_name: str):
    """(y, n_classes) for a probe target: regression values in the basis (``pos`` / ``full``),
    snapped positions (``pos@<partition>``, frustum basis), or a categorical target's
    (..., tiles) integer labels."""
    grid = categorical_target(target)
    if grid is not None:
        y, _ = grid.label_frames(pos, sim)
        return y, grid.n_classes_on(sim)
    bp, bv = _to_basis(pos, vel, sim, basis_name)
    snap = snapped_target(target)
    if snap is not None:
        if basis_name != "frustum":
            raise ValueError(f"{target} is defined in the frustum basis, got basis {basis_name!r}")
        bp, target = snap.snap(pos, sim).astype(np.float32), snap.base
    y = bp.reshape(*bp.shape[:-2], -1)
    if target == "full":
        y = np.concatenate([y, bv.reshape(*bv.shape[:-2], -1)], axis=-1)
    return y, None


def _read_corpus(h5_path: Path, manifest: Path, n_seq: int):
    with h5py.File(h5_path, "r") as f:
        if len(f["obs_intensity"]) < n_seq:
            raise ValueError(f"{h5_path.name} holds {len(f['obs_intensity']):,} sequences; "
                             f"this fit needs {n_seq:,}")
        obs = f["obs_intensity"][:n_seq].astype(np.float32)
        pos = f["positions"][:n_seq, :, :N_OBJ, :].astype(np.float32)
        vel = f["velocities"][:n_seq, :, :N_OBJ, :].astype(np.float32)
    sim = json.load(open(manifest))["sim"]
    return obs, pos, vel, sim


def fit_probes(model, target: str = "pos", n_seq: int = 30_000,
               family: str = "linear", log=print, basis_name: str = "cartesian",
               cache: bool = True, cache_dir: Path | None = None, encoder=None,
               encoder_tag: str | None = None, epochs: int | None = None,
               require_cached: bool = False, probe: dict | tuple | None = None,
               seed: int = SEED) -> dict:
    """One probe per residual point, ``{point: (probe, stats)}``, held out by sequence (seeded
    80/20 split), cached in ``cache_dir``. ``require_cached`` raises on a miss instead of fitting;
    ``encoder`` maps frames to the model's input (token ids), named by ``encoder_tag``."""
    store = ProbeCache(_require_cache_dir(cache_dir))
    if snapped_target(target) is not None and basis_name != "frustum":
        raise ValueError(f"{target} is defined in the frustum basis, got basis {basis_name!r}")
    h5_path, manifest, keyf = _probe_corpus(probe)
    extra = {} if encoder is None else {"encoder": encoder_tag or "custom"}
    if epochs is not None:
        extra["epochs"] = int(epochs)
    fname, prov = store.key(model, target=target, n_seq=int(n_seq), split=keyf["split"],
                            family=family, basis=basis_name, seed=int(seed),
                            data=keyf["data"], **extra)
    if cache:
        hit = store.load(fname, prov, device=DEV)
        if hit is not None:
            if log:
                log(f"    probe cache HIT  {fname}  ({target}/{family}/{basis_name}/"
                    f"n={n_seq:,})")
            return hit
    if require_cached:
        raise RuntimeError(f"no cached probes for {prov} in {store.dir} — this target's probes "
                           f"are fitted by scripts/fit_probes.py, not by the scorer "
                           f"(require_cached=True)")
    obs, pos, vel, sim = _read_corpus(h5_path, manifest, n_seq)
    y, n_classes = _targets(target, pos, vel, sim, basis_name)
    # Transformer-L has a fixed block size (learned absolute positions): truncate to it
    span = getattr(model, "state_span", obs.shape[1])
    obs = obs[:, : min(obs.shape[1], span)]
    if encoder is not None:
        obs = encoder(obs)                      # e.g. (N, T) token ids
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n_seq)
    tr, te = perm[: int(0.8 * n_seq)], perm[int(0.8 * n_seq):]
    sdir = _scratch_dir()
    out = {}
    if n_classes is not None:
        # classification: one residual point at a time, streamed by sequence block
        from pim.probes.baselines import MemmapRows, fit_probe_stream

        y_t = torch.from_numpy(y[:, : obs.shape[1]]).to(DEV)
        for ell in range(model.n_layers + 1):
            _tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False, dir=sdir)
            _tmp.close()
            try:
                R = collect_residuals(model, obs, batch=64, memmap=_tmp.name, points=[ell])
                p, s = fit_probe_stream(MemmapRows(R[0], device=DEV), y_t, tr, te,
                                        hidden=None if family == "linear" else CANONICAL_HIDDEN,
                                        n_classes=n_classes, seed=int(seed),
                                        epochs=epochs or FIT_EPOCHS, batch=FIT_BATCH, log=None)
                del R
            finally:
                os.unlink(_tmp.name)
            out[ell] = (p, s)
            if log:
                log(f"    point {ell}: err {s['error_rate']:.3f}%  "
                    f"(majority {s['majority_class_error_rate']:.3f}%)")
    else:
        _tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False, dir=sdir)
        _tmp.close()
        try:
            R = collect_residuals(model, obs, batch=64, memmap=_tmp.name)  # (NP, N, T, d)
            y = y[:, : R.shape[2]]
            fit = fit_linear if family == "linear" else fit_mlp
            kw = {} if epochs is None else {"epochs": int(epochs)}
            for ell in range(R.shape[0]):
                X = R[ell]
                p, s = fit(X[tr].reshape(-1, X.shape[-1]), y[tr].reshape(-1, y.shape[-1]),
                           X[te].reshape(-1, X.shape[-1]), y[te].reshape(-1, y.shape[-1]),
                           device=DEV, seed=int(seed), **kw)
                out[ell] = (p, s)
                if log:
                    log(f"    point {ell}: R2 {s['r2']:+.4f}  rmse {s['rmse']:.4f}")
            del R
        finally:
            os.unlink(_tmp.name)          # a failed fit must not leave the stack on disk
    if cache:
        store.store(fname, prov, out)
        if log:
            log(f"    probe cache WROTE {fname}")
    return out


def observation_probes(target: str = "full", n_seq: int = 30_000,
                       family: str = "linear", basis_name: str = "cartesian",
                       span: int = 39, cache_dir=None,
                       cache: bool = True, log=print, epochs: int | None = None,
                       align: str = "left", require_cached: bool = False,
                       probe: dict | tuple | None = None) -> tuple:
    """The observation baseline: ``fit_probes``'s probe on the causal observation history (frames
    0..t zero-padded to ``span``) instead of a residual stream, same corpus and held-out
    sequences. Returns one ``(probe, stats)``."""
    from pim.probes.baselines import CausalHistory, fit_baseline_probe

    store = ProbeCache(_require_cache_dir(cache_dir))
    h5_path, manifest, keyf = _probe_corpus(probe)
    extra = {} if epochs is None else {"epochs": int(epochs)}
    if align != "left":                     # left-aligned keys carry no align field
        extra["align"] = align
    fname, prov = store.key(None, kind="observation", target=target, n_seq=int(n_seq),
                            split=keyf["split"], family=family, basis=basis_name, seed=SEED,
                            span=int(span), data=keyf["data"], **extra)
    if cache:
        hit = store.load(fname, prov, device=DEV)
        if hit is not None:
            if log:
                log(f"    obs-baseline cache HIT  {fname}")
            return hit
    if require_cached:
        raise RuntimeError(f"no cached observation probe for {prov} in {store.dir} "
                           f"(require_cached=True)")
    obs, pos, vel, sim = _read_corpus(h5_path, manifest, n_seq)
    y, n_classes = _targets(target, pos, vel, sim, basis_name)
    obs, y = obs[:, :span], y[:, :span]
    # the same permutation fit_probes draws: identical held-out sequences
    perm = np.random.default_rng(SEED).permutation(n_seq)
    tr, te = perm[: int(0.8 * n_seq)], perm[int(0.8 * n_seq):]
    hist = CausalHistory(torch.from_numpy(obs).to(DEV), align=align)
    y_t = torch.from_numpy(y).to(DEV)
    out = fit_baseline_probe(hist, y_t if n_classes else y_t.float(), tr, te,
                             hidden=None if family == "linear" else CANONICAL_HIDDEN,
                             n_classes=n_classes,
                             seed=SEED, log=log, **{k: v for k, v in extra.items() if k != "align"})
    if log:
        from pim.metrics.decodability import probe_skill_from_stats
        log(f"    obs baseline [{basis_name}/{family}]: skill {probe_skill_from_stats(out[1]):+.4f} "
            f"(d_in {out[1]['d_in']})")
    if cache:
        store.store(fname, prov, out)
    return out


# ── scoring plumbing ─────────────────────────────────────────────────────────


def as_activations(model, ell: int):
    """Point a model's ``flat_state`` at residual point ``ell``."""
    model.probe_layer = ell
    return model


@torch.no_grad()
def score(model, b: Bench, roll: np.ndarray, uns_card: dict | None = None) -> dict:
    """The ray-zone scorecard of a rollout; with ``uns_card`` also the fidelity guard."""
    c = edit_scorecard(roll, b.zones, b.gt_roll)
    if uns_card is not None:
        c["fidelity_ratio"] = fidelity_ratio(c, uns_card)
        c.update(fidelity_ci95(c, uns_card))
    return c


@torch.no_grad()
def _roll_hook(model, state, hook, steps: int = K_ROLL):
    """Free-run whose first step is produced under an edit hook; the rest of the rollout is
    recomputed from the observation window, unedited."""
    pred = model.decode(state, edit=hook)
    return free_run(model, pred, model.advance(state, pred), steps).cpu().numpy()


@torch.no_grad()
def unsteered(model, b: Bench) -> dict:
    """No intervention, through the same rollout path an edit takes."""
    roll = unsteered_rollout(model, b)
    c = score(model, b, roll)
    c["fidelity_ratio"] = 1.0
    return c


# ── rollouts: each arm scores exactly these writes ───────────────────────────


@torch.no_grad()
def unsteered_rollout(model, b: Bench) -> np.ndarray:
    """The no-intervention rollout, through the same path an edit takes."""
    ell = model.n_layers
    as_activations(model, ell)
    return model.rollout_with_edit(b.state, ell, model.flat_state(b.state),
                                   K_ROLL).cpu().numpy()


@torch.no_grad()
def pinv_target(probe, h0: torch.Tensor, b: Bench) -> torch.Tensor:
    """What PI asks the linear probe to read after the write: the bench's targets, or for a
    categorical target the probe's own logits with the edited object's classes swapped at the
    moved tiles, flattened to (B, tiles * classes)."""
    if b.kind != "classification":
        return b.tgt
    if b.moves is not None:
        lg = probe(h0)
        for m in range(b.moves["tile"].shape[1]):
            lg = swap_class_logits(lg, b.moves["tile"][:, m], b.moves["old"][:, m],
                                   b.moves["new"][:, m])
        return lg.reshape(h0.shape[0], -1)
    zero = torch.zeros_like(b.cells["cls"])
    lg = swap_class_logits(probe(h0), b.cells["A"], zero, b.cells["cls"])
    lg = swap_class_logits(lg, b.cells["B"], zero, b.cells["cls"])
    return lg.reshape(h0.shape[0], -1)


@torch.no_grad()
def readout_landed(h: torch.Tensor, probe, b: Bench) -> float:
    """Categorical landing check: the fraction of cases whose probe read-out at ``h`` shows the
    edited state."""
    lab = probe(h).argmax(-1)
    ar = torch.arange(h.shape[0], device=h.device)
    if b.moves is not None:
        ok = (lab.gather(1, b.moves["tile"]) == b.moves["new"]).all(1)
        return float(ok.float().mean())
    ok = (lab[ar, b.cells["A"]] == 0) & (lab[ar, b.cells["B"]] == b.cells["cls"])
    return float(ok.float().mean())


@torch.no_grad()
def pinv_rollout(model, b: Bench, probe, ell: int, alpha: float,
                 dims: str = "all") -> np.ndarray:
    """PI's rollout at one residual point and step size."""
    as_activations(model, ell)
    h0 = model.flat_state(b.state)
    h = h0 + alpha * pinv_step(h0, pinv_target(probe, h0, b), probe, dims=dim_idx(dims))
    return model.rollout_with_edit(b.state, ell, h, K_ROLL).cpu().numpy()


def grad_steer_rollout(model, b: Bench, probes: dict, start_layer: int, alpha: float,
                       n_steps: int = 100, beta: float = 0.2,
                       dims: str = "all", record: dict | None = None) -> np.ndarray:
    """GS's rollout from ``start_layer`` and every residual point after it; ``record`` (a dict)
    receives the hook's per-point diagnostics."""
    dim_idx(dims)
    pts = {e: probes[e][0] for e in probes if e >= start_layer}
    cm = b.change_mask
    specs = {}
    for e, pr in pts.items():
        as_activations(model, e)
        specs[e] = build_edit_spec(pr, model.flat_state(b.state), cm,
                                   b.tgt, beta=beta)
    hook = make_intervention_hook(pts, specs, start_layer, alpha=alpha, n_steps=n_steps,
                                  record=record)
    return _roll_hook(model, b.state, hook)


# ── the PI and GS arms ────────────────────────────────────────────────────────


@torch.no_grad()
def pinv_arm(model, b: Bench, probes: dict, alphas, dims: str = "all") -> list[dict]:
    """PI at one residual point, tried at every point, with the step size swept
    (alpha = 1 is the exact jump)."""
    idx = dim_idx(dims)
    recs = []
    for ell, (probe, _) in probes.items():
        as_activations(model, ell)
        h0 = model.flat_state(b.state)
        tgt = pinv_target(probe, h0, b)
        step = pinv_step(h0, tgt, probe, dims=idx)
        if b.kind == "classification":
            landing = lambda h: {"readout_landed": readout_landed(h, probe, b)}  # noqa: E731
            before = {"readout_landed_before": readout_landed(h0, probe, b)}
        else:
            landing = lambda h: {"readout_err_after": readout_error(h, tgt, probe, dims=idx)}  # noqa: E731
            before = {"readout_err_before": readout_error(h0, tgt, probe, dims=idx)}
        for a in alphas:
            h = h0 + a * step
            roll = pinv_rollout(model, b, probe, ell, a, dims=dims)
            recs.append({"editor": "PI[zspace]", "point": ell, "alpha": float(a),
                         "dims": dims,
                         "write_ratio": float((a * step).norm(dim=1)
                                              .div(h0.norm(dim=1)).mean()),
                         **before, **landing(h),
                         **score(model, b, roll)})
    return recs


def grad_steer_arm(model, b: Bench, probes: dict, start_layers, alphas,
                   n_steps: int = 100, beta: float = 0.2,
                   dims: str = "all") -> list[dict]:
    """GS from each start layer and every point after it, with the step size swept."""
    recs = []
    for ls in start_layers:
        for a in alphas:
            rec: dict = {}
            roll = grad_steer_rollout(model, b, probes, ls, a, n_steps=n_steps, beta=beta,
                                      dims=dims, record=rec)
            recs.append({"editor": f"GS@L{ls}", "point": ls, "alpha": float(a),
                         "dims": dims,
                         "write_ratio": float(np.mean(
                             [d["delta_norm"] / d["x_norm"] for d in rec.values()
                              if isinstance(d, dict) and d.get("x_norm", 0) > 0]
                             or [np.nan])),
                         **score(model, b, roll)})
    return recs


# ── IM: write h' = g(s_post), g the inverse map; IM-NN writes the k-nearest-state mean ──


def iter_inverse_maps(model, *, basis_name: str, target: str = "full", n_seq: int = 30_000,
                      probe: dict | tuple | None = None, cache_dir: Path | None = None,
                      seed: int = SEED, k: int | None = None, points=None, encoder=None,
                      encoder_tag: str | None = None, epochs: int | None = None, bank: bool = True,
                      log=print):
    """Yield ``(point, g, bank, stats)`` per residual point: g cached or fitted on the full state
    in ``basis_name``, the retrieval bank on the same training rows, and g's held-out fit. A
    categorical ``target`` maps its labels plus the Cartesian velocity instead (no bank).
    ``bank=False`` yields no bank (``nn_r2`` NaN), so a cache hit reads no probe corpus."""
    if categorical_target(target) is not None:
        yield from _iter_categorical_inverse_maps(
            model, target=target, basis_name=basis_name, n_seq=n_seq, probe=probe,
            cache_dir=cache_dir, seed=seed, points=points, encoder=encoder,
            encoder_tag=encoder_tag, epochs=epochs, log=log)
        return
    from pim.probes.inverse import (INVERSE_EPOCHS, INVERSE_HIDDEN, RETRIEVAL_K, RetrievalBank,
                                    fit_inverse_map)

    epochs = INVERSE_EPOCHS if epochs is None else int(epochs)
    store = ProbeCache(_require_cache_dir(cache_dir))
    h5_path, manifest, keyf = _probe_corpus(probe)
    data = {}

    def corpus() -> dict:
        """The model's inputs, the flattened states and the train / test row masks, read once."""
        if not data:
            obs, pos, vel, sim = _read_corpus(h5_path, manifest, n_seq)
            y, _ = _targets("full", pos, vel, sim, basis_name)                       # (N, T, 8)
            span = getattr(model, "state_span", obs.shape[1])
            obs = obs[:, : min(obs.shape[1], span)]
            y = y[:, : obs.shape[1]]
            if encoder is not None:
                obs = encoder(obs)
            T = y.shape[1]
            perm = np.random.default_rng(int(seed)).permutation(n_seq)
            tr = np.isin(np.repeat(np.arange(n_seq), T), perm[: int(0.8 * n_seq)])
            data.update(obs=obs, X=y.reshape(-1, y.shape[-1]).astype(np.float32), tr=tr, te=~tr)
        return data

    extra = {} if encoder is None else {"encoder": encoder_tag or "custom"}
    for ell in (points if points is not None else range(model.n_layers + 1)):
        fname, prov = store.key(model, kind="inverse_map", target="full", n_seq=int(n_seq),
                                split=keyf["split"], basis=basis_name, seed=int(seed),
                                data=keyf["data"], hidden=INVERSE_HIDDEN, epochs=epochs,
                                point=int(ell), **extra)
        hit = store.load(fname, prov, device=DEV)
        if bank or hit is None:
            d = corpus()
            X_all, tr, te = d["X"], d["tr"], d["te"]
            _tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False, dir=_scratch_dir())
            _tmp.close()
            try:
                R = collect_residuals(model, d["obs"], batch=64, memmap=_tmp.name, points=[ell])[0]
                H = np.ascontiguousarray(R.reshape(-1, R.shape[-1]))
                del R
            finally:
                os.unlink(_tmp.name)
        if hit is not None:
            g, st = hit["g"].to(DEV), hit["stats"]
            if log:
                log(f"    inverse map cache HIT  {fname}  (point {ell}, R² {st['r2']:+.3f})")
        else:
            g, st = fit_inverse_map(X_all[tr], H[tr], X_all[te], H[te], epochs=epochs, seed=int(seed), device=DEV)
            store.store(fname, prov, {"g": g, "stats": st})
            if log:
                log(f"    inverse map point {ell}: held-out R² {st['r2']:+.3f}  rmse {st['rmse']:.4f}  WROTE {fname}")
        if not bank:
            yield int(ell), g, None, {**st, "nn_r2": float("nan")}
            torch.cuda.empty_cache()
            continue
        rb = RetrievalBank(torch.from_numpy(X_all[tr]).to(DEV), torch.from_numpy(H[tr]).to(DEV),
                           metric="euclidean", k=RETRIEVAL_K if k is None else int(k))
        st = {**st, "nn_r2": rb.r2(torch.from_numpy(X_all[te]).to(DEV), torch.from_numpy(H[te]).to(DEV))}
        del H
        yield int(ell), g, rb, st
        del rb
        torch.cuda.empty_cache()


def _iter_categorical_inverse_maps(model, *, target: str, basis_name: str, n_seq: int,
                                   probe, cache_dir, seed: int, points, encoder,
                                   encoder_tag, epochs, log):
    """The categorical inverse map per point: g maps one-hot labels plus the Cartesian velocity
    to the residual, streamed one point at a time; a cache hit reads neither the probe corpus nor
    residuals. Yields ``(point, g, None, stats)``."""
    from pim.probes.baselines import MemmapRows
    from pim.probes.inverse import (CATEGORICAL_STATE, INVERSE_HIDDEN, CategoricalState,
                                    fit_inverse_map_stream)

    epochs = FIT_EPOCHS if epochs is None else int(epochs)
    store = ProbeCache(_require_cache_dir(cache_dir))
    h5_path, manifest, keyf = _probe_corpus(probe)
    grid, sim = categorical_target(target), _corpus_sim(probe)
    n_classes = grid.n_classes_on(sim)
    d_in = grid.n_tiles_on(sim) * n_classes + 2 * N_OBJ
    data = {}

    def corpus() -> dict:
        """The model's inputs, the train / test sequences and the encoded states, read once."""
        if not data:
            obs, pos, vel, sim = _read_corpus(h5_path, manifest, n_seq)
            labels, n_cls = _targets(target, pos, vel, sim, basis_name)             # (N, T, n_tiles) long
            velc = _targets("full", pos, vel, sim, "cartesian")[0][..., 2 * N_OBJ:]  # (N, T, 2 * N_OBJ)
            span = getattr(model, "state_span", obs.shape[1])
            obs = obs[:, : min(obs.shape[1], span)]
            labels, velc = labels[:, : obs.shape[1]], velc[:, : obs.shape[1]]
            if encoder is not None:
                obs = encoder(obs)
            perm = np.random.default_rng(int(seed)).permutation(n_seq)
            data.update(obs=obs, tr=perm[: int(0.8 * n_seq)], te=perm[int(0.8 * n_seq):],
                        state=CategoricalState(labels, n_cls, extra=velc, device=DEV))
        return data

    extra = {} if encoder is None else {"encoder": encoder_tag or "custom"}
    for ell in (points if points is not None else range(model.n_layers + 1)):
        fname, prov = store.key(model, kind="inverse_map", target=target, state=CATEGORICAL_STATE,
                                n_seq=int(n_seq), split=keyf["split"], basis=basis_name, seed=int(seed),
                                data=keyf["data"], hidden=INVERSE_HIDDEN, epochs=epochs,
                                point=int(ell), **extra)
        hit = store.load(fname, prov, device=DEV)
        if hit is not None:
            g, st = hit["g"].to(DEV), hit["stats"]
            if log:
                log(f"    categorical inverse map cache HIT  {fname}  (point {ell}, R² {st['r2']:+.3f})")
        else:
            d = corpus()
            _tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False, dir=_scratch_dir())
            _tmp.close()
            try:
                R = collect_residuals(model, d["obs"], batch=64, memmap=_tmp.name, points=[ell])
                try:
                    g, st = fit_inverse_map_stream(d["state"], MemmapRows(R[0], device=DEV), d["tr"], d["te"],
                                                   epochs=epochs, seed=int(seed), log=None)
                finally:
                    del R
            finally:
                os.unlink(_tmp.name)
            store.store(fname, prov, {"g": g, "stats": st})
            if log:
                log(f"    categorical inverse map point {ell}: held-out R² {st['r2']:+.3f} "
                    f"(in-sample {st['r2_insample']:+.3f})  rmse {st['rmse']:.4f}  WROTE {fname}")
        if st.get("d_in", d_in) != d_in:
            raise ValueError(f"{fname}: the map reads a {st['d_in']}-dim state, the instance's sim gives {d_in}")
        yield int(ell), g, None, {**st, "nn_r2": float("nan"), "n_classes": int(n_classes)}
        torch.cuda.empty_cache()


@torch.no_grad()
def inverse_arms(model, benches: dict, *, basis_name: str, target: str = "full",
                 unsteered_cards: dict | None = None, **fit_kw) -> tuple[dict, dict]:
    """IM and IM-NN arms for every bench in ``benches`` ({key: Bench}) in one pass over the
    points; returns ``({key: [records]}, {"g_r2": [...], ...})``. A categorical ``target`` needs
    its own categorical benches and scores IM only."""
    from pim.editors.inverse import inverse_overwrite, retrieval_overwrite
    from pim.environments.rayworld.bench import full_state_pair
    from pim.probes.inverse import encode_categorical_state

    cat = categorical_target(target) is not None
    S = {}
    for key, b in benches.items():
        if cat != (b.kind == "classification"):
            raise ValueError(f"{key}: a {b.kind} bench cannot be written through the inverse map of target "
                             f"{target!r} — the map must invert the state the block's own probes read")
        s_pre, s_post = full_state_pair(b.pos, b.vel, b.edit_object, b.sim, "cartesian" if cat else basis_name)
        if b.kind == "regression" and b.tgt.shape[1] == s_post.shape[1]:
            # on the regression full bench the pre-dynamics target is the bench's own target
            assert np.allclose(s_post, b.tgt.cpu().numpy(), atol=1e-4), f"{key}: s_post ≠ Bench.tgt"
        S[key] = (torch.from_numpy(s_pre).to(DEV), torch.from_numpy(s_post).to(DEV))
    arms = {key: [] for key in benches}
    stats = {"g_r2": [], "g_rmse": [], "nn_r2": [], "g_r2_insample": []}
    for ell, g, bank, st in iter_inverse_maps(model, basis_name=basis_name, target=target, **fit_kw):
        stats["g_r2"].append(float(st["r2"]))
        stats["g_rmse"].append(float(st["rmse"]))
        stats["nn_r2"].append(float(st["nn_r2"]))
        stats["g_r2_insample"].append(float(st.get("r2_insample", float("nan"))))
        for key, b in benches.items():
            _, s_post = S[key]
            if cat:    # post-edit labels one-hot, plus the Cartesian velocity (a teleport keeps it)
                s_post = encode_categorical_state(b.tgt, s_post[:, 2 * N_OBJ:], st["n_classes"])
            as_activations(model, ell)
            h0 = model.flat_state(b.state)
            u = None if unsteered_cards is None else unsteered_cards.get(key)
            for editor, h_new in ((("IM", inverse_overwrite(g, s_post)),) if bank is None else
                                  (("IM", inverse_overwrite(g, s_post)),
                                   ("IM-NN", retrieval_overwrite(bank, s_post)))):
                roll = model.rollout_with_edit(b.state, ell, h_new, K_ROLL).cpu().numpy()
                rec = {"editor": editor, "point": ell, "alpha": 1.0, "dims": "all",
                       "write_ratio": float((h_new - h0).norm(dim=1).div(h0.norm(dim=1)).mean()),
                       "g_r2": float(st["r2"]), **score(model, b, roll, u)}
                if editor == "IM-NN":
                    rec["k"] = int(bank.k)
                arms[key].append(rec)
    return arms, stats
