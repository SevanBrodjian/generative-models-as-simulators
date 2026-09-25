"""The Bayes floor of next-frame prediction on a Rayworld instance, by posterior sampling.

SMC with Metropolis-Hastings moves over the discs' initial state; a bracket of ``lo`` (posterior
variance of the next frame) and ``hi`` (the sampler's predictive scored on the truth)."""
from __future__ import annotations

import json
import math
import time

import h5py
import numpy as np
import torch

from pim.environments import layout
from pim.environments.rayworld.blink import MARK_VALUE

FLOOR_VERSION = "1.0"
N_FLOOR_SEQ = 1000          # the floor uses the first n eval sequences; so does a run's paired loss
D = torch.float64
MARK = 3                    # observation code of a marker ray (0 background, 1 / 2 the discs)


def unsupported(sim: dict) -> str | None:
    """Why this sampler does not cover an instance, or None if it does."""
    checks = (
        (float(sim.get("obs_noise_std", 0)) == 0, "observation noise"),
        (float(sim.get("position_noise_std", 0)) == 0, "position noise"),
        (float(sim.get("direction_noise_std", 0)) == 0 and float(sim.get("speed_noise_std", 0)) == 0, "velocity noise"),
        (sim.get("soft_shading", "flat") in ("flat", None), "soft shading (continuous ray values)"),
        (int(sim.get("n_observers", 1)) == 1 and sim.get("region", "frustum") == "frustum", "several observers"),
        (sim.get("n_objects") == 2 and bool(sim.get("fixed_reflectivities")), "not two fixed-reflectivity discs"),
        (sim.get("boundary") == "open" and bool(sim.get("always_in_frustum")), "a boundary rule other than open + stay-inside"),
    )
    bad = [why for ok, why in checks if not ok]
    return "; ".join(bad) or None


class World:
    """One instance's geometry, prior and renderer, batched on one device."""

    def __init__(self, sim: dict, n_frames: int, device: str):
        self.dev = device
        g = lambda k: float(sim[k])                                          # noqa: E731
        self.yn, self.yf, self.xn, self.xf = g("y_near"), g("y_far"), g("x_near"), g("x_far")
        self.rad, self.dt = g("radius"), g("dt")
        self.smin, self.smax = g("speed_min"), g("speed_max")
        self.sep = g("collision_margin") * 2.0 * self.rad
        self.drop = bool(sim.get("drop_edge_rays", False))
        self.r_cast = int(sim["obs_res"])
        s = torch.linspace(-1, 1, self.r_cast, dtype=D, device=device) * (self.xf / self.yf)
        nrm = torch.sqrt(s ** 2 + 1)
        self.dx, self.dy = s / nrm, 1 / nrm
        self.frames = torch.arange(n_frames, dtype=D, device=device)

    def xlim(self, y):
        return self.xn + (self.xf - self.xn) * (y - self.yn) / (self.yf - self.yn) - self.rad

    def traj(self, p0, v):
        """(..., 2, 2) initial positions and velocities → (..., T, 2, 2) positions."""
        return p0[..., None, :, :] + self.frames[:, None, None] * self.dt * v[..., None, :, :]

    def accepted(self, tr):
        """``sim.simulate``'s acceptance: every disc fully inside at every frame, and the two
        centers never closer than ``collision_margin * 2r``."""
        x, y = tr[..., 0], tr[..., 1]
        inside = ((y - self.rad >= self.yn) & (y + self.rad <= self.yf) & (x.abs() <= self.xlim(y)))
        sep = (tr[..., 0, :] - tr[..., 1, :]).norm(dim=-1) >= self.sep
        return inside.all(-1).all(-1) & sep.all(-1)

    def render(self, p, vis=None):
        """(..., 2, 2) positions to (..., R) uint8 codes (0 background, 1 disc 0, 2 disc 1), as
        ``renderer.render_frame`` for discs inside the frustum; ``vis`` removes hidden discs."""
        cx, cy = p[..., 0], p[..., 1]
        b = self.dx[:, None] * cx[..., None, :] + self.dy[:, None] * cy[..., None, :]     # (..., R, 2)
        disc = b ** 2 - (cx ** 2 + cy ** 2 - self.rad ** 2)[..., None, :]
        hit = disc >= 0
        if vis is not None:
            hit = hit & vis[..., None, :]
        t = torch.where(hit, b - torch.sqrt(disc.clamp_min(0)), torch.full_like(b, math.inf))
        h0, h1 = hit[..., 0], hit[..., 1]
        code = h1.to(torch.uint8) * 2
        code = torch.where(h0 & (~h1 | (t[..., 0] <= t[..., 1])), torch.ones_like(code), code)
        return code[..., 1:-1] if self.drop else code

    def log_prior(self, p0, v):
        """Log density of the generator's draw, up to a constant: depth uniform then x uniform
        at that depth (1 / half-width), speed and heading uniform (1 / |v| in Cartesian)."""
        return -torch.log(self.xlim(p0[..., 1])).sum(-1) - torch.log(v.norm(dim=-1)).sum(-1)

    def sample_prior(self, n: int, gen: torch.Generator):
        u = lambda *s: torch.rand(*s, dtype=D, device=self.dev, generator=gen)            # noqa: E731
        y = self.yn + self.rad + (self.yf - self.yn - 2 * self.rad) * u(n, 2)
        x = (2 * u(n, 2) - 1) * self.xlim(y)
        sp = self.smin + (self.smax - self.smin) * u(n, 2)
        an = 2 * math.pi * u(n, 2)
        return torch.stack([x, y], -1), torch.stack([sp * an.cos(), sp * an.sin()], -1)


def marker_process(vis: np.ndarray, sim: dict) -> dict:
    """The blink schedule as the observer sees it, mirroring ``blink.blink_schedule``: per frame,
    ``pi`` = P(object j's marker | visibility so far), ``logp_true`` the log probability of the
    markers that occurred, ``entropy`` of the marker configuration; ``vis`` is (S, T, 2)."""
    S, T, _ = vis.shape
    pi = np.zeros((S, T, 2))
    logp, ent = np.zeros((S, T)), np.zeros((S, T))
    p = float(sim.get("blink_prob", 0.0))
    if p <= 0:
        return {"pi": pi, "logp_true": logp, "entropy": ent}
    q, cap, warm = 1.0 / float(sim["blink_mean"]), int(sim["blink_max"]), max(int(sim["blink_warmup"]), 1)
    if (~vis).all(-1).any():
        raise ValueError("both discs hidden on one frame — not a blink_schedule output")
    run = np.zeros((S, 2), int)
    h = lambda *ps: -sum(x * np.log(np.maximum(x, 1e-300)) for x in ps)                   # noqa: E731
    for f in range(T - 1):
        run = (run + 1) * (~vis[:, f])
        m_true = vis[:, f] != vis[:, f + 1]                                               # (S, 2)
        both = vis[:, f].all(-1)
        a = p if f + 1 >= warm else 0.0
        # both visible: object 0 starts w.p. a; object 1 only if 0 did not
        pa = np.stack([np.full(S, a), np.full(S, (1 - a) * a)], -1)
        la = np.select([m_true[:, 0] & ~m_true[:, 1], ~m_true[:, 0] & m_true[:, 1], ~m_true.any(-1)],
                       [np.log(max(a, 1e-300)), np.log(max((1 - a) * a, 1e-300)), 2 * np.log(1 - a)], -np.inf)
        ha = h(np.full(S, a), np.full(S, (1 - a) * a), np.full(S, (1 - a) ** 2))
        # k hidden: its blackout ends w.p. e; only then may the other start one
        k = (~vis[:, f]).argmax(-1)
        e = np.where(run[np.arange(S), k] < cap, q, 1.0)
        pb = np.zeros((S, 2))
        pb[np.arange(S), k], pb[np.arange(S), 1 - k] = e, e * p
        mk, mj = m_true[np.arange(S), k], m_true[np.arange(S), 1 - k]
        lb = np.select([mk & mj, mk & ~mj, ~mk & ~mj],
                       [np.log(e * p), np.log(np.maximum(e * (1 - p), 1e-300)), np.log(np.maximum(1 - e, 1e-300))], -np.inf)
        hb = h(e * p, e * (1 - p), 1 - e)
        pi[:, f] = np.where(both[:, None], pa, pb)
        logp[:, f] = np.where(both, la, lb)
        ent[:, f] = np.where(both, ha, hb)
    if not np.isfinite(logp).all():
        raise ValueError("a marker configuration occurred that this model of blink_schedule gives "
                         "probability 0 — marker_process no longer matches blink_schedule")
    return {"pi": pi, "logp_true": logp, "entropy": ent}


def _load(instance: str, n_seq: int):
    with h5py.File(layout.eval_file("rayworld", instance), "r") as f:
        sim = json.loads(f.attrs["config_json"])["dataset"]["sim"]
        n = min(n_seq, f["obs_intensity"].shape[0])
        obs = f["obs_intensity"][:n].astype(np.float64)
        pos = f["positions"][:n, :, :2, :].astype(np.float64)
        vel = f["velocities"][:n, :, :2, :].astype(np.float64)
        refl = f["reflectivities"][0, :2].astype(np.float64)
        vis = f["blink_visible"][:n].astype(bool) if "blink_visible" in f else np.ones(pos.shape[:3], bool)
    return sim, obs, pos, vel, refl, vis


def trivial_predictors(instance: str, n_seq: int = N_FLOOR_SEQ, alpha: float = 0.5) -> dict:
    """History-blind references on the floor's sequences: the best constant frame (``mse``) and
    pattern distribution (``ce``, tokenizable instances only), both fitted on the probe corpus,
    and repeating the current frame (``persistence_mse``)."""
    from pim.metrics.prediction import mean_se, next_frame_mse

    with h5py.File(layout.eval_file("rayworld", instance), "r") as f:
        ev = f["obs_intensity"][: n_seq].astype(np.float64)
    y = ev[:, 1:]
    R = ev.shape[-1]
    with h5py.File(layout.probe_file("rayworld", instance, "120k"), "r") as f:      # streamed
        d = f["obs_intensity"]
        levels = np.unique(d[:2000].astype(np.float64))
        tokenizable = len(levels) ** R < 2 ** 15 and bool(np.isin(ev, levels).all())
        w = len(levels) ** np.arange(R) if tokenizable else None
        code = (lambda x: (np.searchsorted(levels, x) * w).sum(-1)) if tokenizable else None      # noqa: E731
        tot, n_fr = np.zeros(R), 0
        cnt = np.zeros(len(levels) ** R) if tokenizable else None
        for i in range(0, d.shape[0], 10_000):
            x = d[i: i + 10_000].astype(np.float64).reshape(-1, R)
            tot, n_fr = tot + x.sum(0), n_fr + len(x)
            if tokenizable:
                cnt += np.bincount(code(x), minlength=len(cnt))
    const = tot / n_fr
    m, se = mean_se(next_frame_mse(np.broadcast_to(const, y.shape), y))
    pm_, pse = mean_se(next_frame_mse(ev[:, :-1], y))
    out = {"fit_on": "probe/probe_120k.h5", "n_sequences": int(len(ev)),
           "mse": {"value": m, "se": se, "definition": "the constant frame = the probe corpus's per-ray mean"},
           "persistence_mse": {"value": pm_, "se": pse, "definition": "predict the current frame again"}, "ce": None}
    if tokenizable:
        seen = int((cnt > 0).sum())
        logp = np.log((cnt + alpha) / (cnt.sum() + alpha * (seen + 1)))                   # unseen patterns share one α-cell
        c, cse = mean_se(-logp[code(y)].mean(1))
        out["ce"] = {"value": c, "se": cse, "definition": f"the constant distribution over whole-frame patterns = the probe "
                     f"corpus's frequencies ({seen} patterns, add-{alpha} smoothing)"}
    return out


def observation_codes(obs: np.ndarray, refl: np.ndarray) -> np.ndarray:
    """Float frames to codes 0 / 1 / 2 / MARK; raises on a value that is none of them."""
    levels = np.array([0.0, refl[0], refl[1], MARK_VALUE])
    code = np.abs(obs[..., None] - levels).argmin(-1)
    if np.abs(levels[code] - obs).max() > 1e-5:
        raise ValueError("an observed ray value is not background, a disc's reflectivity or the marker")
    return code.astype(np.uint8)


def _particle_entropy(codes: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """(c, P, R) codes to (c,) plug-in entropy (nats) of the distinct frames among the particles."""
    c, P, R = codes.shape
    w = torch.randint(1, 2 ** 40, (R,), device=codes.device, generator=gen)
    hsh = (codes.long() * w).sum(-1).sort(dim=1).values
    gid = torch.cat([torch.zeros(c, 1, dtype=torch.long, device=codes.device),
                     (hsh[:, 1:] != hsh[:, :-1]).long().cumsum(1)], 1)
    cnt = torch.zeros(c, P, dtype=D, device=codes.device).scatter_add_(1, gid, torch.ones(c, P, dtype=D, device=codes.device))
    pr = cnt / P
    return -(pr * torch.log(pr.clamp_min(1e-300))).sum(1)


def exact_first_position(world: World, n_draws: int, gen: torch.Generator, chunk: int = 500_000) -> dict:
    """The exact floor at t = 0, E[Var(frame 1 | frame 0)], from prior draws grouped by frame-0
    pattern; a check on the sampler. ``usable``: at least half the draws share a pattern."""
    f0, f1 = [], []
    for _ in range(max(1, n_draws // chunk)):
        p0, v = world.sample_prior(chunk, gen)
        tr = world.traj(p0, v)
        keep = world.accepted(tr)
        fr = world.render(tr[keep][:, :2])
        f0.append(fr[:, 0].cpu())
        f1.append(fr[:, 1].cpu())
    f0, f1 = torch.cat(f0), torch.cat(f1)
    w = torch.randint(1, 2 ** 40, (f0.shape[1],), generator=torch.Generator().manual_seed(1))
    _, inv, cnt = torch.unique((f0.long() * w).sum(-1), return_inverse=True, return_counts=True)
    return {"n_accepted": int(len(f0)), "n_patterns": int(len(cnt)),
            "usable": bool((len(f0) - len(cnt)) >= 0.5 * len(f0)), "_inv": inv, "_cnt": cnt, "_f1": f1}


def bayes_floor(instance: str, n_seq: int = N_FLOOR_SEQ, particles: int = 512, sweeps: int = 40,
                init_sweeps: int = 500, device: str | None = None, seed: int = 0,
                exact_draws: int = 20_000_000, max_elements: float = 1.5e8, log=print) -> dict:
    """The instance's floor as brackets for both objectives: ``mse`` (intensity² per ray, the frame
    models') and ``ce`` (nats per frame over whole-frame patterns, the token model's).
    Deterministic in ``seed``."""
    t00 = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sim, obs, pos, vel, refl, vis_np = _load(instance, n_seq)
    why = unsupported(sim)
    if why:
        raise ValueError(f"{instance}: the posterior sampler does not cover {why}")
    S, T, R = obs.shape
    W = World(sim, T, device)
    gen = torch.Generator(device=device).manual_seed(seed)
    levels = torch.tensor([0.0, refl[0], refl[1], MARK_VALUE], dtype=D, device=device)
    code_np = observation_codes(obs, refl)
    mk = marker_process(vis_np, sim)

    # parity: the batched renderer reproduces every stored frame, and the stored x0 is a valid particle
    code_all = torch.from_numpy(code_np).to(device)
    vis_all = torch.from_numpy(vis_np).to(device)
    marked = code_all == MARK
    p0_true, v_true = torch.from_numpy(pos[:, 0]).to(device), torch.from_numpy(vel[:, 0]).to(device)
    tr_true = W.traj(p0_true, v_true)
    same = ((W.render(tr_true, vis_all) == code_all) | marked).all(-1)
    if not bool(same.all()) or not bool(W.accepted(tr_true).all()):
        raise RuntimeError(f"{instance}: renderer / dynamics parity FAILED on {int((~same).sum())} frames "
                           "— the stored initial state must reproduce every stored frame")
    log(f"{instance}: {S} sequences × {T} frames × {R} rays on {device}; parity OK", flush=True)

    lo_exp = -3.0 - math.log10(max(W.r_cast, 10) / 10)          # finer rays → a narrower posterior
    chunk = max(1, int(max_elements / (particles * T * W.r_cast * 2)))
    P = particles
    out = {k: np.zeros((S, T - 1)) for k in ("mse_lo", "mse_hi", "ce_lo", "ce_hi")}
    resets, acc_sum, acc_n, surv_sum = 0, 0.0, 0, 0.0

    for c0 in range(0, S, chunk):
        sl = slice(c0, min(S, c0 + chunk))
        code, vis, msk = code_all[sl], vis_all[sl], marked[sl]
        pt, vt = p0_true[sl], v_true[sl]
        n = code.shape[0]
        pi = torch.from_numpy(mk["pi"][sl]).to(device)                                     # (n, T, 2)

        def valid(p0, v, t):
            sp = v.norm(dim=-1)
            ok = ((sp >= W.smin) & (sp <= W.smax)).all(-1)
            tr = W.traj(p0, v)
            ok &= W.accepted(tr)
            fr = W.render(tr[..., : t + 1, :, :], vis[:, None, : t + 1])
            return ok & ((fr == code[:, None, : t + 1]) | msk[:, None, : t + 1]).all(-1).all(-1)

        def sweep(p0, v, t):
            """One MH move per particle: one disc, a log-uniform scale, either a position move or
            a velocity move pivoting about a random observed frame (that frame's position kept)."""
            shp = (n, P)
            r = lambda *s: torch.rand(*s, dtype=D, device=device, generator=gen)           # noqa: E731
            j = (r(*shp) < 0.5).long()
            sc = 10 ** (lo_exp - lo_exp * r(*shp))
            kind = r(*shp) < 0.5
            k = (r(*shp) * (t + 1)).floor()
            z = torch.randn(*shp, 2, 2, dtype=D, device=device, generator=gen)
            dv = z[..., 0, :] * (sc * 0.07)[..., None] * (~kind)[..., None]
            dp = z[..., 1, :] * (sc * 2.0)[..., None] * kind[..., None] - (k * W.dt)[..., None] * dv
            oh = torch.nn.functional.one_hot(j, 2).to(D)[..., None]
            p0n, vn = p0 + oh * dp[..., None, :], v + oh * dv[..., None, :]
            lr = (W.log_prior(p0n, vn) - W.log_prior(p0, v)).nan_to_num(nan=-math.inf)
            a = valid(p0n, vn, t) & (torch.log(r(*shp)) < lr)
            return torch.where(a[..., None, None], p0n, p0), torch.where(a[..., None, None], vn, v), float(a.double().mean())

        p0, v = pt[:, None].repeat(1, P, 1, 1), vt[:, None].repeat(1, P, 1, 1)
        for _ in range(init_sweeps):
            p0, v, _a = sweep(p0, v, 0)
        for t in range(T - 1):
            nxt = W.render(W.traj(p0, v)[..., t + 1, :, :], vis[:, None, t + 1])           # (n, P, R)
            y = levels[nxt.long()]
            mu, var_y = y.mean(1), y.var(1, unbiased=True)
            pr = torch.zeros(n, R, dtype=D, device=device)
            pr[:, 0], pr[:, R - 1] = pi[:, t + 1, 0], pi[:, t + 1, 1]
            truth = levels[code[:, t + 1].long()]
            out["mse_lo"][sl, t] = ((1 - pr) * var_y + pr * (1 - pr) * (MARK_VALUE - mu) ** 2).mean(-1).cpu().numpy()
            out["mse_hi"][sl, t] = ((pr * MARK_VALUE + (1 - pr) * mu - truth) ** 2).mean(-1).cpu().numpy()
            alive = ((nxt == code[:, None, t + 1]) | msk[:, None, t + 1]).all(-1)          # (n, P)
            k_alive = alive.sum(1).to(D)
            # H(render) + H(markers) slightly overcounts where a marker hides a ray's render
            out["ce_lo"][sl, t] = _particle_entropy(nxt, gen).cpu().numpy() + mk["entropy"][sl, t + 1]
            out["ce_hi"][sl, t] = (-torch.log((k_alive + 0.5) / (P + 1))).cpu().numpy() - mk["logp_true"][sl, t + 1]
            surv_sum += float(alive.double().mean()) * n
            dead = ~alive.any(1)
            resets += int(dead.sum())
            idx = torch.multinomial(alive.to(D) + dead[:, None].to(D), P, replacement=True, generator=gen)
            g = idx[..., None, None].expand(-1, -1, 2, 2)
            p0, v = p0.gather(1, g), v.gather(1, g)
            if bool(dead.any()):                     # every particle contradicted: restart from the truth
                p0[dead], v[dead] = pt[dead][:, None], vt[dead][:, None]
            for _ in range(sweeps):
                p0, v, a_ = sweep(p0, v, t + 1)
                acc_sum, acc_n = acc_sum + a_, acc_n + 1
        log(f"  {min(S, c0 + chunk)}/{S} sequences  [{(time.time() - t00) / 60:.1f} min]", flush=True)

    def ms(a):                                        # mean over positions per sequence → mean, SE
        per = a.mean(1)
        return float(per.mean()), float(per.std(ddof=1) / math.sqrt(len(per))) if len(per) > 1 else float("nan")

    res = {"instance": instance, "version": FLOOR_VERSION, "method": "posterior sampling over the initial state "
           "(SMC over frames + MH rejuvenation); see pim/environments/rayworld/bayes.py",
           "split": "eval/test.h5", "n_sequences": int(S), "sequences": f"the first {S}",
           "settings": {"particles": P, "sweeps": sweeps, "init_sweeps": init_sweeps, "seed": seed, "device": device},
           "diagnostics": {"parity": 1.0, "resets": resets, "reset_share": resets / (S * (T - 1)),
                           "survivors_mean": surv_sum / (S * (T - 1)), "mh_acceptance": acc_sum / max(acc_n, 1),
                           "blink": bool(float(sim.get("blink_prob", 0)) > 0)}}
    for obj, unit in (("mse", "MSE (intensity² per ray)"), ("ce", "CE (nats per frame)")):
        lo, lo_se = ms(out[f"{obj}_lo"])
        hi, hi_se = ms(out[f"{obj}_hi"])
        res[obj] = {"lo": lo, "lo_se": lo_se, "hi": hi, "hi_se": hi_se, "unit": unit,
                    "by_position": {"lo": out[f"{obj}_lo"].mean(0).tolist(), "hi": out[f"{obj}_hi"].mean(0).tolist()}}
    if exact_draws:
        ex = exact_first_position(W, exact_draws, gen)
        f1 = levels.cpu()[ex["_f1"].long()]
        sm = torch.zeros(ex["n_patterns"], R, dtype=D).index_add_(0, ex["_inv"], f1)
        sq = torch.zeros(ex["n_patterns"], R, dtype=D).index_add_(0, ex["_inv"], f1 ** 2)
        ss = float((sq - sm ** 2 / ex["_cnt"][:, None].to(D)).sum())
        res["check_position_0"] = {"exact": ss / max(ex["n_accepted"] - ex["n_patterns"], 1) / R,
                                   "sampler_lo": float(out["mse_lo"][:, 0].mean()),
                                   "sampler_lo_se": float(out["mse_lo"][:, 0].std(ddof=1) / math.sqrt(S)),
                                   "n_accepted": ex["n_accepted"], "n_patterns": ex["n_patterns"], "usable": ex["usable"]}
    res["trivial"] = trivial_predictors(instance, n_seq=S)
    res["minutes"] = round((time.time() - t00) / 60, 1)
    log(f"{instance}: MSE floor {res['mse']['lo']:.5f}–{res['mse']['hi']:.5f}, CE floor {res['ce']['lo']:.4f}–"
        f"{res['ce']['hi']:.4f}, resets {resets}, {res['minutes']} min", flush=True)
    return res
