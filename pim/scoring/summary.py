"""Per-run summaries: the headline of each ``scores.json``, human-readable."""
import json


def _fmt(v):
    return f"{v:>9.4f}" if isinstance(v, (int, float)) else f"{'-':>9}"


def _rayworld(s) -> None:
    cons = s.get("ei_construction", "ray-zone")
    for key, T in s["bases"].items():
        u = T["unedited"]
        print(f"  block={key}  target={T['target']}  basis={T['basis']}  kind={T['kind']}  "
              f"Edit Index construction={cons}  unedited {u['edit_index']:+.4f}   "
              f"Probe Skill (best point) linear {max(T['probe_skill_linear']):+.4f}  "
              f"mlp {max(T['probe_skill_mlp']):+.4f}  "
              f"sanity violations {T['probe_sanity']['n_violations']}")
        print(f"  {'editor':<8}{'dims':>6}{'pt':>4}{'alpha':>8}{'EI':>9}{'ratio':>7}"
              f"{'target':>9}{'collat':>9}")
        for ed, bst in T["best"].items():
            if bst:
                print(f"  {ed:<8}{bst.get('dims', '-'):>6}{bst['point']:>4}"
                      f"{bst['alpha']:>8}{bst['edit_index']:>+9.4f}"
                      f"{bst['fidelity_ratio']:>7.3f}{_fmt(bst.get('target_rmse'))}"
                      f"{_fmt(bst.get('collateral_rmse'))}")


def _othello(s) -> None:
    g = s["gates"]
    print(f"  gates: legal mass {g['legal_mass']:.4f}  top-1 legal {g['top1_legal']:.4f}  "
          f"CE {g['ce']:.4f} (Bayes {g['bayes_ce']:.4f}, excess {g['ce'] - g['bayes_ce']:+.4f})")
    for key in ("mine|linear|sequence", "mine|mlp|sequence"):
        if key in s["probe_skill"]:
            print(f"  Probe Skill [{key}]: best point {max(s['probe_skill'][key]):+.4f}")
    u = s["unedited"]
    print(f"  unedited Edit Index (union) {u['edit_index_union']:+.4f}   (symdiff) {u['edit_index_symdiff']:+.4f}")
    print(f"  {'editor':<8}{'pt':>4}{'alpha':>8}{'EI(un)':>9}{'EI(sd)':>9}{'li post':>9}{'li pre':>9}{'legal':>8}")
    for ed, bst in s["best"].items():
        if bst:
            print(f"  {ed:<8}{bst['point']:>4}{bst['alpha']:>8}"
                  f"{bst['edit_index_union']:>+9.4f}{bst['edit_index_symdiff']:>+9.4f}"
                  f"{bst['li_error_vs_post']:>9.3f}{bst['li_error_vs_pre']:>9.3f}{bst['legal_mass']:>8.4f}")


def print_summaries(runs) -> None:
    """Rayworld: one table per probe-target block; Othello: gates, Probe Skill and editors. The
    ``best`` arm shown is the top arm before the fidelity cutoff, with its fidelity ratio (1 minus
    Edit Fidelity); the tables apply the selection rule."""
    for r in runs:
        sp = r["dir"] / "scores.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        print(f"\n{'=' * 86}\n{s['run']}   ({s['arch']} on {s['env']}/{s['instance']})   "
              f"val {s['val_loss']:.5f}\n{'=' * 86}")
        (_rayworld if s["env"] == "rayworld" else _othello)(s)
