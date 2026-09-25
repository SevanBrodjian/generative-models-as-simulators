"""Fingerprinted, provenance-verified probe cache.

The key hashes the model's weights with every input to the fit, and a hit is checked against
its stored provenance before it is returned. Writes are atomic (``.partial`` then ``replace``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def fingerprint(model) -> str:
    """12 hex chars over every parameter — a model's identity, cheaply."""
    h = hashlib.blake2b(digest_size=6)
    for _, v in sorted(model.state_dict().items()):
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class ProbeCache:
    """One directory of cached probe fits, each stored with its full provenance."""

    #: bump to invalidate every cached probe after a change to probe fitting
    VERSION = 2

    def __init__(self, cache_dir: str | Path) -> None:
        self.dir = Path(cache_dir)

    def key(self, model, **prov) -> tuple[str, dict]:
        """(filename, provenance). Every input that changes the fitted probe belongs in
        ``prov`` — target, n_seq, split, hidden, basis, seed, data path, …"""
        # model=None is the observation baseline (probes on the raw input history); "none"
        # cannot collide with a 12-hex fingerprint
        full = {"model": fingerprint(model) if model is not None else "none",
                "span": int(getattr(model, "state_span", -1)),
                "v": self.VERSION, **prov}
        h = hashlib.blake2b(repr(sorted(full.items())).encode(), digest_size=8).hexdigest()
        return f"probes_{h}.pt", full

    def load(self, fname: str, prov: dict, device: str = "cpu"):
        """Return the cached probes, or None on a miss. Raises on provenance mismatch."""
        fpath = self.dir / fname
        if not fpath.exists():
            return None
        blob = torch.load(fpath, map_location=device, weights_only=False)
        if blob.get("provenance") != prov:
            raise RuntimeError(
                f"probe cache provenance mismatch at {fpath}\n"
                f"  on disk: {blob.get('provenance')}\n  wanted : {prov}\n"
                f"Delete the file to refit. The filename is a hash of this dict, so a "
                f"mismatch means the file was modified.")
        return blob["probes"]

    def store(self, fname: str, prov: dict, probes) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = (self.dir / fname).with_suffix(".pt.partial")
        torch.save({"provenance": prov, "probes": probes}, tmp)
        tmp.replace(self.dir / fname)  # atomic
        self.write_index()

    def write_index(self) -> Path:
        """Rewrite ``INDEX.md``, a readable table of each file's provenance (derived from the files)."""
        entries = []
        for f in sorted(self.dir.glob("probes_*.pt")):
            try:
                prov = torch.load(f, map_location="cpu", weights_only=False)["provenance"]
            except Exception as e:                     # list an unreadable file rather than skip it
                entries.append((f.name, {"ERROR": type(e).__name__}))
                continue
            entries.append((f.name, prov))
        cols, seen = [], set()
        for _, prov in entries:                        # union of keys, stable order
            for k in prov:
                if k not in seen and k not in ("model", "v"):
                    seen.add(k)
                    cols.append(k)
        lines = ["# Probe index", "",
                 "_Written by `pim.probes.cache.ProbeCache.write_index` on every store. "
                 "Each filename is a hash of the provenance dict in its row._", "",
                 "| file | " + " | ".join(cols) + " | model fingerprint |",
                 "|" + "---|" * (len(cols) + 2)]
        for name, prov in entries:
            cells = [str(prov.get(c, "—")) for c in cols]
            lines.append(f"| `{name}` | " + " | ".join(cells) + " | "
                         f"`{str(prov.get('model', '—'))[:12]}` |")
        lines.append("")
        out = self.dir / "INDEX.md"
        out.write_text("\n".join(lines))
        return out
