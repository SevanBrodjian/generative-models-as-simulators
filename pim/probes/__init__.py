"""Probes that read state from the residual stream, the inverse map from state to residual, and the probe cache.

Probe fits are held out by sequence, never by frame: consecutive frames are near-duplicates.
"""

from pim.probes.base import WorldStateProbe, collect_residuals, fit_probe
from pim.probes.cache import ProbeCache, fingerprint
from pim.probes.inverse import RETRIEVAL_K, RetrievalBank, fit_inverse_map
from pim.probes.linear import fit_linear
from pim.probes.mlp import CANONICAL_HIDDEN, ProbeSanityError, check_probe_sanity, fit_mlp

__all__ = [
    "WorldStateProbe",
    "collect_residuals",
    "fit_probe",
    "fit_linear",
    "fit_mlp",
    "CANONICAL_HIDDEN",
    "ProbeSanityError",
    "check_probe_sanity",
    "fit_inverse_map",
    "RetrievalBank",
    "RETRIEVAL_K",
    "ProbeCache",
    "fingerprint",
]
