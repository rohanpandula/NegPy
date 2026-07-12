"""Dependency-neutral acceptance policy for split RGB/IR alignment."""

SPLIT_MIN_PHASE_RESPONSE = 0.10  # Every RGB channel must lock at least this hard.
SPLIT_MAX_CHANNEL_SPREAD_PX = 1.0  # R/G/B translations must agree at estimation scale.
SPLIT_MIN_ECC = 0.65  # Translation-only ECC coefficient on the mean-RGB proxy.
SPLIT_MAX_ECC_DIVERGENCE_PX = 2.0  # ECC refinement must corroborate the phase shift.
SPLIT_MIN_TEXTURE_STD = 1e-4  # Full-scale dead/uniform-capture guard.
SPLIT_MIN_OVERLAP_FRACTION = 0.5  # Phase-predicted overlap required on each axis.


__all__ = [
    "SPLIT_MAX_CHANNEL_SPREAD_PX",
    "SPLIT_MAX_ECC_DIVERGENCE_PX",
    "SPLIT_MIN_ECC",
    "SPLIT_MIN_OVERLAP_FRACTION",
    "SPLIT_MIN_PHASE_RESPONSE",
    "SPLIT_MIN_TEXTURE_STD",
]
