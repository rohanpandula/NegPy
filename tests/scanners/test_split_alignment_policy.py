"""Dependency-boundary tests for split-capture alignment policy."""

import subprocess
import sys


def test_split_alignment_policy_import_does_not_load_the_concrete_sane_backend() -> None:
    script = """
import sys
from negpy.infrastructure.scanners.split_alignment_policy import (
    SPLIT_MAX_ECC_DIVERGENCE_PX,
    SPLIT_MAX_CHANNEL_SPREAD_PX,
    SPLIT_MIN_ECC,
    SPLIT_MIN_OVERLAP_FRACTION,
    SPLIT_MIN_PHASE_RESPONSE,
    SPLIT_MIN_TEXTURE_STD,
)

assert SPLIT_MIN_PHASE_RESPONSE == 0.10
assert SPLIT_MAX_CHANNEL_SPREAD_PX == 1.0
assert SPLIT_MIN_ECC == 0.65
assert SPLIT_MAX_ECC_DIVERGENCE_PX == 2.0
assert SPLIT_MIN_TEXTURE_STD == 1e-4
assert SPLIT_MIN_OVERLAP_FRACTION == 0.5
assert "negpy.infrastructure.scanners.sane_backend" not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
