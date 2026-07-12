from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitIrAlignment:
    """Measured transform + confidence for a split RGB/IR Coolscan capture.

    ``mode`` is "identity" when the registration proxy was byte-identical to
    the multisampled RGB (no resample applied), or "phase-ecc" when the IR
    plane was warped by a measured translation. Confidence fields are None in
    identity mode — they were never measured, and fabricating them would make
    the acceptance report lie.
    """

    mode: str  # "identity" | "phase-ecc"
    dx_px: float
    dy_px: float
    phase_responses: tuple[float, ...] = ()
    channel_spread_px: float | None = None
    ecc_coefficient: float | None = None


@dataclass(frozen=True)
class ScanResult:
    rgb: np.ndarray
    ir: np.ndarray | None
    dpi: int
    device_model: str
    # Present only for split Coolscan RGB+IR captures: how IR was registered
    # to the multisampled RGB, with the measured confidence that acceptance
    # tooling must be able to audit.
    ir_alignment: SplitIrAlignment | None = None
    ir_valid_mask: np.ndarray | None = None
