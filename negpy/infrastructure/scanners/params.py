from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class ScanMode(StrEnum):
    NEGATIVE = "Negative"
    POSITIVE = "Positive"
    TRANSPARENCY = "Transparency"


@dataclass(frozen=True)
class RegisteredScanGeometry:
    """Coupled transport position and scan window for one registered frame.

    ``br_y_device_px`` is the inclusive bottom coordinate in the scanner's
    native device-pixel grid, not a height and not an output-resolution row.
    ``frame`` may be omitted for a single-frame carrier or the current adapter
    position.
    """

    subframe_mm: float
    br_y_device_px: int
    frame: int | None = None

    def __post_init__(self) -> None:
        if self.frame is not None and (type(self.frame) is not int or self.frame < 1):
            raise ValueError("registered frame must be a positive integer or None")
        if type(self.subframe_mm) not in (int, float) or not isfinite(self.subframe_mm) or self.subframe_mm < 0:
            raise ValueError("registered subframe must be a finite non-negative distance")
        if type(self.br_y_device_px) is not int or self.br_y_device_px < 0:
            raise ValueError("registered bottom coordinate must be a non-negative integer")


@dataclass(frozen=True)
class ScanParams:
    dpi: int
    depth: int
    capture_ir: bool
    # Normalized (x1,y1,x2,y2) window 0..1; backend maps to device units (coolscan3 int px).
    window: tuple[float, float, float, float] | None = None
    # Normalized (x1,y1,x2,y2) area 0..1; legacy name kept for roll scanning.
    area: tuple[float, float, float, float] | None = None
    # coolscan3 `subframe` (mm), applied to every frame. 0 = scanner default.
    frame_offset_mm: float = 0.0
    autofocus: bool = True
    # Select a frame on a roll-fed scanner (coolscan3) before scanning. If a
    # frame is requested and the device has no frame option, the scan fails
    # rather than reading whatever frame is under the sensor.
    frame: int | None = None
    # Registration geometry is opt-in and keeps the fine transport shift and
    # shortened scan window inseparable. ``frame`` above remains supported for
    # callers that only need legacy frame selection.
    registered_geometry: RegisteredScanGeometry | None = None
    # Hardware auto-exposure (SANE `ae`), distinct from NegPy's rendering
    # auto-exposure. An explicit request fails if the option is unavailable.
    auto_exposure: bool = False
    # Hardware multi-sampling passes per line (1 = off). Archival intent:
    # if a value > 1 cannot be honored, the scan fails rather than silently
    # degrading to a single pass.
    samples_per_scan: int = 1


MIN_FRAME_EXTENT_MM = 1.0  # below this a capped scan is a useless sliver


def clamp_frame_offset_mm(offset_mm: float, pitch_mm: float) -> float:
    """Effective feed-axis offset, floored at 0 and held short of one frame pitch.

    The transport cannot back up, and the scan blacks out one pitch past the frame
    start — at `offset >= pitch` the window collapses to zero height and the scan
    comes back empty. Pitch 0 means unknown: floor only.
    """
    offset = max(0.0, offset_mm)
    if pitch_mm <= 0:
        return offset
    return min(offset, max(0.0, pitch_mm - MIN_FRAME_EXTENT_MM))


def scan_window_to_area(
    rect: tuple[float, float, float, float] | None,
    max_area_mm: tuple[float, float],
) -> tuple[float, float, float, float] | None:
    """Normalized (x1,y1,x2,y2) window → approximate mm for the UI readout only.

    The scan maps the fraction to device units in the backend; this is display-only.
    """
    if rect is None or len(rect) != 4:
        return None
    x1, y1, x2, y2 = rect
    w, h = max_area_mm
    return (x1 * w, y1 * h, x2 * w, y2 * h)
