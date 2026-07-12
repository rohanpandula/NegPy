#!/usr/bin/env python3
"""Capture the two native RGBI sources needed by a clean-room Digital ICE path.

Nikon's captured caller consumes a distinct 285 dpi RGBI prepass before the
main RGBI image.  This runner reproduces that *acquisition boundary*
without loading Nikon code:

* one libsane initialization and one device handle;
* full-aperture 16-bit RGBI at 285 dpi, then 500 or 4000 dpi;
* focus/exposure measured by the prepass and replayed for the main pass;
* raw ``sane_read`` bytes, avoiding python-sane's known final-RGBI-row loss;
* immutable full-aperture arrays plus centered oracle-sized candidate crops;
* a receipt containing every option write, shape, digest, and ordering check.

The centered crops are a hypothesis, not a claim that Nikon's upstream crop
origin is recovered.  The full arrays are always retained, so every possible
crop with the observed dimensions can be evaluated without rescanning.

Default invocation is scanner-free and prints the exact live command:

    uv run python -m negpy.infrastructure.scanners.dice_dual_source_runner

Live acquisition requires an explicit physical confirmation:

    uv run python -m negpy.infrastructure.scanners.dice_dual_source_runner \
      --live --confirm-film-stationary
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import math
import os
import shlex
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from negpy.infrastructure.scanners.params import ScannerCaptureState


NATIVE_OPTICAL_DPI = 4000
ORACLE_PREPASS_DPI = 285
ORACLE_MAIN_DPI = 500
NATIVE_MAIN_DPI = 4000
ROLL_FULL_APERTURE = (0, 0, 3945, 5958)  # inclusive tl_x, tl_y, br_x, br_y
MOUNTED_FULL_APERTURE = (0, 0, 3945, 5781)
FULL_APERTURE = ROLL_FULL_APERTURE
ORACLE_PREPASS_SHAPE = (413, 281)
ORACLE_MAIN_SHAPE = (676, 443)
NATIVE_MAIN_SHAPE = (5959, 3946)
MOUNTED_NATIVE_MAIN_SHAPE = (5782, 3946)

SANE_STATUS_GOOD = 0
SANE_STATUS_EOF = 5
SANE_ACTION_GET_VALUE = 0
SANE_ACTION_SET_VALUE = 1
SANE_TYPE_BOOL = 0
SANE_TYPE_INT = 1
SANE_TYPE_FIXED = 2
SANE_TYPE_STRING = 3
SANE_TYPE_BUTTON = 4
SANE_CONSTRAINT_NONE = 0
SANE_CONSTRAINT_RANGE = 1
SANE_CONSTRAINT_WORD_LIST = 2
SANE_CONSTRAINT_STRING_LIST = 3
SANE_CAP_SOFT_SELECT = 1 << 0
SANE_CAP_INACTIVE = 1 << 5
SANE_INFO_RELOAD_OPTIONS = 1 << 1
SANE_FRAME_RGB = 1


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _normalise_option_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _strip_net_prefix(device_id: str) -> str:
    if not device_id.startswith("net:"):
        return device_id
    rest = device_id[4:]
    if rest.startswith("["):
        close = rest.find("]:")
        return rest[close + 2 :] if close >= 0 else device_id
    _, separator, backend = rest.partition(":")
    return backend if separator else device_id


@dataclass(frozen=True)
class PixelWindow:
    tl_x: int
    tl_y: int
    br_x: int
    br_y: int

    def __post_init__(self) -> None:
        values = (self.tl_x, self.tl_y, self.br_x, self.br_y)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("pixel-window coordinates must be non-negative integers")
        if self.br_x < self.tl_x or self.br_y < self.tl_y:
            raise ValueError("pixel-window bottom-right must not precede top-left")

    @property
    def native_width(self) -> int:
        return self.br_x - self.tl_x + 1

    @property
    def native_height(self) -> int:
        return self.br_y - self.tl_y + 1

    def output_shape(self, dpi: int) -> tuple[int, int]:
        if type(dpi) is not int or dpi <= 0:
            raise ValueError("dpi must be a positive integer")
        pitch = NATIVE_OPTICAL_DPI // dpi
        if pitch <= 0 or NATIVE_OPTICAL_DPI // pitch != dpi:
            raise ValueError(f"{dpi} dpi is not an exact LS-5000 integer-pitch mode")
        return self.native_height // pitch, self.native_width // pitch


@dataclass(frozen=True)
class DiceDualSourcePlan:
    """One bounded 285→main-DPI acquisition for Digital ICE."""

    window: PixelWindow = PixelWindow(*FULL_APERTURE)
    prepass_dpi: int = ORACLE_PREPASS_DPI
    main_dpi: int = ORACLE_MAIN_DPI
    prepass_target_shape: tuple[int, int] = ORACLE_PREPASS_SHAPE
    main_target_shape: tuple[int, int] = ORACLE_MAIN_SHAPE
    depth: int = 16
    frame: int | None = None
    subframe_mm: float | None = None
    transport: str = "roll"

    @classmethod
    def for_main_dpi(
        cls,
        main_dpi: int,
        *,
        frame: int | None = None,
        subframe_mm: float | None = None,
        transport: str = "roll",
    ) -> DiceDualSourcePlan:
        """Build one supported 500- or 4000-dpi main-capture plan."""

        if transport not in {"roll", "mounted"}:
            raise ValueError("transport must be 'roll' or 'mounted'")
        if transport == "mounted" and (frame is not None or subframe_mm is not None):
            raise ValueError("mounted transport cannot select a roll frame or subframe")
        window = PixelWindow(
            *(MOUNTED_FULL_APERTURE if transport == "mounted" else ROLL_FULL_APERTURE)
        )

        if main_dpi == ORACLE_MAIN_DPI:
            target_shape = ORACLE_MAIN_SHAPE
        elif main_dpi == NATIVE_MAIN_DPI:
            target_shape = window.output_shape(main_dpi)
        else:
            raise ValueError(
                f"main_dpi must be {ORACLE_MAIN_DPI} or {NATIVE_MAIN_DPI}"
            )
        return cls(
            window=window,
            main_dpi=main_dpi,
            main_target_shape=target_shape,
            frame=frame,
            subframe_mm=subframe_mm,
            transport=transport,
        )

    def __post_init__(self) -> None:
        if self.transport not in {"roll", "mounted"}:
            raise ValueError("transport must be 'roll' or 'mounted'")
        if self.depth != 16:
            raise ValueError("the DICE source contract requires 16-bit samples")
        if self.frame is not None and (type(self.frame) is not int or self.frame < 1):
            raise ValueError("frame must be a positive integer or None")
        if self.subframe_mm is not None:
            if self.frame is None:
                raise ValueError("subframe_mm requires a selected roll frame")
            if not math.isfinite(self.subframe_mm) or self.subframe_mm < 0:
                raise ValueError("subframe_mm must be finite and non-negative")
        for label, shape in (
            ("prepass_target_shape", self.prepass_target_shape),
            ("main_target_shape", self.main_target_shape),
        ):
            if len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape):
                raise ValueError(f"{label} must contain two positive integers")
        self.crop_offsets("prepass")
        self.crop_offsets("main")

    @property
    def prepass_full_shape(self) -> tuple[int, int]:
        return self.window.output_shape(self.prepass_dpi)

    @property
    def main_full_shape(self) -> tuple[int, int]:
        return self.window.output_shape(self.main_dpi)

    def crop_offsets(self, epoch: str) -> tuple[int, int]:
        if epoch == "prepass":
            full, target = self.prepass_full_shape, self.prepass_target_shape
        elif epoch == "main":
            full, target = self.main_full_shape, self.main_target_shape
        else:
            raise ValueError(f"unknown DICE epoch {epoch!r}")
        delta_y, delta_x = full[0] - target[0], full[1] - target[1]
        if delta_y < 0 or delta_x < 0:
            raise ValueError(f"{epoch} target {target} exceeds full SANE shape {full}")
        if delta_y % 2 or delta_x % 2:
            raise ValueError(f"{epoch} target {target} cannot be centered exactly in full SANE shape {full}")
        return delta_y // 2, delta_x // 2

    def semantic_dict(self) -> dict[str, object]:
        return {
            "native_optical_dpi": NATIVE_OPTICAL_DPI,
            "window": asdict(self.window),
            "prepass": {
                "dpi": self.prepass_dpi,
                "full_shape_hw": list(self.prepass_full_shape),
                "dice_candidate_shape_hw": list(self.prepass_target_shape),
                "center_crop_offset_yx": list(self.crop_offsets("prepass")),
            },
            "main": {
                "dpi": self.main_dpi,
                "full_shape_hw": list(self.main_full_shape),
                "dice_candidate_shape_hw": list(self.main_target_shape),
                "center_crop_offset_yx": list(self.crop_offsets("main")),
            },
            "depth": self.depth,
            "channels": ["red", "green", "blue", "infrared"],
            "transport": self.transport,
            "frame": self.frame,
            "subframe_mm": self.subframe_mm,
            "candidate_crop_status": "centered hypothesis; full source retained for exhaustive crop search",
        }


@dataclass(frozen=True)
class OptionInfo:
    name: str
    value_type: int
    active: bool
    settable: bool
    constraint: tuple[float | int | str, ...] | None = None
    range_constraint: tuple[float, float, float] | None = None

    def supports(self, value: bool | int | float | str) -> bool:
        if self.constraint is not None:
            return any(
                candidate == value
                or (
                    isinstance(candidate, (int, float))
                    and isinstance(value, (int, float))
                    and math.isclose(float(candidate), float(value), rel_tol=0.0, abs_tol=1e-9)
                )
                for candidate in self.constraint
            )
        if self.range_constraint is None:
            return True
        lower, upper, quantum = self.range_constraint
        numeric = float(value)
        if not lower <= numeric <= upper:
            return False
        if quantum <= 0:
            return True
        steps = (numeric - lower) / quantum
        return math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1e-9)


@dataclass(frozen=True)
class RawFrame:
    rgbi: np.ndarray
    bytes_per_line: int
    bytes_read: int
    frame_format: int = SANE_FRAME_RGB
    last_frame: bool = True
    depth: int = 16


class RawSaneDevice(Protocol):
    device_id: str

    def options(self) -> dict[str, OptionInfo]: ...
    def set_option(self, name: str, value: bool | int | float | str) -> None: ...
    def get_option(self, name: str) -> bool | int | float | str: ...
    def read_rgbi(self, *, expected_shape: tuple[int, int], label: str) -> RawFrame: ...
    def cancel(self) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class DualSourceCapture:
    prepass_full: np.ndarray
    prepass_candidate: np.ndarray
    main_full: np.ndarray
    main_candidate: np.ndarray
    capture_state: ScannerCaptureState
    events: tuple[dict[str, object], ...]
    assertions: dict[str, bool]


def _center_crop(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError(f"expected an RGBI array, found shape {array.shape}")
    target_h, target_w = shape
    delta_h, delta_w = array.shape[0] - target_h, array.shape[1] - target_w
    if delta_h < 0 or delta_w < 0 or delta_h % 2 or delta_w % 2:
        raise ValueError(f"cannot center {shape} inside RGBI shape {array.shape[:2]}")
    top, left = delta_h // 2, delta_w // 2
    return np.ascontiguousarray(array[top : top + target_h, left : left + target_w])


def crop_oracle_candidate(
    array: np.ndarray,
    *,
    plan: DiceDualSourcePlan,
    epoch: str,
) -> np.ndarray:
    """Return the deterministic centered candidate for one observed DICE epoch.

    This helper intentionally validates the *full* native acquisition before
    cropping.  It cannot silently accept an already-truncated python-sane
    array, and it keeps the centered-origin assumption explicit at the call
    site.  The candidate is suitable for the first native-vs-Nikon geometry
    comparison; it is not evidence that Nikon used a centered crop.
    """

    if epoch == "prepass":
        full_shape = plan.prepass_full_shape
        target_shape = plan.prepass_target_shape
    elif epoch == "main":
        full_shape = plan.main_full_shape
        target_shape = plan.main_target_shape
    else:
        raise ValueError(f"unknown DICE epoch {epoch!r}")
    expected = (*full_shape, 4)
    if array.shape != expected:
        raise ValueError(f"{epoch} full RGBI shape {array.shape} does not match {expected}")
    return _center_crop(array, target_shape)


def _decode_rgbi16(payload: bytes | bytearray, *, width: int, height: int, bytes_per_line: int) -> np.ndarray:
    """Decode every raw SANE row, including rows python-sane discards."""

    expected_bpl = width * 4 * 2
    if bytes_per_line != expected_bpl:
        raise RuntimeError(f"raw RGBI bytes_per_line {bytes_per_line} != expected {expected_bpl}")
    expected_bytes = bytes_per_line * height
    if len(payload) != expected_bytes:
        raise RuntimeError(f"raw RGBI payload has {len(payload)} bytes; expected {expected_bytes}")
    return np.frombuffer(payload, dtype=np.dtype("=u2")).reshape(height, width, 4).copy()


def _validate_raw_frame(frame: RawFrame, *, expected_shape: tuple[int, int], label: str) -> None:
    expected_array_shape = (*expected_shape, 4)
    expected_bpl = expected_shape[1] * 4 * 2
    expected_bytes = expected_bpl * expected_shape[0]
    failures: list[str] = []
    if frame.rgbi.shape != expected_array_shape:
        failures.append(f"shape={frame.rgbi.shape}, expected {expected_array_shape}")
    if frame.rgbi.dtype != np.uint16:
        failures.append(f"dtype={frame.rgbi.dtype}, expected uint16")
    if frame.bytes_per_line != expected_bpl:
        failures.append(f"bytes_per_line={frame.bytes_per_line}, expected {expected_bpl}")
    if frame.bytes_read != expected_bytes:
        failures.append(f"bytes_read={frame.bytes_read}, expected {expected_bytes}")
    if frame.frame_format != SANE_FRAME_RGB:
        failures.append(f"format={frame.frame_format}, expected RGB")
    if not frame.last_frame:
        failures.append("last_frame is false")
    if frame.depth != 16:
        failures.append(f"depth={frame.depth}, expected 16")
    if failures:
        raise RuntimeError(f"{label} raw RGBI frame refused: " + "; ".join(failures))


def _read_state(device: RawSaneDevice) -> ScannerCaptureState:
    try:
        state = ScannerCaptureState(
            focus_position=int(device.get_option("focus")),
            exposure_multiplier=float(device.get_option("exposure")),
            red_exposure_us=float(device.get_option("red_exposure")),
            green_exposure_us=float(device.get_option("green_exposure")),
            blue_exposure_us=float(device.get_option("blue_exposure")),
        )
    except Exception as exc:
        raise RuntimeError(f"could not read locked focus/exposure state: {exc}") from exc
    if state.focus_position <= 0:
        raise RuntimeError(f"scanner returned uncalibrated focus position {state.focus_position}")
    return state


def _preflight(device: RawSaneDevice, plan: DiceDualSourcePlan) -> dict[str, OptionInfo]:
    options = device.options()
    required = {
        "depth": plan.depth,
        "resolution": plan.prepass_dpi,
        "preview": False,
        "negative": False,
        "samples_per_scan": 1,
        "infrared": True,
        "autofocus": True,
        "ae": True,
        "focus": 1,
        "exposure": 1.0,
        "red_exposure": 1.0,
        "green_exposure": 1.0,
        "blue_exposure": 1.0,
        "tl_x": plan.window.tl_x,
        "tl_y": plan.window.tl_y,
        "br_x": plan.window.br_x,
        "br_y": plan.window.br_y,
    }
    if plan.frame is not None:
        required["frame"] = plan.frame
        required["frame_count"] = 1
    if plan.subframe_mm is not None:
        required["subframe"] = plan.subframe_mm
    failures: list[str] = []
    for name, value in required.items():
        info = options.get(name)
        if info is None:
            failures.append(f"option {name!r} is missing")
            continue
        if not info.active:
            failures.append(f"option {name!r} is inactive")
        if not info.settable:
            failures.append(f"option {name!r} is not settable")
        # Exposure/focus values above only prove writability; the measured
        # values are not known until after the prepass.
        if name not in {"focus", "exposure", "red_exposure", "green_exposure", "blue_exposure"} and not info.supports(value):
            failures.append(f"option {name!r} cannot accept {value!r}")
    resolution = options.get("resolution")
    if resolution is not None and not resolution.supports(plan.main_dpi):
        failures.append(f"option 'resolution' cannot accept main dpi {plan.main_dpi}")
    if failures:
        raise RuntimeError("dual RGBI preflight failed before scanner mutation: " + "; ".join(failures))
    return options


def _derive_assertions(events: list[dict[str, object]], plan: DiceDualSourcePlan) -> dict[str, bool]:
    starts = [index for index, event in enumerate(events) if event["event"] == "read_begin"]
    ends = [index for index, event in enumerate(events) if event["event"] == "read_end"]
    sets = [(index, event) for index, event in enumerate(events) if event["event"] == "set"]
    resolution_sets = [event["value"] for _, event in sets if event["option"] == "resolution"]
    frame_sets = [index for index, event in sets if event["option"] == "frame"]
    transport_after_first = any(
        index > starts[0] and event["option"] in {"frame", "subframe"}
        for index, event in sets
    ) if starts else True
    checks = {
        "exactly_two_reads": len(starts) == 2 and len(ends) == 2,
        "prepass_then_main": len(starts) == 2 and len(ends) == 2 and starts[0] < ends[0] < starts[1] < ends[1],
        "resolution_prepass_then_main": resolution_sets
        == [plan.prepass_dpi, plan.main_dpi],
        "one_or_zero_frame_write_before_prepass": (
            (plan.frame is None and not frame_sets) or (plan.frame is not None and len(frame_sets) == 1 and frame_sets[0] < starts[0])
        ) if starts else False,
        "no_transport_write_after_prepass_started": not transport_after_first,
        "raw_reader_preserved_prepass_rows": any(
            event["event"] == "read_end" and event["epoch"] == "prepass" and event["shape"] == [*plan.prepass_full_shape, 4]
            for event in events
        ),
        "raw_reader_preserved_main_rows": any(
            event["event"] == "read_end" and event["epoch"] == "main" and event["shape"] == [*plan.main_full_shape, 4]
            for event in events
        ),
    }
    checks["all_passed"] = all(checks.values())
    return checks


def acquire_dual_sources(device: RawSaneDevice, plan: DiceDualSourcePlan) -> DualSourceCapture:
    """Acquire prepass then main on one already-open raw SANE handle."""

    options = _preflight(device, plan)
    events: list[dict[str, object]] = []

    def record(event: str, **fields: object) -> None:
        events.append({"sequence": len(events) + 1, "ts": _now(), "event": event, **fields})

    def set_value(name: str, value: bool | int | float | str) -> None:
        device.set_option(name, value)
        record("set", option=name, value=value)

    # Establish a raw, single-sample, full-aperture RGBI contract.  Disable
    # infrared before lowering samples so a stale unsafe device state cannot
    # make the first write fail or wedge the LS-5000.
    set_value("preview", False)
    set_value("infrared", False)
    set_value("samples_per_scan", 1)
    set_value("depth", plan.depth)
    set_value("negative", False)
    if plan.frame is not None:
        set_value("frame", plan.frame)
        set_value("frame_count", 1)
    if plan.subframe_mm is not None:
        set_value("subframe", plan.subframe_mm)
    for name, value in (
        ("tl_x", plan.window.tl_x),
        ("tl_y", plan.window.tl_y),
        ("br_x", plan.window.br_x),
        ("br_y", plan.window.br_y),
    ):
        set_value(name, value)
    set_value("infrared", True)
    set_value("resolution", plan.prepass_dpi)
    set_value("autofocus", True)
    set_value("ae", True)

    record("read_begin", epoch="prepass", expected_shape=[*plan.prepass_full_shape, 4])
    prepass = device.read_rgbi(expected_shape=plan.prepass_full_shape, label="prepass")
    _validate_raw_frame(prepass, expected_shape=plan.prepass_full_shape, label="prepass")
    record(
        "read_end",
        epoch="prepass",
        shape=list(prepass.rgbi.shape),
        dtype=np.dtype(prepass.rgbi.dtype).name,
        bytes=prepass.bytes_read,
        sha256=_sha256_bytes(memoryview(np.ascontiguousarray(prepass.rgbi)).cast("B")),
    )
    state = _read_state(device)
    record("capture_state_read", **asdict(state))

    # sane_read consumes the roll exposure counter.  A mounted holder exposes
    # frame_count inactive; a roll adapter needs one exact reset here.
    frame_count = options.get("frame_count")
    if frame_count is not None and frame_count.active and frame_count.settable:
        set_value("frame_count", 1)

    set_value("autofocus", False)
    set_value("ae", False)
    for name, value in (
        ("focus", state.focus_position),
        ("exposure", state.exposure_multiplier),
        ("red_exposure", state.red_exposure_us),
        ("green_exposure", state.green_exposure_us),
        ("blue_exposure", state.blue_exposure_us),
    ):
        set_value(name, value)
    set_value("resolution", plan.main_dpi)

    record("read_begin", epoch="main", expected_shape=[*plan.main_full_shape, 4])
    main = device.read_rgbi(expected_shape=plan.main_full_shape, label="main")
    _validate_raw_frame(main, expected_shape=plan.main_full_shape, label="main")
    record(
        "read_end",
        epoch="main",
        shape=list(main.rgbi.shape),
        dtype=np.dtype(main.rgbi.dtype).name,
        bytes=main.bytes_read,
        sha256=_sha256_bytes(memoryview(np.ascontiguousarray(main.rgbi)).cast("B")),
    )
    replayed_state = _read_state(device)
    if replayed_state != state:
        raise RuntimeError(f"main acquisition did not retain locked capture state: {replayed_state!r} != {state!r}")
    record("capture_state_verified", **asdict(replayed_state))

    prepass_full = np.ascontiguousarray(prepass.rgbi)
    main_full = np.ascontiguousarray(main.rgbi)
    assertions = _derive_assertions(events, plan)
    if not assertions["all_passed"]:
        raise RuntimeError(f"dual RGBI ordering assertions failed: {assertions}")
    return DualSourceCapture(
        prepass_full=prepass_full,
        prepass_candidate=crop_oracle_candidate(prepass_full, plan=plan, epoch="prepass"),
        main_full=main_full,
        main_candidate=crop_oracle_candidate(main_full, plan=plan, epoch="main"),
        capture_state=state,
        events=tuple(events),
        assertions=assertions,
    )


class _SaneRange(ctypes.Structure):
    _fields_ = [("minimum", ctypes.c_int32), ("maximum", ctypes.c_int32), ("quant", ctypes.c_int32)]


class _SaneConstraint(ctypes.Union):
    _fields_ = [
        ("string_list", ctypes.POINTER(ctypes.c_char_p)),
        ("word_list", ctypes.POINTER(ctypes.c_int32)),
        ("range", ctypes.POINTER(_SaneRange)),
    ]


class _SaneOptionDescriptor(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("title", ctypes.c_char_p),
        ("description", ctypes.c_char_p),
        ("value_type", ctypes.c_int),
        ("unit", ctypes.c_int),
        ("size", ctypes.c_int32),
        ("cap", ctypes.c_int32),
        ("constraint_type", ctypes.c_int),
        ("constraint", _SaneConstraint),
    ]


class _SaneParameters(ctypes.Structure):
    _fields_ = [
        ("frame_format", ctypes.c_int),
        ("last_frame", ctypes.c_int32),
        ("bytes_per_line", ctypes.c_int32),
        ("pixels_per_line", ctypes.c_int32),
        ("lines", ctypes.c_int32),
        ("depth", ctypes.c_int32),
    ]


class _SaneDevice(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("vendor", ctypes.c_char_p), ("model", ctypes.c_char_p), ("kind", ctypes.c_char_p)]


def _decode_c_string(value: bytes | None) -> str:
    return value.decode("utf-8", errors="replace") if value is not None else ""


class Libsane:
    """Minimal raw SANE binding used only where python-sane loses RGBI rows."""

    def __init__(self, library_path: str | None = None) -> None:
        candidates = [
            library_path,
            ctypes.util.find_library("sane"),
            "/opt/homebrew/opt/sane-backends/lib/libsane.1.dylib",
            "/usr/local/lib/libsane.so.1",
            "libsane.so.1",
        ]
        last_error: OSError | None = None
        for candidate in candidates:
            if not candidate:
                continue
            try:
                self._lib = ctypes.CDLL(candidate)
                self.library_path = str(candidate)
                break
            except OSError as error:
                last_error = error
        else:
            raise RuntimeError(f"could not load libsane: {last_error}")
        self._configure_signatures()
        version = ctypes.c_int32()
        self._check(self._lib.sane_init(ctypes.byref(version), None), "sane_init")
        self.version_code = int(version.value)
        self._closed = False

    def _configure_signatures(self) -> None:
        lib = self._lib
        lib.sane_init.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_void_p]
        lib.sane_init.restype = ctypes.c_int
        lib.sane_exit.argtypes = []
        lib.sane_exit.restype = None
        device_list_type = ctypes.POINTER(ctypes.POINTER(_SaneDevice))
        lib.sane_get_devices.argtypes = [ctypes.POINTER(device_list_type), ctypes.c_int32]
        lib.sane_get_devices.restype = ctypes.c_int
        lib.sane_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.sane_open.restype = ctypes.c_int
        lib.sane_close.argtypes = [ctypes.c_void_p]
        lib.sane_close.restype = None
        lib.sane_get_option_descriptor.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.sane_get_option_descriptor.restype = ctypes.POINTER(_SaneOptionDescriptor)
        lib.sane_control_option.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
        lib.sane_control_option.restype = ctypes.c_int
        lib.sane_start.argtypes = [ctypes.c_void_p]
        lib.sane_start.restype = ctypes.c_int
        lib.sane_get_parameters.argtypes = [ctypes.c_void_p, ctypes.POINTER(_SaneParameters)]
        lib.sane_get_parameters.restype = ctypes.c_int
        lib.sane_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
        lib.sane_read.restype = ctypes.c_int
        lib.sane_cancel.argtypes = [ctypes.c_void_p]
        lib.sane_cancel.restype = None
        lib.sane_strstatus.argtypes = [ctypes.c_int]
        lib.sane_strstatus.restype = ctypes.c_char_p

    def _check(self, status: int, action: str) -> None:
        if status != SANE_STATUS_GOOD:
            message = _decode_c_string(self._lib.sane_strstatus(status))
            raise RuntimeError(f"{action} failed with SANE status {status}: {message}")

    def list_devices(self) -> list[str]:
        device_list = ctypes.POINTER(ctypes.POINTER(_SaneDevice))()
        self._check(self._lib.sane_get_devices(ctypes.byref(device_list), 0), "sane_get_devices")
        result: list[str] = []
        index = 0
        while device_list[index]:
            result.append(_decode_c_string(device_list[index].contents.name))
            index += 1
        return result

    def discover_coolscan3(self) -> str:
        devices = self.list_devices()
        matches = [device for device in devices if _strip_net_prefix(device).startswith("coolscan3:")]
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one coolscan3 device, found {matches or 'none'} (all devices: {devices or 'none'})")
        return matches[0]

    def open(self, device_id: str) -> LibsaneRawDevice:
        handle = ctypes.c_void_p()
        self._check(self._lib.sane_open(device_id.encode(), ctypes.byref(handle)), f"sane_open({device_id!r})")
        return LibsaneRawDevice(self, device_id, handle)

    def close(self) -> None:
        if not self._closed:
            self._lib.sane_exit()
            self._closed = True


class LibsaneRawDevice:
    def __init__(self, owner: Libsane, device_id: str, handle: ctypes.c_void_p) -> None:
        self._owner = owner
        self._lib = owner._lib
        self.device_id = device_id
        self._handle = handle
        self._closed = False
        self._options: dict[str, tuple[int, _SaneOptionDescriptor]] = {}
        self._refresh_options()

    def _check(self, status: int, action: str) -> None:
        self._owner._check(status, action)

    def _refresh_options(self) -> None:
        count_value = ctypes.c_int32()
        info = ctypes.c_int32()
        self._check(
            self._lib.sane_control_option(self._handle, 0, SANE_ACTION_GET_VALUE, ctypes.byref(count_value), ctypes.byref(info)),
            "read SANE option count",
        )
        options: dict[str, tuple[int, _SaneOptionDescriptor]] = {}
        for index in range(1, int(count_value.value)):
            pointer = self._lib.sane_get_option_descriptor(self._handle, index)
            if not pointer or not pointer.contents.name:
                continue
            descriptor = pointer.contents
            options[_normalise_option_name(_decode_c_string(descriptor.name))] = (index, descriptor)
        self._options = options

    @staticmethod
    def _fixed(raw: int) -> float:
        return raw / 65536.0

    def _option_info(self, name: str, descriptor: _SaneOptionDescriptor) -> OptionInfo:
        constraint: tuple[float | int | str, ...] | None = None
        range_constraint: tuple[float, float, float] | None = None
        if descriptor.constraint_type == SANE_CONSTRAINT_WORD_LIST and descriptor.constraint.word_list:
            words = descriptor.constraint.word_list
            count = int(words[0])
            values: list[float | int] = []
            for index in range(1, count + 1):
                raw = int(words[index])
                values.append(self._fixed(raw) if descriptor.value_type == SANE_TYPE_FIXED else raw)
            constraint = tuple(values)
        elif descriptor.constraint_type == SANE_CONSTRAINT_RANGE and descriptor.constraint.range:
            raw = descriptor.constraint.range.contents
            if descriptor.value_type == SANE_TYPE_FIXED:
                range_constraint = (self._fixed(raw.minimum), self._fixed(raw.maximum), self._fixed(raw.quant))
            else:
                range_constraint = (float(raw.minimum), float(raw.maximum), float(raw.quant))
        elif descriptor.constraint_type == SANE_CONSTRAINT_STRING_LIST and descriptor.constraint.string_list:
            values_string: list[str] = []
            index = 0
            while descriptor.constraint.string_list[index]:
                values_string.append(_decode_c_string(descriptor.constraint.string_list[index]))
                index += 1
            constraint = tuple(values_string)
        return OptionInfo(
            name=name,
            value_type=descriptor.value_type,
            active=not bool(descriptor.cap & SANE_CAP_INACTIVE),
            settable=bool(descriptor.cap & SANE_CAP_SOFT_SELECT),
            constraint=constraint,
            range_constraint=range_constraint,
        )

    def options(self) -> dict[str, OptionInfo]:
        return {name: self._option_info(name, descriptor) for name, (_, descriptor) in self._options.items()}

    def _descriptor(self, name: str) -> tuple[int, _SaneOptionDescriptor]:
        normalized = _normalise_option_name(name)
        try:
            return self._options[normalized]
        except KeyError as exc:
            raise RuntimeError(f"SANE option {normalized!r} is unavailable") from exc

    def set_option(self, name: str, value: bool | int | float | str) -> None:
        index, descriptor = self._descriptor(name)
        info = ctypes.c_int32()
        if descriptor.value_type in (SANE_TYPE_BOOL, SANE_TYPE_INT, SANE_TYPE_FIXED):
            if descriptor.value_type == SANE_TYPE_FIXED:
                raw_value = round(float(value) * 65536.0)
            else:
                raw_value = int(value)
            payload: object = ctypes.c_int32(raw_value)
        elif descriptor.value_type == SANE_TYPE_STRING:
            encoded = str(value).encode()
            if len(encoded) + 1 > descriptor.size:
                raise RuntimeError(f"value for SANE option {name!r} exceeds its {descriptor.size}-byte buffer")
            payload = ctypes.create_string_buffer(encoded, descriptor.size)
        elif descriptor.value_type == SANE_TYPE_BUTTON:
            payload = ctypes.c_int32()
        else:
            raise RuntimeError(f"unsupported SANE option type {descriptor.value_type} for {name!r}")
        self._check(
            self._lib.sane_control_option(self._handle, index, SANE_ACTION_SET_VALUE, ctypes.byref(payload), ctypes.byref(info)),
            f"set SANE option {name}={value!r}",
        )
        if info.value & SANE_INFO_RELOAD_OPTIONS:
            self._refresh_options()

    def get_option(self, name: str) -> bool | int | float | str:
        index, descriptor = self._descriptor(name)
        info = ctypes.c_int32()
        if descriptor.value_type in (SANE_TYPE_BOOL, SANE_TYPE_INT, SANE_TYPE_FIXED):
            payload: object = ctypes.c_int32()
        elif descriptor.value_type == SANE_TYPE_STRING:
            payload = ctypes.create_string_buffer(descriptor.size)
        else:
            raise RuntimeError(f"SANE option {name!r} cannot be read as a scalar")
        self._check(
            self._lib.sane_control_option(self._handle, index, SANE_ACTION_GET_VALUE, ctypes.byref(payload), ctypes.byref(info)),
            f"read SANE option {name}",
        )
        if descriptor.value_type == SANE_TYPE_BOOL:
            return bool(payload.value)  # type: ignore[attr-defined]
        if descriptor.value_type == SANE_TYPE_INT:
            return int(payload.value)  # type: ignore[attr-defined]
        if descriptor.value_type == SANE_TYPE_FIXED:
            return self._fixed(int(payload.value))  # type: ignore[attr-defined]
        return bytes(payload.value).decode("utf-8", errors="replace")  # type: ignore[attr-defined]

    def read_rgbi(self, *, expected_shape: tuple[int, int], label: str) -> RawFrame:
        self._check(self._lib.sane_start(self._handle), f"start {label} scan")
        parameters = _SaneParameters()
        self._check(self._lib.sane_get_parameters(self._handle, ctypes.byref(parameters)), f"read {label} parameters")
        height, width = expected_shape
        expected_bpl = width * 4 * 2
        failures = []
        if parameters.frame_format != SANE_FRAME_RGB:
            failures.append(f"format={parameters.frame_format}, expected RGB")
        if parameters.last_frame != 1:
            failures.append(f"last_frame={parameters.last_frame}, expected 1")
        if parameters.depth != 16:
            failures.append(f"depth={parameters.depth}, expected 16")
        if (parameters.lines, parameters.pixels_per_line) != expected_shape:
            failures.append(f"shape={(parameters.lines, parameters.pixels_per_line)}, expected {expected_shape}")
        if parameters.bytes_per_line != expected_bpl:
            failures.append(f"bytes_per_line={parameters.bytes_per_line}, expected {expected_bpl}")
        if failures:
            raise RuntimeError(f"{label} SANE metadata refused: " + "; ".join(failures))

        expected_bytes = expected_bpl * height
        payload = bytearray(expected_bytes)
        offset = 0
        chunk = (ctypes.c_ubyte * (1024 * 1024))()
        while True:
            delivered = ctypes.c_int32()
            status = self._lib.sane_read(self._handle, chunk, len(chunk), ctypes.byref(delivered))
            count = int(delivered.value)
            if count < 0 or offset + count > expected_bytes:
                raise RuntimeError(f"{label} raw SANE read exceeded declared frame size")
            if count:
                payload[offset : offset + count] = ctypes.string_at(chunk, count)
                offset += count
            if status == SANE_STATUS_EOF:
                break
            self._check(status, f"read {label} RGBI bytes")
            if count == 0:
                raise RuntimeError(f"{label} raw SANE read returned success with zero bytes")
        if offset != expected_bytes:
            raise RuntimeError(f"{label} raw SANE frame ended at {offset} bytes; expected {expected_bytes}")
        array = _decode_rgbi16(payload, width=width, height=height, bytes_per_line=parameters.bytes_per_line)
        return RawFrame(
            rgbi=array,
            bytes_per_line=parameters.bytes_per_line,
            bytes_read=offset,
            frame_format=parameters.frame_format,
            last_frame=bool(parameters.last_frame),
            depth=parameters.depth,
        )

    def cancel(self) -> None:
        if not self._closed:
            self._lib.sane_cancel(self._handle)

    def close(self) -> None:
        if not self._closed:
            self._lib.sane_close(self._handle)
            self._closed = True


def _write_npy(path: Path, array: np.ndarray) -> dict[str, object]:
    with path.open("wb") as stream:
        np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "shape": list(array.shape),
        "dtype": np.dtype(array.dtype).name,
        "array_payload_sha256": _sha256_bytes(memoryview(np.ascontiguousarray(array)).cast("B")),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_capture_bundle(
    output_dir: str | Path,
    *,
    device_id: str,
    plan: DiceDualSourcePlan,
    capture: DualSourceCapture,
    run_id: str | None = None,
) -> Path:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = datetime.now().astimezone().strftime("dice-dual-%Y%m%d-%H%M%S") + f"-{uuid.uuid4().hex[:8]}"
    if not run_id or "/" in run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be one safe path component")
    final = root / run_id
    if final.exists():
        raise FileExistsError(f"capture bundle already exists: {final}")
    partial = Path(tempfile.mkdtemp(prefix=f".{run_id}.", suffix=".partial", dir=root))
    try:
        arrays = {
            "prepass_full": capture.prepass_full,
            "prepass_candidate": capture.prepass_candidate,
            "main_full": capture.main_full,
            "main_candidate": capture.main_candidate,
        }
        artifacts = {role: _write_npy(partial / f"{role}.npy", array) for role, array in arrays.items()}
        manifest = {
            "schema": "negpy.dice-dual-rgbi.v1",
            "created_at": _now(),
            "device_id": device_id,
            "reader": "direct libsane sane_read; no python-sane arr_snap and no Nikon runtime",
            "native_byteorder": sys.byteorder,
            "plan": plan.semantic_dict(),
            "capture_state": asdict(capture.capture_state),
            "events": list(capture.events),
            "assertions": capture.assertions,
            "artifacts": artifacts,
        }
        manifest_path = partial / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        receipt = {
            "schema": "negpy.dice-dual-rgbi-receipt.v1",
            "manifest": manifest_path.name,
            "manifest_sha256": _sha256_file(manifest_path),
        }
        receipt_path = partial / "receipt.json"
        with receipt_path.open("w", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(partial)
        os.replace(partial, final)
        _fsync_directory(root)
    except BaseException:
        for path in partial.iterdir():
            path.unlink(missing_ok=True)
        partial.rmdir()
        raise
    return final


def exact_next_command(
    *,
    output_dir: str | Path,
    device_id: str | None = None,
    frame: int | None = None,
    subframe_mm: float | None = None,
    main_dpi: int = ORACLE_MAIN_DPI,
    transport: str = "roll",
) -> str:
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "negpy.infrastructure.scanners.dice_dual_source_runner",
        "--live",
        "--confirm-film-stationary",
        "--out-dir",
        str(output_dir),
    ]
    if main_dpi != ORACLE_MAIN_DPI:
        command.extend(("--main-dpi", str(main_dpi)))
    if transport != "roll":
        command.extend(("--transport", transport))
    if device_id is not None:
        command.extend(("--device", device_id))
    if frame is not None:
        command.extend(("--frame", str(frame)))
    if subframe_mm is not None:
        command.extend(("--subframe-mm", str(subframe_mm)))
    return shlex.join(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="perform both scans; default only prints the bounded plan")
    parser.add_argument(
        "--confirm-film-stationary",
        action="store_true",
        help="required with --live: confirms film is loaded, aligned, and no other scanner client is running",
    )
    parser.add_argument(
        "--transport",
        choices=("roll", "mounted"),
        default="roll",
        help="loaded holder type; mounted uses the MA-21 physical aperture",
    )
    parser.add_argument("--device", help="explicit SANE device id; default discovers exactly one coolscan3 device")
    parser.add_argument("--frame", type=int, help="optional roll-adapter frame; omit for a mounted holder")
    parser.add_argument("--subframe-mm", type=float, help="optional registered roll subframe; requires --frame")
    parser.add_argument(
        "--main-dpi",
        type=int,
        choices=(ORACLE_MAIN_DPI, NATIVE_MAIN_DPI),
        default=ORACLE_MAIN_DPI,
        help="main RGBI resolution; 500 reproduces the frozen oracle, 4000 captures the native product domain",
    )
    parser.add_argument("--out-dir", default="dice-dual-rgbi-results")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = DiceDualSourcePlan.for_main_dpi(
            args.main_dpi,
            frame=args.frame,
            subframe_mm=args.subframe_mm,
            transport=args.transport,
        )
    except ValueError as error:
        parser.error(str(error))
    next_command = exact_next_command(
        output_dir=args.out_dir,
        device_id=args.device,
        frame=args.frame,
        subframe_mm=args.subframe_mm,
        main_dpi=args.main_dpi,
        transport=args.transport,
    )
    if not args.live:
        print(json.dumps({"plan": plan.semantic_dict(), "next_command": next_command}, indent=2, sort_keys=True))
        return 0
    if not args.confirm_film_stationary:
        parser.error("--live requires --confirm-film-stationary")

    sane = Libsane()
    device: LibsaneRawDevice | None = None
    try:
        device_id = args.device or sane.discover_coolscan3()
        if not _strip_net_prefix(device_id).startswith("coolscan3:"):
            raise RuntimeError(f"refusing non-coolscan3 device {device_id!r}")
        device = sane.open(device_id)
        capture = acquire_dual_sources(device, plan)
        bundle = write_capture_bundle(args.out_dir, device_id=device_id, plan=plan, capture=capture)
    except KeyboardInterrupt:
        if device is not None:
            device.cancel()
        raise
    finally:
        if device is not None:
            device.close()
        sane.close()
    print(f"capture: {bundle}")
    print(f"manifest: {bundle / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
