"""Preview-derived registration policy for roll-fed film scanners."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import floor, isfinite
from typing import Mapping

import numpy as np

from negpy.infrastructure.scanners.frame_registration import (
    FilmBaseLearningConfig,
    FilmBaseModel,
    TailDetectorConfig,
    learn_film_base,
    measure_leading_margin,
    measure_target_start,
    measure_transport_tails,
    robust_pitch_prediction,
)
from negpy.infrastructure.scanners.params import RegisteredScanGeometry
from negpy.services.scanning.roll_service import (
    AlignmentVerification,
    FrameRegistration,
)


@dataclass(frozen=True)
class RollRegistrationConfig:
    """Geometry calibration shared by the preview and device-pixel grids."""

    preview_dpi: int = 400
    device_dpi: int = 4000
    retained_margin_mm: float = 0.57
    tail_guard_device_px: int = 20
    inclusive_endpoint_adjustment_px: int = 2
    alignment_tolerance_rows: int = 4
    maximum_edge_prediction_error_rows: float = 12.0
    excluded_frames: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        for field, value in (
            ("preview_dpi", self.preview_dpi),
            ("device_dpi", self.device_dpi),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        for field, value in (
            ("tail_guard_device_px", self.tail_guard_device_px),
            ("inclusive_endpoint_adjustment_px", self.inclusive_endpoint_adjustment_px),
            ("alignment_tolerance_rows", self.alignment_tolerance_rows),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        for field, value in (
            ("retained_margin_mm", self.retained_margin_mm),
            ("maximum_edge_prediction_error_rows", self.maximum_edge_prediction_error_rows),
        ):
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and non-negative")
        if type(self.excluded_frames) is not frozenset or any(
            type(frame) is not int or frame < 1 for frame in self.excluded_frames
        ):
            raise ValueError("excluded_frames must be a frozenset of positive integers")


def registered_scan_geometry(
    *,
    frame: int,
    target_start_row: int,
    tail_start_row: float,
    config: RollRegistrationConfig,
) -> RegisteredScanGeometry:
    """Convert measured preview bounds into coupled native scanner geometry."""

    if frame < 1 or target_start_row < 0:
        raise ValueError("frame must be positive and target row non-negative")
    if not isfinite(tail_start_row) or tail_start_row <= target_start_row:
        raise ValueError("tail boundary must be finite and follow the target edge")
    if config.preview_dpi <= 0 or config.device_dpi <= 0:
        raise ValueError("preview and device DPI must be positive")

    target_start_mm = target_start_row * 25.4 / config.preview_dpi
    subframe_mm = round(max(0.0, target_start_mm - config.retained_margin_mm), 2)
    subframe_device_px = floor(subframe_mm * config.device_dpi / 25.4)
    tail_device_px = floor(tail_start_row * config.device_dpi / config.preview_dpi + 0.5)
    br_y_device_px = tail_device_px - subframe_device_px - config.inclusive_endpoint_adjustment_px - config.tail_guard_device_px
    if br_y_device_px < 0:
        raise ValueError("registered scan window is empty")
    return RegisteredScanGeometry(
        frame=frame,
        subframe_mm=subframe_mm,
        br_y_device_px=br_y_device_px,
    )


class PreviewRollRegistration:
    """Jointly calibrate roll previews without assuming leader length or RGB."""

    def __init__(self, config: RollRegistrationConfig) -> None:
        self.config = config
        self._film_base: FilmBaseModel | None = None

    @property
    def preview_dpi(self) -> int:
        return self.config.preview_dpi

    @property
    def minimum_preview_count(self) -> int:
        return max(3, FilmBaseLearningConfig().minimum_preview_support)

    @property
    def registration_signature(self) -> dict[str, object]:
        config = asdict(self.config)
        config["excluded_frames"] = sorted(self.config.excluded_frames)
        return {
            "algorithm": "negpy.preview-roll-registration",
            "version": 1,
            "config": config,
            "film_base_learning": asdict(FilmBaseLearningConfig()),
            "tail_detection": asdict(TailDetectorConfig()),
            "tail_residual_tolerance_rows": 1.5,
            "target_edge_minimum_run_rows": 3,
            "leading_margin_minimum_run_rows": 3,
            "leading_margin_max_initial_outlier_rows": 1,
        }

    def calibrate(
        self, previews: Mapping[int, np.ndarray]
    ) -> dict[int, FrameRegistration]:
        if not previews:
            return {}
        ordered = {int(frame): np.asarray(image) for frame, image in previews.items()}
        if len(ordered) != len(previews):
            raise ValueError("preview frame positions must be unique integers")
        if len(ordered) < self.minimum_preview_count:
            raise ValueError(
                f"registration requires at least {self.minimum_preview_count} preview frames; got {len(ordered)}"
            )

        model = learn_film_base(list(ordered.values()))
        self._film_base = model
        edges = {frame: measure_target_start(image, model) for frame, image in ordered.items()}

        included_count = len(set(ordered) - set(self.config.excluded_frames))
        if included_count >= 3:
            observed = {frame: edge.row for frame, edge in edges.items()}
            refined = {}
            for frame, image in ordered.items():
                if frame in self.config.excluded_frames:
                    refined[frame] = edges[frame]
                    continue
                prediction = robust_pitch_prediction(
                    observed,
                    frame=frame,
                    excluded_frames=self.config.excluded_frames,
                )
                refined[frame] = measure_target_start(
                    image,
                    model,
                    expected_row=prediction,
                    maximum_prediction_error=(self.config.maximum_edge_prediction_error_rows),
                )
            edges = refined

        tails = measure_transport_tails(ordered)
        registrations: dict[int, FrameRegistration] = {}
        for frame in ordered:
            edge = edges[frame]
            tail = tails[frame]
            geometry = registered_scan_geometry(
                frame=frame,
                target_start_row=edge.row,
                tail_start_row=tail.tail_start_row,
                config=self.config,
            )
            confidence = "high" if edge.confidence == "high" and tail.confidence == "high" and tail.source == "direct" else "medium"
            registrations[frame] = FrameRegistration(
                frame=frame,
                target_start_row=edge.row,
                usable_tail_row=float(tail.tail_start_row),
                confidence=confidence,
                geometry=geometry,
            )
        return registrations

    def verify(
        self,
        frame: int,
        preview: np.ndarray,
        registration: FrameRegistration,
    ) -> AlignmentVerification:
        if frame != registration.frame:
            raise ValueError(f"verification frame mismatch: expected {registration.frame}, got {frame}")
        if self._film_base is None:
            raise RuntimeError("calibrate must run before aligned-preview verification")
        target_rows = round(self.config.retained_margin_mm * self.config.preview_dpi / 25.4)
        try:
            margin = measure_leading_margin(preview, self._film_base)
        except ValueError:
            return AlignmentVerification(
                leading_margin_rows=None,
                target_margin_rows=target_rows,
                tolerance_rows=self.config.alignment_tolerance_rows,
                confidence="unresolved",
            )
        return AlignmentVerification(
            leading_margin_rows=margin.row,
            target_margin_rows=target_rows,
            tolerance_rows=self.config.alignment_tolerance_rows,
            confidence=margin.confidence,
        )
