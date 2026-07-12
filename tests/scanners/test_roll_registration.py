"""Scanner-independent registration policy and geometry tests."""

import math
from pathlib import Path

import numpy as np
import pytest

from negpy.infrastructure.scanners.params import RegisteredScanGeometry
from negpy.services.scanning.roll_registration import (
    PreviewRollRegistration,
    RollRegistrationConfig,
    registered_scan_geometry,
)


FIXTURE = Path(__file__).with_name("fixtures") / "ls5000_roll_registration_frame03.npz"


def test_measured_frame_three_edge_and_tail_reproduce_the_verified_scan_geometry() -> None:
    geometry = registered_scan_geometry(
        frame=3,
        target_start_row=109,
        tail_start_row=602.5,
        config=RollRegistrationConfig(),
    )
    assert geometry == RegisteredScanGeometry(
        frame=3,
        subframe_mm=6.35,
        br_y_device_px=5003,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"preview_dpi": 0},
        {"device_dpi": True},
        {"retained_margin_mm": -0.01},
        {"retained_margin_mm": math.inf},
        {"tail_guard_device_px": -1},
        {"inclusive_endpoint_adjustment_px": -1},
        {"alignment_tolerance_rows": -1},
        {"maximum_edge_prediction_error_rows": math.nan},
        {"excluded_frames": frozenset({0})},
    ],
)
def test_registration_config_rejects_impossible_values(kwargs) -> None:
    with pytest.raises(ValueError):
        RollRegistrationConfig(**kwargs)


def test_real_policy_declares_and_defensively_checks_three_frame_minimum() -> None:
    policy = PreviewRollRegistration(RollRegistrationConfig())
    previews = {
        2: np.zeros((12, 16, 3), dtype=np.uint16),
        3: np.zeros((12, 16, 3), dtype=np.uint16),
    }

    assert policy.minimum_preview_count == 3
    with pytest.raises(ValueError, match="at least 3"):
        policy.calibrate(previews)


def test_policy_replays_the_compact_hardware_sweep_and_verifies_frame_three() -> None:
    with np.load(FIXTURE) as fixture:
        frames = tuple(int(frame) for frame in fixture["frames"])
        previews = {frame: fixture[f"wide_{frame:02d}"].copy() for frame in frames}
        aligned_frame_three = fixture["aligned_03"].copy()
    policy = PreviewRollRegistration(
        RollRegistrationConfig(excluded_frames=frozenset({1}))
    )

    registrations = policy.calibrate(previews)
    frame_three = registrations[3]
    verification = policy.verify(3, aligned_frame_three, frame_three)

    assert frames == (1, 2, 3, 8, 20, 21, 24)
    assert frame_three.target_start_row == 107
    assert frame_three.usable_tail_row == 602.5
    assert frame_three.geometry == RegisteredScanGeometry(
        frame=3,
        subframe_mm=6.22,
        br_y_device_px=5024,
    )
    assert {frame: registrations[frame].usable_tail_row for frame in (8, 20, 21, 24)} == {
        8: 590.0,
        20: 560.0,
        21: 558.0,
        24: 550.0,
    }
    assert verification.passed
    assert verification.leading_margin_rows == 7


def _wide_preview(*, edge: int, tail: int, seed: int) -> np.ndarray:
    height, width = 72, 32
    rng = np.random.default_rng(seed)
    image = np.empty((height, width, 3), dtype=np.float64)
    base = np.asarray((42000, 18000, 9500))
    image[:edge] = base + rng.normal(0, 6, size=(edge, width, 3))
    y, x = np.mgrid[: tail - edge, :width]
    scene = np.asarray((12000, 8000, 4500))
    texture = 800 * np.sin(x / 2.5) + 350 * np.cos(y / 3.0)
    image[edge:tail] = scene + texture[..., None] + rng.normal(0, 20, size=(tail - edge, width, 3))
    image[tail - 4 : tail] -= 5000
    image[tail:] = np.asarray((35000, 30000, 24000)) + 500 * np.sin(np.arange(width) / 2.5)[:, None]
    return np.clip(image, 0, 65535).astype(np.uint16)


def _aligned_preview(*, margin: int, seed: int) -> np.ndarray:
    image = _wide_preview(edge=margin, tail=60, seed=seed)
    return image[:52].copy()  # shortened before the synthetic transport tail


def test_policy_jointly_measures_edges_and_transport_tails_then_verifies_margin() -> None:
    frames = range(10, 16)
    edges = {frame: 10 + frame % 3 for frame in frames}
    tails = {frame: 80 - 2 * frame for frame in frames}
    previews = {frame: _wide_preview(edge=edges[frame], tail=tails[frame], seed=frame) for frame in frames}
    policy = PreviewRollRegistration(RollRegistrationConfig())

    registrations = policy.calibrate(previews)

    assert set(registrations) == set(frames)
    for frame in frames:
        registration = registrations[frame]
        assert registration.target_start_row == edges[frame]
        assert registration.usable_tail_row == tails[frame]
        assert registration.geometry.frame == frame

    verification = policy.verify(
        12,
        _aligned_preview(margin=9, seed=50),
        registrations[12],
    )
    assert verification.passed
    assert verification.leading_margin_rows == 9


def test_excluded_cut_frame_keeps_its_local_edge_instead_of_normal_pitch_prior() -> None:
    previews = {
        frame: _wide_preview(
            edge=30 if frame == 1 else 10,
            tail=65 - 2 * frame,
            seed=70 + frame,
        )
        for frame in range(1, 7)
    }
    policy = PreviewRollRegistration(RollRegistrationConfig(excluded_frames=frozenset({1})))

    registrations = policy.calibrate(previews)

    assert registrations[1].target_start_row == 30
    assert all(registrations[frame].target_start_row == 10 for frame in range(2, 7))


def test_uncertain_aligned_preview_is_recorded_for_human_review_instead_of_aborting() -> None:
    previews = {frame: _wide_preview(edge=10, tail=80 - 2 * frame, seed=90 + frame) for frame in range(10, 16)}
    policy = PreviewRollRegistration(RollRegistrationConfig())
    registrations = policy.calibrate(previews)
    no_visible_film_base = np.full((52, 32, 3), 7000, dtype=np.uint16)

    verification = policy.verify(10, no_visible_film_base, registrations[10])

    assert verification.leading_margin_rows is None
    assert verification.confidence == "unresolved"
    assert not verification.passed
