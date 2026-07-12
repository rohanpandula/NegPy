"""Roll-scanning workflow tests.

The service owns scanner sequencing and persistence.  Registration math is
injected so these tests can exercise the workflow without scanner hardware or
large image fixtures.
"""

from __future__ import annotations

import threading
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import tifffile

import negpy.services.scanning.roll_service as roll_service_module
from negpy.infrastructure.scanners.params import RegisteredScanGeometry, ScanParams
from negpy.infrastructure.scanners.result import ScanResult
from negpy.services.scanning.roll_service import (
    AlignmentVerification,
    FrameRegistration,
    RollScanService,
    RollStage,
)


class FakeScanner:
    def __init__(self) -> None:
        self.calls: list[ScanParams] = []
        self.after_scan = None

    def run_scan(self, device_id, params, progress, cancel):
        assert device_id == "coolscan3:test"
        assert not cancel.is_set()
        self.calls.append(params)
        dtype = np.uint8 if params.depth == 8 else np.uint16
        fill = (
            (100 if params.registered_geometry is None else 200)
            if dtype == np.uint8
            else (1000 if params.registered_geometry is None else 2000)
        )
        rgb = np.full((12, 16, 3), fill + int(params.frame or 0), dtype=dtype)
        ir = np.full((12, 16), 500 + int(params.frame or 0), dtype=np.uint16) if params.capture_ir else None
        result = ScanResult(rgb=rgb, ir=ir, dpi=params.dpi, device_model="Fake Coolscan")
        if self.after_scan is not None:
            self.after_scan(params)
        return result


@dataclass
class FakeRegistration:
    preview_dpi: int = 400
    minimum_preview_count: int = 1
    signature_version: int = 1
    measurement_shift: int = 0
    force_unresolved: bool = False
    calibrated_frames: tuple[int, ...] = ()
    calibrated_signatures: tuple[tuple[int, str, int], ...] = ()
    verification_signatures: tuple[tuple[int, str, int], ...] = ()

    @property
    def registration_signature(self):
        return {
            "algorithm": "fake-registration",
            "version": self.signature_version,
            "preview_dpi": self.preview_dpi,
        }

    def calibrate(self, previews):
        self.calibrated_frames = tuple(previews)
        self.calibrated_signatures = tuple(
            (frame, np.dtype(preview.dtype).name, int(preview.max()))
            for frame, preview in previews.items()
        )
        return {
            frame: FrameRegistration(
                frame=frame,
                target_start_row=8 + frame + self.measurement_shift,
                usable_tail_row=580 - frame,
                confidence="high",
                geometry=RegisteredScanGeometry(
                    frame=frame,
                    subframe_mm=1.25 + frame / 100 + self.measurement_shift / 100,
                    br_y_device_px=5000 - frame - self.measurement_shift,
                ),
            )
            for frame in previews
        }

    def verify(self, frame, preview, registration):
        self.verification_signatures += (
            (frame, np.dtype(preview.dtype).name, int(preview.max())),
        )
        expected_values = {2000 + frame, (200 + frame) * 257}
        assert int(preview.min()) == int(preview.max())
        assert int(preview.max()) in expected_values
        if self.force_unresolved:
            return AlignmentVerification(
                leading_margin_rows=None,
                target_margin_rows=9,
                tolerance_rows=4,
                confidence="unresolved",
            )
        return AlignmentVerification(
            leading_margin_rows=9,
            target_margin_rows=9,
            tolerance_rows=4,
            confidence="high",
        )


class UnresolvedRegistration(FakeRegistration):
    def verify(self, frame, preview, registration):
        return AlignmentVerification(
            leading_margin_rows=None,
            target_margin_rows=9,
            tolerance_rows=4,
            confidence="unresolved",
        )


def test_prepare_roll_acquires_wide_then_registered_previews_and_persists_review_plan(tmp_path: Path) -> None:
    scanner = FakeScanner()
    registration = FakeRegistration()
    service = RollScanService(scanner=scanner, registration=registration)
    params = ScanParams(dpi=400, depth=16, capture_ir=False, autofocus=False)

    plan = service.prepare_roll(
        device_id="coolscan3:test",
        frames=(2, 3),
        output_dir=tmp_path,
        preview_params=params,
        stop=threading.Event(),
    )

    assert registration.calibrated_frames == (2, 3)
    assert [(call.frame, call.registered_geometry is not None) for call in scanner.calls] == [
        (2, False),
        (3, False),
        (2, True),
        (3, True),
    ]
    assert plan.stage is RollStage.REVIEW_REQUIRED
    assert plan.approved is False
    assert [record.frame for record in plan.frames] == [2, 3]
    assert all(record.verification and record.verification.passed for record in plan.frames)
    assert (tmp_path / "roll-scan.json").is_file()
    assert (tmp_path / "wide" / "frame002.tif").is_file()
    assert (tmp_path / "aligned" / "frame003.tif").is_file()


def test_fine_scan_is_refused_until_aligned_previews_are_explicitly_approved(tmp_path: Path) -> None:
    scanner = FakeScanner()
    service = RollScanService(scanner=scanner, registration=FakeRegistration())
    preview_params = ScanParams(dpi=400, depth=16, capture_ir=False, autofocus=False)
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(3,),
        output_dir=tmp_path,
        preview_params=preview_params,
        stop=threading.Event(),
    )

    fine_params = ScanParams(dpi=4000, depth=16, capture_ir=True)
    with pytest.raises(RuntimeError, match="approve"):
        service.scan_fine(
            plan_path=tmp_path / "roll-scan.json",
            fine_params=fine_params,
            stop=threading.Event(),
        )

    assert len(scanner.calls) == 2  # wide + aligned only; no fine transfer started


def test_verified_aligned_previews_can_be_explicitly_approved(tmp_path: Path) -> None:
    service = RollScanService(scanner=FakeScanner(), registration=FakeRegistration())
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(2, 3),
        output_dir=tmp_path,
        preview_params=ScanParams(dpi=400, depth=16, capture_ir=False, autofocus=False),
        stop=threading.Event(),
    )

    plan = service.approve_roll(tmp_path / "roll-scan.json")

    assert plan.approved is True
    assert plan.stage is RollStage.APPROVED
    saved = json.loads((tmp_path / "roll-scan.json").read_text())
    assert saved["approved"] is True
    assert saved["stage"] == "approved"


def test_approval_requires_every_aligned_preview_to_still_be_inspectable(
    tmp_path: Path,
) -> None:
    service = RollScanService(scanner=FakeScanner(), registration=FakeRegistration())
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(3,),
        output_dir=tmp_path,
        preview_params=ScanParams(dpi=400, depth=16, capture_ir=False, autofocus=False),
        stop=threading.Event(),
    )
    (tmp_path / "aligned" / "frame003.tif").unlink()

    with pytest.raises(RuntimeError, match="aligned preview"):
        service.approve_roll(tmp_path / "roll-scan.json")


def test_resume_regenerates_a_missing_approved_preview_and_requires_review_again(
    tmp_path: Path,
) -> None:
    scanner = FakeScanner()
    service = RollScanService(scanner=scanner, registration=FakeRegistration())
    params = ScanParams(dpi=400, depth=16, capture_ir=False, autofocus=False)
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(3,),
        output_dir=tmp_path,
        preview_params=params,
        stop=threading.Event(),
    )
    service.approve_roll(tmp_path / "roll-scan.json")
    (tmp_path / "aligned" / "frame003.tif").unlink()
    calls_before_resume = len(scanner.calls)

    plan = service.prepare_roll(
        device_id="coolscan3:test",
        frames=(3,),
        output_dir=tmp_path,
        preview_params=params,
        stop=threading.Event(),
    )

    assert [(call.frame, call.registered_geometry is not None) for call in scanner.calls[calls_before_resume:]] == [
        (3, True),
    ]
    assert plan.stage is RollStage.REVIEW_REQUIRED
    assert plan.approved is False


def test_aligned_rescan_commit_failure_cannot_reuse_stale_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scanner = FakeScanner()
    registration = FakeRegistration()
    service = RollScanService(scanner=scanner, registration=registration)
    params = ScanParams(dpi=400, depth=16, capture_ir=False)
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(3,),
        output_dir=tmp_path,
        preview_params=params,
        stop=threading.Event(),
    )
    service.approve_roll(tmp_path / "roll-scan.json")
    (tmp_path / "aligned" / "frame003.tif").unlink()
    registration.force_unresolved = True
    original_atomic_write = roll_service_module._atomic_write_json

    def fail_when_new_verification_is_committed(path, payload):
        frames = payload.get("frames", ())
        verification = frames[0].get("verification") if frames else None
        if verification is not None and verification.get("confidence") == "unresolved":
            raise OSError("simulated verification checkpoint failure")
        original_atomic_write(path, payload)

    monkeypatch.setattr(
        roll_service_module,
        "_atomic_write_json",
        fail_when_new_verification_is_committed,
    )
    with pytest.raises(OSError, match="verification checkpoint failure"):
        service.prepare_roll(
            device_id="coolscan3:test",
            frames=(3,),
            output_dir=tmp_path,
            preview_params=params,
            stop=threading.Event(),
        )

    saved = json.loads((tmp_path / "roll-scan.json").read_text())
    assert saved["frames"][0]["aligned_path"] is None
    assert saved["frames"][0]["verification"] is None

    monkeypatch.setattr(roll_service_module, "_atomic_write_json", original_atomic_write)
    calls_before_resume = len(scanner.calls)
    review = service.prepare_roll(
        device_id="coolscan3:test",
        frames=(3,),
        output_dir=tmp_path,
        preview_params=params,
        stop=threading.Event(),
    )
    assert [call.frame for call in scanner.calls[calls_before_resume:]] == [3]
    assert review.frames[0].verification is not None
    assert review.frames[0].verification.confidence == "unresolved"


def test_fine_scan_refuses_when_an_approved_preview_is_no_longer_inspectable(
    tmp_path: Path,
) -> None:
    service = RollScanService(scanner=FakeScanner(), registration=FakeRegistration())
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(3,),
        output_dir=tmp_path,
        preview_params=ScanParams(dpi=400, depth=16, capture_ir=False),
        stop=threading.Event(),
    )
    service.approve_roll(tmp_path / "roll-scan.json")
    (tmp_path / "aligned" / "frame003.tif").unlink()

    with pytest.raises(RuntimeError, match="aligned preview"):
        service.scan_fine(
            plan_path=tmp_path / "roll-scan.json",
            fine_params=ScanParams(dpi=4000, depth=16, capture_ir=True),
            stop=threading.Event(),
        )


def test_fine_scan_can_target_frame_three_and_reuses_its_verified_geometry(tmp_path: Path) -> None:
    scanner = FakeScanner()
    service = RollScanService(scanner=scanner, registration=FakeRegistration())
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(2, 3),
        output_dir=tmp_path,
        preview_params=ScanParams(dpi=400, depth=16, capture_ir=False, autofocus=False),
        stop=threading.Event(),
    )
    service.approve_roll(tmp_path / "roll-scan.json")

    plan = service.scan_fine(
        plan_path=tmp_path / "roll-scan.json",
        fine_params=ScanParams(dpi=4000, depth=16, capture_ir=True),
        frames=(3,),
        stop=threading.Event(),
    )

    fine_call = scanner.calls[-1]
    assert fine_call.frame == 3
    assert fine_call.dpi == 4000
    assert fine_call.capture_ir is True
    assert fine_call.registered_geometry == RegisteredScanGeometry(frame=3, subframe_mm=1.28, br_y_device_px=4997)
    assert plan.stage is RollStage.APPROVED  # frame 2 can still be scanned later
    assert plan.frames[0].fine_path is None
    assert plan.frames[1].fine_path is not None
    fine_path = tmp_path / plan.frames[1].fine_path
    assert fine_path.is_file()
    assert fine_path.with_name(f"{fine_path.stem}_IR.tif").is_file()


def test_fine_scan_resume_skips_an_already_completed_frame(tmp_path: Path) -> None:
    scanner = FakeScanner()
    service = RollScanService(scanner=scanner, registration=FakeRegistration())
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(2, 3),
        output_dir=tmp_path,
        preview_params=ScanParams(dpi=400, depth=16, capture_ir=False, autofocus=False),
        stop=threading.Event(),
    )
    service.approve_roll(tmp_path / "roll-scan.json")
    fine_params = ScanParams(dpi=4000, depth=16, capture_ir=True)
    service.scan_fine(
        plan_path=tmp_path / "roll-scan.json",
        fine_params=fine_params,
        frames=(3,),
        stop=threading.Event(),
    )
    calls_before_resume = len(scanner.calls)

    plan = service.scan_fine(
        plan_path=tmp_path / "roll-scan.json",
        fine_params=fine_params,
        stop=threading.Event(),
    )

    resumed_calls = scanner.calls[calls_before_resume:]
    assert [call.frame for call in resumed_calls] == [2]
    assert plan.stage is RollStage.COMPLETE
    assert all(record.fine_path for record in plan.frames)


def test_fine_resume_rescans_when_multisampling_recipe_changes(tmp_path: Path) -> None:
    scanner = FakeScanner()
    service = RollScanService(scanner=scanner, registration=FakeRegistration())
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(3,),
        output_dir=tmp_path,
        preview_params=ScanParams(dpi=400, depth=16, capture_ir=False, autofocus=False),
        stop=threading.Event(),
    )
    service.approve_roll(tmp_path / "roll-scan.json")
    service.scan_fine(
        plan_path=tmp_path / "roll-scan.json",
        fine_params=ScanParams(dpi=4000, depth=16, capture_ir=True, samples_per_scan=1),
        stop=threading.Event(),
    )
    calls_before_change = len(scanner.calls)

    service.scan_fine(
        plan_path=tmp_path / "roll-scan.json",
        fine_params=ScanParams(dpi=4000, depth=16, capture_ir=True, samples_per_scan=16),
        stop=threading.Event(),
    )

    changed_calls = scanner.calls[calls_before_change:]
    assert [call.frame for call in changed_calls] == [3]
    assert changed_calls[0].samples_per_scan == 16


def test_mixed_fine_recipes_do_not_mark_the_roll_complete(tmp_path: Path) -> None:
    scanner = FakeScanner()
    service = RollScanService(scanner=scanner, registration=FakeRegistration())
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(2, 3),
        output_dir=tmp_path,
        preview_params=ScanParams(dpi=400, depth=16, capture_ir=False),
        stop=threading.Event(),
    )
    service.approve_roll(tmp_path / "roll-scan.json")
    service.scan_fine(
        plan_path=tmp_path / "roll-scan.json",
        fine_params=ScanParams(dpi=4000, depth=16, capture_ir=True, samples_per_scan=1),
        stop=threading.Event(),
    )

    mixed = service.scan_fine(
        plan_path=tmp_path / "roll-scan.json",
        fine_params=ScanParams(dpi=4000, depth=16, capture_ir=True, samples_per_scan=16),
        frames=(3,),
        stop=threading.Event(),
    )

    assert mixed.stage is RollStage.APPROVED
    assert mixed.frames[0].fine_recipe is not None
    assert mixed.frames[0].fine_recipe.samples_per_scan == 1
    assert mixed.frames[1].fine_recipe is not None
    assert mixed.frames[1].fine_recipe.samples_per_scan == 16


def test_invalid_preview_and_fine_recipes_fail_before_transfer(tmp_path: Path) -> None:
    scanner = FakeScanner()
    service = RollScanService(scanner=scanner, registration=FakeRegistration())

    with pytest.raises(ValueError, match="depth"):
        service.prepare_roll(
            device_id="coolscan3:test",
            frames=(3,),
            output_dir=tmp_path,
            preview_params=ScanParams(dpi=400, depth=12, capture_ir=False),
            stop=threading.Event(),
        )

    assert scanner.calls == []
    assert not (tmp_path / "roll-scan.json").exists()

    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(3,),
        output_dir=tmp_path,
        preview_params=ScanParams(dpi=400, depth=16, capture_ir=False),
        stop=threading.Event(),
    )
    service.approve_roll(tmp_path / "roll-scan.json")
    calls_before_fine = len(scanner.calls)
    with pytest.raises(ValueError, match="depth"):
        service.scan_fine(
            plan_path=tmp_path / "roll-scan.json",
            fine_params=ScanParams(dpi=4000, depth=12, capture_ir=True),
            stop=threading.Event(),
        )
    assert len(scanner.calls) == calls_before_fine


def test_prepare_demotes_complete_plan_when_a_fine_artifact_is_missing(tmp_path: Path) -> None:
    scanner = FakeScanner()
    service = RollScanService(scanner=scanner, registration=FakeRegistration())
    preview_params = ScanParams(dpi=400, depth=16, capture_ir=False)
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(3,),
        output_dir=tmp_path,
        preview_params=preview_params,
        stop=threading.Event(),
    )
    service.approve_roll(tmp_path / "roll-scan.json")
    completed = service.scan_fine(
        plan_path=tmp_path / "roll-scan.json",
        fine_params=ScanParams(dpi=4000, depth=16, capture_ir=True),
        stop=threading.Event(),
    )
    assert completed.frames[0].fine_path is not None
    (tmp_path / completed.frames[0].fine_path).unlink()
    calls_before_prepare = len(scanner.calls)

    resumed = service.prepare_roll(
        device_id="coolscan3:test",
        frames=(3,),
        output_dir=tmp_path,
        preview_params=preview_params,
        stop=threading.Event(),
    )

    assert resumed.stage is RollStage.APPROVED
    assert len(scanner.calls) == calls_before_prepare


def test_recipe_change_cannot_overwrite_the_artifact_named_by_the_old_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecipeMarkedScanner(FakeScanner):
        def run_scan(self, device_id, params, progress, cancel):
            result = super().run_scan(device_id, params, progress, cancel)
            if params.dpi == 4000:
                result.rgb.fill(params.samples_per_scan)
            return result

    scanner = RecipeMarkedScanner()
    service = RollScanService(scanner=scanner, registration=FakeRegistration())
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(3,),
        output_dir=tmp_path,
        preview_params=ScanParams(dpi=400, depth=16, capture_ir=False),
        stop=threading.Event(),
    )
    service.approve_roll(tmp_path / "roll-scan.json")
    original = service.scan_fine(
        plan_path=tmp_path / "roll-scan.json",
        fine_params=ScanParams(dpi=4000, depth=16, capture_ir=True, samples_per_scan=16),
        stop=threading.Event(),
    )
    assert original.frames[0].fine_path is not None
    original_path = tmp_path / original.frames[0].fine_path
    assert int(tifffile.imread(original_path)[0, 0, 0]) == 16

    monkeypatch.setattr(
        roll_service_module,
        "_atomic_write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("simulated plan commit failure")),
    )
    with pytest.raises(OSError, match="simulated plan commit failure"):
        service.scan_fine(
            plan_path=tmp_path / "roll-scan.json",
            fine_params=ScanParams(dpi=4000, depth=16, capture_ir=True, samples_per_scan=1),
            stop=threading.Event(),
        )

    assert int(tifffile.imread(original_path)[0, 0, 0]) == 16


def test_interrupted_complete_roll_recipe_upgrade_remains_resumable(tmp_path: Path) -> None:
    class FailOnceScanner(FakeScanner):
        fail_fine_frame: int | None = None

        def run_scan(self, device_id, params, progress, cancel):
            result = super().run_scan(device_id, params, progress, cancel)
            if params.dpi == 4000 and params.frame == self.fail_fine_frame:
                self.fail_fine_frame = None
                raise RuntimeError("simulated second-frame transfer failure")
            return result

    scanner = FailOnceScanner()
    service = RollScanService(scanner=scanner, registration=FakeRegistration())
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(2, 3),
        output_dir=tmp_path,
        preview_params=ScanParams(dpi=400, depth=16, capture_ir=False),
        stop=threading.Event(),
    )
    service.approve_roll(tmp_path / "roll-scan.json")
    service.scan_fine(
        plan_path=tmp_path / "roll-scan.json",
        fine_params=ScanParams(dpi=4000, depth=16, capture_ir=True, samples_per_scan=1),
        stop=threading.Event(),
    )
    scanner.fail_fine_frame = 3

    with pytest.raises(RuntimeError, match="second-frame transfer failure"):
        service.scan_fine(
            plan_path=tmp_path / "roll-scan.json",
            fine_params=ScanParams(dpi=4000, depth=16, capture_ir=True, samples_per_scan=16),
            stop=threading.Event(),
        )

    saved = json.loads((tmp_path / "roll-scan.json").read_text())
    assert saved["stage"] == "approved"
    calls_before_resume = len(scanner.calls)
    resumed = service.scan_fine(
        plan_path=tmp_path / "roll-scan.json",
        fine_params=ScanParams(dpi=4000, depth=16, capture_ir=True, samples_per_scan=16),
        stop=threading.Event(),
    )
    assert [call.frame for call in scanner.calls[calls_before_resume:]] == [3]
    assert resumed.stage is RollStage.COMPLETE


def test_new_registration_measurement_invalidates_an_old_fine_scan(tmp_path: Path) -> None:
    scanner = FakeScanner()
    registration = FakeRegistration()
    service = RollScanService(scanner=scanner, registration=registration)
    preview_params = ScanParams(dpi=400, depth=16, capture_ir=False)
    fine_params = ScanParams(dpi=4000, depth=16, capture_ir=True)
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(3,),
        output_dir=tmp_path,
        preview_params=preview_params,
        stop=threading.Event(),
    )
    service.approve_roll(tmp_path / "roll-scan.json")
    service.scan_fine(
        plan_path=tmp_path / "roll-scan.json",
        fine_params=fine_params,
        stop=threading.Event(),
    )
    registration.measurement_shift = 1
    (tmp_path / "aligned" / "frame003.tif").unlink()

    review = service.prepare_roll(
        device_id="coolscan3:test",
        frames=(3,),
        output_dir=tmp_path,
        preview_params=preview_params,
        stop=threading.Event(),
    )

    assert review.frames[0].fine_path is None
    service.approve_roll(tmp_path / "roll-scan.json")
    calls_before_fine = len(scanner.calls)
    service.scan_fine(
        plan_path=tmp_path / "roll-scan.json",
        fine_params=fine_params,
        stop=threading.Event(),
    )
    assert [call.frame for call in scanner.calls[calls_before_fine:]] == [3]
    assert scanner.calls[-1].registered_geometry == RegisteredScanGeometry(
        frame=3,
        subframe_mm=1.29,
        br_y_device_px=4996,
    )


def test_resume_rejects_a_valid_tiff_with_the_wrong_saved_dimensions(tmp_path: Path) -> None:
    scanner = FakeScanner()
    service = RollScanService(scanner=scanner, registration=FakeRegistration())
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(3,),
        output_dir=tmp_path,
        preview_params=ScanParams(dpi=400, depth=16, capture_ir=False, autofocus=False),
        stop=threading.Event(),
    )
    service.approve_roll(tmp_path / "roll-scan.json")
    fine_params = ScanParams(dpi=4000, depth=16, capture_ir=True)
    completed = service.scan_fine(
        plan_path=tmp_path / "roll-scan.json",
        fine_params=fine_params,
        stop=threading.Event(),
    )
    assert completed.frames[0].fine_path is not None
    output = tmp_path / completed.frames[0].fine_path
    tifffile.imwrite(output, np.zeros((3, 4, 3), dtype=np.uint16), photometric="rgb")
    calls_before_resume = len(scanner.calls)

    service.scan_fine(
        plan_path=tmp_path / "roll-scan.json",
        fine_params=fine_params,
        frames=(3,),
        stop=threading.Event(),
    )

    assert [call.frame for call in scanner.calls[calls_before_resume:]] == [3]
    assert tifffile.imread(output).shape == (12, 16, 3)


def test_prepare_resume_keeps_a_finished_transfer_and_stops_before_the_next_one(tmp_path: Path) -> None:
    scanner = FakeScanner()
    service = RollScanService(scanner=scanner, registration=FakeRegistration())
    stop = threading.Event()
    scanner.after_scan = lambda _params: stop.set()
    preview_params = ScanParams(dpi=400, depth=16, capture_ir=False, autofocus=False)

    partial = service.prepare_roll(
        device_id="coolscan3:test",
        frames=(2, 3),
        output_dir=tmp_path,
        preview_params=preview_params,
        stop=stop,
    )

    assert partial.stage is RollStage.WIDE_PREVIEW
    assert [call.frame for call in scanner.calls] == [2]
    assert (tmp_path / "wide" / "frame002.tif").is_file()
    assert (tmp_path / "roll-scan.json").is_file()

    stop.clear()
    scanner.after_scan = None
    calls_before_resume = len(scanner.calls)
    completed = service.prepare_roll(
        device_id="coolscan3:test",
        frames=(2, 3),
        output_dir=tmp_path,
        preview_params=preview_params,
        stop=stop,
    )

    resumed = scanner.calls[calls_before_resume:]
    assert [(call.frame, call.registered_geometry is not None) for call in resumed] == [
        (3, False),
        (2, True),
        (3, True),
    ]
    assert completed.stage is RollStage.REVIEW_REQUIRED


def test_eight_bit_preview_resume_uses_committed_tiff_metadata(tmp_path: Path) -> None:
    scanner = FakeScanner()
    service = RollScanService(scanner=scanner, registration=FakeRegistration())
    stop = threading.Event()
    scanner.after_scan = lambda _params: stop.set()
    preview_params = ScanParams(dpi=400, depth=8, capture_ir=False, autofocus=False)
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(2, 3),
        output_dir=tmp_path,
        preview_params=preview_params,
        stop=stop,
    )

    stop.clear()
    scanner.after_scan = None
    calls_before_resume = len(scanner.calls)
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(2, 3),
        output_dir=tmp_path,
        preview_params=preview_params,
        stop=stop,
    )

    resumed = scanner.calls[calls_before_resume:]
    assert [(call.frame, call.registered_geometry is not None) for call in resumed] == [
        (3, False),
        (2, True),
        (3, True),
    ]
    assert service._registration.calibrated_signatures == (  # type: ignore[attr-defined]
        (2, "uint16", 102 * 257),
        (3, "uint16", 103 * 257),
    )
    assert service._registration.verification_signatures[-2:] == (  # type: ignore[attr-defined]
        (2, "uint16", 202 * 257),
        (3, "uint16", 203 * 257),
    )


def test_human_can_explicitly_approve_a_visually_checked_unresolved_frame(tmp_path: Path) -> None:
    service = RollScanService(scanner=FakeScanner(), registration=UnresolvedRegistration())
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(3,),
        output_dir=tmp_path,
        preview_params=ScanParams(dpi=400, depth=16, capture_ir=False),
        stop=threading.Event(),
    )
    plan_path = tmp_path / "roll-scan.json"

    with pytest.raises(RuntimeError, match="verification failed"):
        service.approve_roll(plan_path)

    approved = service.approve_roll(
        plan_path,
        allow_unverified_frames=(3,),
    )
    assert approved.approved
    assert approved.stage is RollStage.APPROVED
    assert approved.visual_override_frames == (3,)
    saved = json.loads(plan_path.read_text())
    assert saved["visual_override_frames"] == [3]


def test_preview_dpi_must_match_the_registration_coordinate_system(tmp_path: Path) -> None:
    scanner = FakeScanner()
    service = RollScanService(
        scanner=scanner,
        registration=FakeRegistration(preview_dpi=1000),
    )

    with pytest.raises(ValueError, match="preview DPI"):
        service.prepare_roll(
            device_id="coolscan3:test",
            frames=(2, 3),
            output_dir=tmp_path,
            preview_params=ScanParams(dpi=400, depth=16, capture_ir=False, autofocus=False),
            stop=threading.Event(),
        )

    assert scanner.calls == []


def test_preview_resume_rejects_a_quality_recipe_change_before_scanning(tmp_path: Path) -> None:
    scanner = FakeScanner()
    service = RollScanService(scanner=scanner, registration=FakeRegistration())
    stop = threading.Event()
    scanner.after_scan = lambda _params: stop.set()
    original = ScanParams(
        dpi=400,
        depth=16,
        capture_ir=False,
        autofocus=False,
        samples_per_scan=1,
        auto_exposure=False,
    )
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(2, 3),
        output_dir=tmp_path,
        preview_params=original,
        stop=stop,
    )
    calls_before_resume = len(scanner.calls)

    with pytest.raises(ValueError, match="preview recipe"):
        service.prepare_roll(
            device_id="coolscan3:test",
            frames=(2, 3),
            output_dir=tmp_path,
            preview_params=ScanParams(
                dpi=400,
                depth=16,
                capture_ir=False,
                autofocus=True,
                samples_per_scan=1,
                auto_exposure=False,
            ),
            stop=threading.Event(),
        )

    assert len(scanner.calls) == calls_before_resume


def test_preview_resume_rejects_a_registration_policy_change_before_scanning(tmp_path: Path) -> None:
    scanner = FakeScanner()
    stop = threading.Event()
    scanner.after_scan = lambda _params: stop.set()
    RollScanService(scanner=scanner, registration=FakeRegistration(signature_version=1)).prepare_roll(
        device_id="coolscan3:test",
        frames=(2, 3),
        output_dir=tmp_path,
        preview_params=ScanParams(dpi=400, depth=16, capture_ir=False),
        stop=stop,
    )
    calls_before_resume = len(scanner.calls)

    with pytest.raises(ValueError, match="registration policy"):
        RollScanService(scanner=scanner, registration=FakeRegistration(signature_version=2)).prepare_roll(
            device_id="coolscan3:test",
            frames=(2, 3),
            output_dir=tmp_path,
            preview_params=ScanParams(dpi=400, depth=16, capture_ir=False),
            stop=threading.Event(),
        )

    assert len(scanner.calls) == calls_before_resume


def test_approval_and_fine_scan_reject_a_changed_registration_policy(tmp_path: Path) -> None:
    scanner = FakeScanner()
    original = RollScanService(
        scanner=scanner,
        registration=FakeRegistration(signature_version=1),
    )
    original.prepare_roll(
        device_id="coolscan3:test",
        frames=(3,),
        output_dir=tmp_path,
        preview_params=ScanParams(dpi=400, depth=16, capture_ir=False),
        stop=threading.Event(),
    )
    changed = RollScanService(
        scanner=scanner,
        registration=FakeRegistration(signature_version=2),
    )
    calls_before_checks = len(scanner.calls)

    with pytest.raises(ValueError, match="registration policy"):
        changed.approve_roll(tmp_path / "roll-scan.json")

    original.approve_roll(tmp_path / "roll-scan.json")
    with pytest.raises(ValueError, match="registration policy"):
        changed.scan_fine(
            plan_path=tmp_path / "roll-scan.json",
            fine_params=ScanParams(dpi=4000, depth=16, capture_ir=True),
            stop=threading.Event(),
        )

    assert len(scanner.calls) == calls_before_checks


def test_too_few_frames_for_registration_are_rejected_before_scanner_movement(tmp_path: Path) -> None:
    scanner = FakeScanner()
    service = RollScanService(
        scanner=scanner,
        registration=FakeRegistration(minimum_preview_count=3),
    )

    with pytest.raises(ValueError, match="at least 3"):
        service.prepare_roll(
            device_id="coolscan3:test",
            frames=(2, 3),
            output_dir=tmp_path,
            preview_params=ScanParams(dpi=400, depth=16, capture_ir=False),
            stop=threading.Event(),
        )

    assert scanner.calls == []
    assert not (tmp_path / "roll-scan.json").exists()


def test_non_integer_frame_selection_is_rejected_before_scanner_movement(tmp_path: Path) -> None:
    scanner = FakeScanner()
    service = RollScanService(scanner=scanner, registration=FakeRegistration())

    with pytest.raises(ValueError, match="positive integers"):
        service.prepare_roll(
            device_id="coolscan3:test",
            frames=(True, 2, 3),
            output_dir=tmp_path,
            preview_params=ScanParams(dpi=400, depth=16, capture_ir=False),
            stop=threading.Event(),
        )

    assert scanner.calls == []
    assert not (tmp_path / "roll-scan.json").exists()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("approved", "false"),
        ("wide_path", "../outside.tif"),
        ("wide_path", r"..\outside.tif"),
    ],
)
def test_malformed_saved_plan_is_rejected_without_scanning(
    tmp_path: Path,
    field: str,
    bad_value: object,
) -> None:
    scanner = FakeScanner()
    stop = threading.Event()
    scanner.after_scan = lambda _params: stop.set()
    params = ScanParams(dpi=400, depth=16, capture_ir=False)
    service = RollScanService(scanner=scanner, registration=FakeRegistration())
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(2, 3),
        output_dir=tmp_path,
        preview_params=params,
        stop=stop,
    )
    plan_path = tmp_path / "roll-scan.json"
    payload = json.loads(plan_path.read_text())
    if field == "approved":
        payload[field] = bad_value
    else:
        payload["frames"][0][field] = bad_value
    plan_path.write_text(json.dumps(payload))
    calls_before_resume = len(scanner.calls)

    with pytest.raises(ValueError, match="malformed roll plan"):
        service.prepare_roll(
            device_id="coolscan3:test",
            frames=(2, 3),
            output_dir=tmp_path,
            preview_params=params,
            stop=threading.Event(),
        )

    assert len(scanner.calls) == calls_before_resume


def test_complete_plan_with_mixed_fine_recipes_is_rejected(tmp_path: Path) -> None:
    scanner = FakeScanner()
    service = RollScanService(scanner=scanner, registration=FakeRegistration())
    preview_params = ScanParams(dpi=400, depth=16, capture_ir=False)
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(2, 3),
        output_dir=tmp_path,
        preview_params=preview_params,
        stop=threading.Event(),
    )
    service.approve_roll(tmp_path / "roll-scan.json")
    service.scan_fine(
        plan_path=tmp_path / "roll-scan.json",
        fine_params=ScanParams(dpi=4000, depth=16, capture_ir=True),
        stop=threading.Event(),
    )
    plan_path = tmp_path / "roll-scan.json"
    payload = json.loads(plan_path.read_text())
    payload["frames"][1]["fine_recipe"]["samples_per_scan"] = 16
    plan_path.write_text(json.dumps(payload))
    calls_before_resume = len(scanner.calls)

    with pytest.raises(ValueError, match="mixed or missing fine recipes"):
        service.prepare_roll(
            device_id="coolscan3:test",
            frames=(2, 3),
            output_dir=tmp_path,
            preview_params=preview_params,
            stop=threading.Event(),
        )

    assert len(scanner.calls) == calls_before_resume


def test_approved_unverified_frame_requires_a_persisted_visual_override(tmp_path: Path) -> None:
    service = RollScanService(scanner=FakeScanner(), registration=UnresolvedRegistration())
    service.prepare_roll(
        device_id="coolscan3:test",
        frames=(3,),
        output_dir=tmp_path,
        preview_params=ScanParams(dpi=400, depth=16, capture_ir=False),
        stop=threading.Event(),
    )
    plan_path = tmp_path / "roll-scan.json"
    service.approve_roll(plan_path, allow_unverified_frames=(3,))
    payload = json.loads(plan_path.read_text())
    payload["visual_override_frames"] = []
    plan_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="without visual overrides"):
        service.scan_fine(
            plan_path=plan_path,
            fine_params=ScanParams(dpi=4000, depth=16, capture_ir=True),
            stop=threading.Event(),
        )
