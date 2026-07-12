from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from negpy.infrastructure.scanners.dice_dual_source_runner import (
    DiceDualSourcePlan,
    OptionInfo,
    RawFrame,
    _decode_rgbi16,
    acquire_dual_sources,
    crop_oracle_candidate,
    exact_next_command,
    main,
    write_capture_bundle,
)


def _option(name: str, *, constraint=None, active: bool = True, settable: bool = True) -> OptionInfo:
    return OptionInfo(name=name, value_type=1, active=active, settable=settable, constraint=constraint)


@dataclass
class FakeRawDevice:
    prepass: np.ndarray
    main: np.ndarray
    device_id: str = "coolscan3:usb:test"
    values: dict[str, object] = field(default_factory=dict)
    writes: list[tuple[str, object]] = field(default_factory=list)
    reads: list[str] = field(default_factory=list)
    cancelled: bool = False
    closed: bool = False

    def __post_init__(self) -> None:
        self.values.update(
            {
                "focus": 0,
                "exposure": 1.0,
                "red_exposure": 1200.0,
                "green_exposure": 1200.0,
                "blue_exposure": 1000.0,
            }
        )
        names = (
            "depth",
            "resolution",
            "preview",
            "negative",
            "samples_per_scan",
            "infrared",
            "autofocus",
            "ae",
            "focus",
            "exposure",
            "red_exposure",
            "green_exposure",
            "blue_exposure",
            "tl_x",
            "tl_y",
            "br_x",
            "br_y",
            "frame_count",
        )
        self.option_map = {name: _option(name) for name in names}
        self.option_map["depth"] = _option("depth", constraint=(8, 16))
        self.option_map["resolution"] = _option(
            "resolution", constraint=(285, 500, 4000)
        )
        self.option_map["samples_per_scan"] = _option("samples_per_scan", constraint=(1, 2, 4, 8, 16))
        self.option_map["tl_x"] = OptionInfo("tl_x", 1, True, True, range_constraint=(0.0, 3945.0, 1.0))
        self.option_map["tl_y"] = OptionInfo("tl_y", 1, True, True, range_constraint=(0.0, 5958.0, 1.0))
        self.option_map["br_x"] = OptionInfo("br_x", 1, True, True, range_constraint=(0.0, 3945.0, 1.0))
        self.option_map["br_y"] = OptionInfo("br_y", 1, True, True, range_constraint=(0.0, 5958.0, 1.0))

    def options(self) -> dict[str, OptionInfo]:
        return dict(self.option_map)

    def set_option(self, name: str, value) -> None:
        self.values[name] = value
        self.writes.append((name, value))

    def get_option(self, name: str):
        return self.values[name]

    def read_rgbi(self, *, expected_shape: tuple[int, int], label: str) -> RawFrame:
        self.reads.append(label)
        if label == "prepass":
            self.values.update(
                {
                    "focus": 216,
                    "exposure": 1.0,
                    "red_exposure": 1370.0,
                    "green_exposure": 1290.0,
                    "blue_exposure": 1120.0,
                }
            )
            array = self.prepass
        else:
            array = self.main
        bpl = array.shape[1] * 4 * 2
        return RawFrame(rgbi=array, bytes_per_line=bpl, bytes_read=bpl * array.shape[0])

    def cancel(self) -> None:
        self.cancelled = True

    def close(self) -> None:
        self.closed = True


def _arrays(plan: DiceDualSourcePlan) -> tuple[np.ndarray, np.ndarray]:
    prepass = np.zeros((*plan.prepass_full_shape, 4), dtype=np.uint16)
    main_array = np.zeros((*plan.main_full_shape, 4), dtype=np.uint16)
    # Make edge/crop preservation visible without allocating random temporaries.
    prepass[:, :, 0] = np.arange(prepass.shape[0], dtype=np.uint16)[:, None]
    prepass[:, :, 3] = np.arange(prepass.shape[1], dtype=np.uint16)[None, :]
    main_array[:, :, 0] = np.arange(main_array.shape[0], dtype=np.uint16)[:, None]
    main_array[:, :, 3] = np.arange(main_array.shape[1], dtype=np.uint16)[None, :]
    return prepass, main_array


def test_oracle_profile_is_exact_integer_pitch_and_center_crop() -> None:
    plan = DiceDualSourcePlan()

    assert plan.prepass_full_shape == (425, 281)
    assert plan.main_full_shape == (744, 493)
    assert plan.crop_offsets("prepass") == (6, 0)
    assert plan.crop_offsets("main") == (34, 25)
    assert plan.prepass_target_shape == (413, 281)
    assert plan.main_target_shape == (676, 443)


def test_native_4000_profile_preserves_the_complete_optical_frame() -> None:
    plan = DiceDualSourcePlan.for_main_dpi(4000)

    assert plan.prepass_full_shape == (425, 281)
    assert plan.prepass_target_shape == (413, 281)
    assert plan.main_full_shape == (5959, 3946)
    assert plan.main_target_shape == (5959, 3946)
    assert plan.crop_offsets("main") == (0, 0)


def test_mounted_native_4000_profile_uses_the_holder_aperture() -> None:
    plan = DiceDualSourcePlan.for_main_dpi(4000, transport="mounted")

    assert plan.prepass_full_shape == (413, 281)
    assert plan.prepass_target_shape == (413, 281)
    assert plan.main_full_shape == (5782, 3946)
    assert plan.main_target_shape == (5782, 3946)
    assert plan.crop_offsets("prepass") == (0, 0)
    assert plan.crop_offsets("main") == (0, 0)
    assert plan.semantic_dict()["transport"] == "mounted"


def test_pure_oracle_crop_helper_has_exact_shapes_offsets_and_pixels() -> None:
    plan = DiceDualSourcePlan()
    prepass, main_array = _arrays(plan)

    prepass_candidate = crop_oracle_candidate(prepass, plan=plan, epoch="prepass")
    main_candidate = crop_oracle_candidate(main_array, plan=plan, epoch="main")

    assert prepass_candidate.shape == (413, 281, 4)
    assert main_candidate.shape == (676, 443, 4)
    assert np.array_equal(prepass_candidate, prepass[6:-6, :, :])
    assert np.array_equal(main_candidate, main_array[34:-34, 25:-25, :])
    assert prepass_candidate.flags.c_contiguous
    assert main_candidate.flags.c_contiguous


def test_oracle_crop_helper_rejects_a_python_sane_short_frame() -> None:
    plan = DiceDualSourcePlan()
    prepass, _ = _arrays(plan)

    with pytest.raises(ValueError, match="full RGBI shape"):
        crop_oracle_candidate(prepass[:-1], plan=plan, epoch="prepass")


@pytest.mark.parametrize("height", (413, 676))
def test_raw_decoder_preserves_the_final_rgbi_row_python_sane_loses(height: int) -> None:
    width = 3
    expected = np.arange(height * width * 4, dtype=np.uint16).reshape(height, width, 4)
    decoded = _decode_rgbi16(expected.tobytes(), width=width, height=height, bytes_per_line=width * 4 * 2)

    assert decoded.shape == expected.shape
    assert np.array_equal(decoded[-1], expected[-1])


def test_acquisition_is_one_handle_prepass_then_locked_main() -> None:
    plan = DiceDualSourcePlan()
    prepass, main_array = _arrays(plan)
    device = FakeRawDevice(prepass, main_array)

    capture = acquire_dual_sources(device, plan)

    assert device.reads == ["prepass", "main"]
    assert capture.assertions["all_passed"] is True
    assert capture.prepass_candidate.shape == (413, 281, 4)
    assert capture.main_candidate.shape == (676, 443, 4)
    assert np.array_equal(capture.prepass_candidate, prepass[6:-6])
    assert np.array_equal(capture.main_candidate, main_array[34:-34, 25:-25])
    assert capture.prepass_full.shape == (425, 281, 4)
    assert capture.main_full.shape == (744, 493, 4)
    assert capture.capture_state.focus_position == 216

    resolutions = [value for name, value in device.writes if name == "resolution"]
    assert resolutions == [285, 500]
    first_locked_write = device.writes.index(("autofocus", False))
    assert ("focus", 216) in device.writes[first_locked_write:]
    assert ("red_exposure", 1370.0) in device.writes[first_locked_write:]
    assert device.writes.count(("frame_count", 1)) == 1


def test_preflight_refuses_missing_raw_ir_before_any_option_write() -> None:
    plan = DiceDualSourcePlan()
    prepass, main_array = _arrays(plan)
    device = FakeRawDevice(prepass, main_array)
    del device.option_map["infrared"]

    with pytest.raises(RuntimeError, match="before scanner mutation.*infrared"):
        acquire_dual_sources(device, plan)

    assert device.writes == []
    assert device.reads == []


def test_bad_prepass_shape_refuses_before_main_scan() -> None:
    plan = DiceDualSourcePlan()
    prepass, main_array = _arrays(plan)
    device = FakeRawDevice(prepass[:-1], main_array)

    with pytest.raises(RuntimeError, match="prepass raw RGBI frame refused.*shape"):
        acquire_dual_sources(device, plan)

    assert device.reads == ["prepass"]


def test_capture_bundle_is_receipt_bound_and_arrays_round_trip(tmp_path: Path) -> None:
    plan = DiceDualSourcePlan()
    prepass, main_array = _arrays(plan)
    capture = acquire_dual_sources(FakeRawDevice(prepass, main_array), plan)

    bundle = write_capture_bundle(tmp_path, device_id="coolscan3:usb:test", plan=plan, capture=capture, run_id="fixture")

    manifest_path = bundle / "manifest.json"
    receipt = json.loads((bundle / "receipt.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert receipt["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert manifest["assertions"]["all_passed"] is True
    assert manifest["plan"]["prepass"]["center_crop_offset_yx"] == [6, 0]
    assert manifest["plan"]["main"]["center_crop_offset_yx"] == [34, 25]
    assert np.array_equal(np.load(bundle / "prepass_full.npy", allow_pickle=False), prepass)
    assert np.array_equal(np.load(bundle / "main_full.npy", allow_pickle=False), main_array)


def test_default_cli_is_scanner_free_and_prints_exact_live_command(capsys) -> None:
    assert main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["prepass"]["dpi"] == 285
    assert payload["plan"]["main"]["dpi"] == 500
    assert "--live --confirm-film-stationary" in payload["next_command"]


def test_4000_cli_is_scanner_free_and_prints_exact_live_command(capsys) -> None:
    assert main(["--main-dpi", "4000"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["prepass"]["dpi"] == 285
    assert payload["plan"]["main"]["dpi"] == 4000
    assert payload["plan"]["main"]["full_shape_hw"] == [5959, 3946]
    assert "--main-dpi 4000" in payload["next_command"]


def test_mounted_4000_cli_prints_holder_geometry_and_exact_command(capsys) -> None:
    assert main(["--main-dpi", "4000", "--transport", "mounted"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["transport"] == "mounted"
    assert payload["plan"]["prepass"]["full_shape_hw"] == [413, 281]
    assert payload["plan"]["main"]["full_shape_hw"] == [5782, 3946]
    assert "--transport mounted" in payload["next_command"]


def test_exact_command_can_bind_roll_registration() -> None:
    command = exact_next_command(
        output_dir="/tmp/dice",
        device_id="net:10.0.0.100:coolscan3:usb:test",
        frame=20,
        subframe_mm=6.35,
    )

    assert "--device net:10.0.0.100:coolscan3:usb:test" in command
    assert "--frame 20" in command
    assert "--subframe-mm 6.35" in command
