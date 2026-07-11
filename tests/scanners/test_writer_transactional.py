"""Transactional-write tests for write_tiff_16bit: RGB+IR must land as a unit.

A failure writing the IR sidecar after the RGB payload already succeeded must
not leave an orphan RGB file with no indication that its IR channel is
missing. IR exists for dust detection (see TiffLoader._read_sidecar_ir); a
silently-orphaned RGB later paired with an unrelated, stale `_IR.tif`
sidecar would misattribute someone else's dust map to this frame — purely by
filename convention, with no shape/content cross-check.
"""

import os
import tempfile

import numpy as np
import pytest
import tifffile

from negpy.infrastructure.scanners.result import ScanResult
from negpy.services.scanning.writer import write_tiff_16bit


def _result(with_ir: bool = True) -> ScanResult:
    rgb = np.random.randint(0, 65535, (20, 30, 3), dtype=np.uint16)
    ir = np.random.randint(0, 65535, (20, 30), dtype=np.uint16) if with_ir else None
    return ScanResult(rgb=rgb, ir=ir, dpi=1200, device_model="TestScanner")


class TestIrFailureAfterRgbSucceeds:
    """monkeypatch tifffile.imwrite to fail on the 2nd call (the IR write),
    simulating a disk/codec failure after the RGB payload already succeeded."""

    def test_no_orphan_rgb_left_when_ir_write_fails(self, monkeypatch) -> None:
        real_imwrite = tifffile.imwrite
        calls = {"n": 0}

        def flaky_imwrite(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("simulated disk failure writing IR sidecar")
            return real_imwrite(*args, **kwargs)

        monkeypatch.setattr(tifffile, "imwrite", flaky_imwrite)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "frame001")
            result = _result(with_ir=True)

            with pytest.raises(Exception):
                write_tiff_16bit(result, path)

            # Archival invariant: no silent RGB-without-IR. The API already
            # surfaces a clear failure (the raise above); on top of that, no
            # orphan RGB may be left looking like a complete, valid scan.
            rgb_path = path + ".tif"
            ir_path = path + "_IR.tif"
            assert not os.path.exists(ir_path)
            assert not os.path.exists(rgb_path), (
                "IR write failed after RGB succeeded but the orphan RGB file was left on disk with no indication its IR channel is missing"
            )
            # No temp files left behind either.
            assert os.listdir(tmpdir) == []

    def test_rgb_write_failure_leaves_nothing(self, monkeypatch) -> None:
        """Sanity companion: a failure on the *first* (RGB) write must also
        leave no partial file — the baseline the IR-failure case above is
        held to."""

        def failing_imwrite(*args, **kwargs):
            raise OSError("simulated disk failure writing RGB")

        monkeypatch.setattr(tifffile, "imwrite", failing_imwrite)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "frame002")
            result = _result(with_ir=True)

            with pytest.raises(Exception):
                write_tiff_16bit(result, path)

            assert os.listdir(tmpdir) == []


class TestStaleIrSidecar:
    """A stale `_IR.tif` from a previous IR-enabled write of the same target
    must not be silently left next to a fresh IR-less RGB write."""

    def test_stale_sidecar_removed_when_new_write_has_no_ir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "frame003")
            rgb_path = path + ".tif"
            ir_path = path + "_IR.tif"

            # Leftover IR sidecar from an earlier, unrelated write.
            stale_ir = np.random.randint(0, 65535, (99, 77), dtype=np.uint16)
            tifffile.imwrite(ir_path, stale_ir, photometric="minisblack")
            assert os.path.exists(ir_path)

            result = _result(with_ir=False)
            returned_path = write_tiff_16bit(result, path)

            assert returned_path == rgb_path
            assert os.path.exists(rgb_path)
            assert not os.path.exists(ir_path), (
                "stale _IR.tif sidecar survived a fresh IR-less write of the same target — a downstream loader would misattribute it"
            )

    def test_fresh_ir_overwrites_stale_sidecar(self) -> None:
        """When the new write DOES have IR, it should simply overwrite the
        stale sidecar with its own (correctly paired) data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "frame004")
            ir_path = path + "_IR.tif"

            stale_ir = np.zeros((5, 5), dtype=np.uint16)
            tifffile.imwrite(ir_path, stale_ir, photometric="minisblack")

            result = _result(with_ir=True)
            write_tiff_16bit(result, path)

            readback = tifffile.imread(ir_path)
            assert readback.shape == result.ir.shape
            assert np.array_equal(readback, result.ir)
