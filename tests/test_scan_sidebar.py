"""Offline tests for the Scan sidebar's archival-recipe controls.

Constructs the real ScanSidebar + AppController under QT_QPA_PLATFORM=offscreen
(set globally by tests/conftest.py) with a mocked DesktopSessionManager and a
fabricated ScannerDevice/ScannerCapabilities -- no live SANE device is ever
opened. This proves the new controls (frame selection, hardware AE, the
RGB4x+IR1x archival split-capture toggle, registered geometry) instantiate,
capability-gate, and wire their Qt signals correctly.

It does NOT and cannot verify that the rendered UI looks/behaves right on a
real device -- that remains an unverified gap requiring a live app + hardware.
"""

import gc
import json
import sys
import unittest
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QApplication

from negpy.desktop.controller import AppController
from negpy.desktop.session import DesktopSessionManager, AppState
from negpy.desktop.view.sidebar.scan import ScanSidebar
from negpy.infrastructure.scanners.base import ScannerCapabilities, ScannerDevice
from negpy.infrastructure.scanners.params import ScanMode
from negpy.services.rendering.preview_manager import PreviewManager

if not QApplication.instance():
    _app = QApplication(sys.argv)


# A device exposing every capability the new controls gate on: frame
# selection, hardware auto-exposure, registered geometry, and both IR +
# multi-sampling (needed together for the archival split-capture toggle).
FULL_CAPS = ScannerCapabilities(
    ir_channel=True,
    supported_dpi=(1000, 4000),
    supported_depths=(16,),
    sources=(ScanMode.NEGATIVE,),
    max_area_mm=(36.0, 24.0),
    multi_sample=True,
    adapter_frame_capacity=40,
    auto_exposure=True,
    registered_geometry=True,
)
FULL_DEVICE = ScannerDevice(id="coolscan3:usb:libusb:001:007", vendor="Nikon", model="LS-5000", capabilities=FULL_CAPS)

# A device with none of the archival extras (only the baseline fields every
# existing control already handled).
MINIMAL_CAPS = ScannerCapabilities(
    ir_channel=False,
    supported_dpi=(1200, 2400),
    supported_depths=(8, 16),
    sources=(ScanMode.NEGATIVE,),
    max_area_mm=(36.0, 24.0),
)
MINIMAL_DEVICE = ScannerDevice(id="plustek:libusb:001:008", vendor="Plustek", model="OpticFilm", capabilities=MINIMAL_CAPS)


def _build_controller() -> AppController:
    mock_session_manager = MagicMock(spec=DesktopSessionManager)
    mock_session_manager.state = AppState()
    mock_session_manager.repo = MagicMock()

    with (
        patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
        patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
    ):
        mock_rw_class.return_value = MagicMock()
        mock_pm_class.return_value = MagicMock(spec=PreviewManager)
        mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
        controller = AppController(mock_session_manager)
    return controller


def _stop_threads(controller: AppController) -> None:
    for thread in [
        controller.render_thread,
        controller.export_thread,
        controller.thumb_thread,
        controller.norm_thread,
        controller.discovery_thread,
        controller.preview_load_thread,
        controller.scan_thread,
    ]:
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait()
    del controller
    gc.collect()


def _select_device(sidebar: ScanSidebar, device: ScannerDevice) -> None:
    """Populate the sidebar as if devices_ready delivered exactly `device`,
    without going through the (mocked-away) scan worker thread."""
    sidebar._devices = [device]
    sidebar.device_combo.blockSignals(True)
    sidebar.device_combo.clear()
    sidebar.device_combo.addItem(device.model, device.id)
    sidebar.device_combo.setCurrentIndex(0)
    sidebar.device_combo.blockSignals(False)
    sidebar._update_device_caps()


class ScanSidebarTestCase(unittest.TestCase):
    """Base class: real AppController + real ScanSidebar, offscreen, no device I/O."""

    def setUp(self):
        self.controller = _build_controller()
        self.sidebar = ScanSidebar(self.controller)

    def tearDown(self):
        del self.sidebar
        _stop_threads(self.controller)


class TestNewControlsInstantiate(ScanSidebarTestCase):
    def test_new_widgets_exist(self):
        for name in (
            "frame_spin",
            "ae_check",
            "archival_split_check",
            "registered_geometry_check",
            "subframe_spin",
            "br_y_spin",
            "load_registration_btn",
        ):
            self.assertTrue(hasattr(self.sidebar, name), f"missing widget: {name}")

    def test_new_controls_disabled_with_no_device_selected(self):
        # Fresh widgets default to Qt's enabled=True until something actually
        # runs the no-device gating -- exactly like the pre-existing
        # dpi/depth/ir/samples controls, none of which are disabled by
        # __init__ alone either. _update_device_caps() is what a real
        # "no device" state (e.g. _on_device_changed on the placeholder item)
        # actually triggers.
        self.sidebar._update_device_caps()
        self.assertFalse(self.sidebar.frame_spin.isEnabled())
        self.assertFalse(self.sidebar.ae_check.isEnabled())
        self.assertFalse(self.sidebar.archival_split_check.isEnabled())
        self.assertFalse(self.sidebar.registered_geometry_check.isEnabled())
        self.assertFalse(self.sidebar.subframe_spin.isEnabled())
        self.assertFalse(self.sidebar.br_y_spin.isEnabled())
        self.assertFalse(self.sidebar.load_registration_btn.isEnabled())

    def test_controller_signals_connected_without_error(self):
        # _connect_signals() already ran in setUp via __init__; a bad signal/slot
        # signature would have raised there. Emitting confirms the real
        # pyqtSignal -> slot binding is live (not a MagicMock no-op).
        self.sidebar.controller.scan_progress.emit(0.5)
        self.assertEqual(self.sidebar.progress_bar.value(), 50)


class TestCapabilityGating(ScanSidebarTestCase):
    def test_full_capability_device_enables_new_controls(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.assertTrue(self.sidebar.frame_spin.isEnabled())
        self.assertEqual(self.sidebar.frame_spin.maximum(), 40)
        self.assertTrue(self.sidebar.ae_check.isEnabled())
        self.assertTrue(self.sidebar.archival_split_check.isEnabled())
        self.assertTrue(self.sidebar.registered_geometry_check.isEnabled())
        self.assertTrue(self.sidebar.load_registration_btn.isEnabled())
        # Subframe/BR-Y stay disabled until "Use Registered Geometry" is checked.
        self.assertFalse(self.sidebar.subframe_spin.isEnabled())
        self.assertFalse(self.sidebar.br_y_spin.isEnabled())

    def test_minimal_capability_device_disables_new_controls(self):
        _select_device(self.sidebar, MINIMAL_DEVICE)
        self.assertFalse(self.sidebar.frame_spin.isEnabled())
        self.assertFalse(self.sidebar.ae_check.isEnabled())
        self.assertFalse(self.sidebar.archival_split_check.isEnabled())
        self.assertFalse(self.sidebar.registered_geometry_check.isEnabled())
        self.assertFalse(self.sidebar.load_registration_btn.isEnabled())
        # Pre-existing controls unaffected by the new gating.
        self.assertFalse(self.sidebar.ir_check.isEnabled())
        self.assertTrue(self.sidebar.dpi_combo.isEnabled())

    def test_switching_to_a_minimal_device_clears_stale_registration(self):
        """Registered geometry is frame/device-specific; it must never
        silently carry over onto a different device."""
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.registered_geometry_check.setChecked(True)
        self.sidebar.subframe_spin.setValue(6.35)
        self.sidebar.br_y_spin.setValue(5003)

        _select_device(self.sidebar, MINIMAL_DEVICE)

        self.assertFalse(self.sidebar.registered_geometry_check.isChecked())
        self.assertEqual(self.sidebar.subframe_spin.value(), 0.0)
        self.assertEqual(self.sidebar.br_y_spin.value(), 0)


class TestArchivalSplitInterlock(ScanSidebarTestCase):
    def test_checking_archival_forces_and_locks_ir_and_samples(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.ir_check.setChecked(False)

        self.sidebar.archival_split_check.setChecked(True)

        self.assertTrue(self.sidebar.ir_check.isChecked())
        self.assertEqual(self.sidebar.samples_combo.currentData(), 4)
        self.assertFalse(self.sidebar.ir_check.isEnabled())
        self.assertFalse(self.sidebar.samples_combo.isEnabled())

    def test_unchecking_archival_restores_capability_derived_enabled_state(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.archival_split_check.setChecked(True)

        self.sidebar.archival_split_check.setChecked(False)

        self.assertTrue(self.sidebar.ir_check.isEnabled())
        self.assertTrue(self.sidebar.samples_combo.isEnabled())

    def test_archival_unavailable_on_minimal_device(self):
        _select_device(self.sidebar, MINIMAL_DEVICE)
        self.assertFalse(self.sidebar.archival_split_check.isChecked())
        self.assertFalse(self.sidebar.archival_split_check.isEnabled())


class TestRegisteredGeometryToggle(ScanSidebarTestCase):
    def test_checking_enables_subframe_and_br_y_fields(self):
        _select_device(self.sidebar, FULL_DEVICE)

        self.sidebar.registered_geometry_check.setChecked(True)

        self.assertTrue(self.sidebar.subframe_spin.isEnabled())
        self.assertTrue(self.sidebar.br_y_spin.isEnabled())

    def test_unchecking_disables_subframe_and_br_y_fields(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.registered_geometry_check.setChecked(True)

        self.sidebar.registered_geometry_check.setChecked(False)

        self.assertFalse(self.sidebar.subframe_spin.isEnabled())
        self.assertFalse(self.sidebar.br_y_spin.isEnabled())

    def test_load_registration_json_populates_fields(self):
        import tempfile
        import os

        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.frame_spin.setValue(3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = os.path.join(tmp_dir, "registration.json")
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump({"frames": [{"frame": 3, "subframe_mm": 6.35, "br_y": 5003}]}, fh)

            with patch(
                "negpy.desktop.view.sidebar.scan.QFileDialog.getOpenFileName",
                return_value=(manifest_path, "JSON Files (*.json)"),
            ):
                self.sidebar._on_load_registration_json()

        self.assertTrue(self.sidebar.registered_geometry_check.isChecked())
        self.assertAlmostEqual(self.sidebar.subframe_spin.value(), 6.35, places=2)
        self.assertEqual(self.sidebar.br_y_spin.value(), 5003)
        self.assertIn("Loaded registration for frame 3", self.sidebar.status_label.text())

    def test_load_registration_json_requires_frame_first(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.frame_spin.setValue(0)  # "Current" sentinel -> no frame chosen

        with patch("negpy.desktop.view.sidebar.scan.QFileDialog.getOpenFileName") as mock_dialog:
            self.sidebar._on_load_registration_json()
            mock_dialog.assert_not_called()

        self.assertIn("Frame #", self.sidebar.status_label.text())

    def test_load_registration_json_no_matching_frame_shows_error(self):
        import tempfile
        import os

        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.frame_spin.setValue(9)

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = os.path.join(tmp_dir, "registration.json")
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump({"frames": [{"frame": 3, "subframe_mm": 6.35, "br_y": 5003}]}, fh)

            with patch(
                "negpy.desktop.view.sidebar.scan.QFileDialog.getOpenFileName",
                return_value=(manifest_path, "JSON Files (*.json)"),
            ):
                self.sidebar._on_load_registration_json()

        self.assertFalse(self.sidebar.registered_geometry_check.isChecked())
        self.assertIn("Could not load registration manifest", self.sidebar.status_label.text())


class TestScanParamsAssembly(ScanSidebarTestCase):
    """_on_scan() gathers widget values and hands them to
    controller.build_scan_params(); this checks that hand-off end to end
    against the real (non-mocked) controller method, without ever calling
    controller.start_scan / touching the scan worker thread."""

    def test_archival_recipe_end_to_end_through_build_scan_params(self):
        _select_device(self.sidebar, FULL_DEVICE)
        self.sidebar.folder_edit.setText("/tmp/negpy-scan-test")
        self.sidebar.archival_split_check.setChecked(True)
        self.sidebar.ae_check.setChecked(True)
        self.sidebar.frame_spin.setValue(3)
        self.sidebar.registered_geometry_check.setChecked(True)
        self.sidebar.subframe_spin.setValue(6.35)
        self.sidebar.br_y_spin.setValue(5003)

        captured = {}
        self.controller.start_scan = MagicMock(side_effect=lambda req: captured.update(req=req))

        self.sidebar._on_scan()

        params = captured["req"].params
        self.assertTrue(params.capture_ir)
        self.assertEqual(params.samples_per_scan, 4)
        self.assertTrue(params.auto_exposure)
        self.assertIsNone(params.frame)  # rides inside registered_geometry instead
        self.assertEqual(params.registered_geometry.frame, 3)
        self.assertEqual(params.registered_geometry.subframe_mm, 6.35)
        self.assertEqual(params.registered_geometry.br_y_device_px, 5003)


if __name__ == "__main__":
    unittest.main()
