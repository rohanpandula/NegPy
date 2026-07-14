import json
import os

import qtawesome as qta
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.templates import hint_label, section_subheader
from negpy.desktop.view.styles.theme import THEME
from negpy.infrastructure.scanners.base import ScannerCapabilities, ScannerDevice
from negpy.infrastructure.scanners.settings import ScannerSettings


class ScanSidebar(QWidget):
    """Scanner control panel — replaces the originally planned modal ScanDialog."""

    def __init__(self, controller) -> None:
        super().__init__()
        self.controller = controller
        self._settings: ScannerSettings = self._load_settings()
        self._devices: list[ScannerDevice] = []
        self._scanning = False
        self._devices_loaded = False
        self._init_ui()
        self._connect_signals()

    # ── settings persistence ──────────────────────────────────────────

    def _load_settings(self) -> ScannerSettings:
        data = self.controller.session.repo.get_global_setting("scanner_settings", default={})
        if isinstance(data, dict) and data:
            try:
                return ScannerSettings(**data)
            except Exception:
                pass
        return ScannerSettings.defaults()

    def _save_settings(self) -> None:
        from dataclasses import asdict

        self.controller.session.repo.save_global_setting("scanner_settings", asdict(self._settings))

    @property
    def settings(self) -> ScannerSettings:
        return self._settings

    @settings.setter
    def settings(self, value: ScannerSettings) -> None:
        self._settings = value
        self._save_settings()

    # ── UI construction ───────────────────────────────────────────────

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 5)
        layout.setSpacing(10)

        # ── DEVICE ───────────────────────────────────────────
        layout.addWidget(section_subheader("DEVICE"))

        device_row = QHBoxLayout()
        self.device_combo = QComboBox()
        self.device_combo.setToolTip("Select scanner")
        self.device_combo.addItem("Detecting scanners…", None)

        self.refresh_btn = QPushButton()
        self.refresh_btn.setIcon(qta.icon("fa5s.redo", color=THEME.text_secondary))
        self.refresh_btn.setToolTip("Refresh device list")
        self.refresh_btn.setFixedWidth(32)

        device_row.addWidget(self.device_combo, 1)
        device_row.addWidget(self.refresh_btn)
        layout.addLayout(device_row)

        # ── CAPS INFO ───────────────────────────────────────
        self.frame_label = hint_label("")
        layout.addWidget(self.frame_label)

        # ── SETTINGS ────────────────────────────────────────
        self.form = QFormLayout()
        self.form.setSpacing(6)

        self.dpi_combo = QComboBox()
        self.dpi_combo.setToolTip("Resolution (DPI)")
        self.dpi_combo.setEditable(True)
        self.form.addRow("DPI", self.dpi_combo)

        self.ir_check = QCheckBox("IR")
        self.ir_check.setToolTip("Scan a separate infrared channel for dust detection")

        depth_row = QHBoxLayout()
        depth_row.setContentsMargins(0, 0, 0, 0)
        self.depth_combo = QComboBox()
        self.depth_combo.setToolTip("Bit depth")
        depth_row.addWidget(self.depth_combo, 1)
        depth_row.addWidget(self.ir_check)
        self.form.addRow("Depth", depth_row)

        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 0)
        self.frame_spin.setSpecialValueText("Current")
        self.frame_spin.setToolTip("Frame selection not supported by this device")
        self.form.addRow("Frame #", self.frame_spin)

        self.autofocus_check = QCheckBox("Autofocus")
        self.autofocus_check.setChecked(True)
        self.autofocus_check.setToolTip("Autofocus before scanning (recommended — film is rarely perfectly flat)")
        self.form.addRow("Autofocus", self.autofocus_check)

        self.ae_check = QCheckBox("Auto-Exposure")
        self.ae_check.setToolTip("Hardware auto-exposure not supported by this device")
        self.form.addRow("Auto-Exposure", self.ae_check)

        self.samples_combo = QComboBox()
        self.samples_combo.setToolTip("Hardware multi-sampling passes per line (higher = less noise, slower)")
        for n in (1, 2, 4, 8, 16):
            self.samples_combo.addItem(f"{n}x", n)
        self.form.addRow("Samples/pass", self.samples_combo)

        self.archival_split_check = QCheckBox("Archival Split (RGB4x + IR1x)")
        self.archival_split_check.setToolTip("Requires both IR and hardware multi-sampling support")
        self.form.addRow("Archival", self.archival_split_check)

        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["TIFF", "DNG"])
        self.fmt_combo.setToolTip("Output file format")
        self.form.addRow("Format", self.fmt_combo)

        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Output folder…")
        self.folder_edit.setToolTip("Directory for scanned files")
        self.browse_btn = QPushButton("…")
        self.browse_btn.setFixedWidth(32)
        self.browse_btn.setToolTip("Browse for output folder")
        folder_row.addWidget(self.folder_edit)
        folder_row.addWidget(self.browse_btn)
        self.form.addRow("Folder", folder_row)

        self.pattern_edit = QLineEdit()
        self.pattern_edit.setToolTip('Jinja2 template. Variables: {{ date }}, {{ seq }}.\nExample: {{ date }}_{{ "%03d" % seq }}')
        self.form.addRow("Filename", self.pattern_edit)

        layout.addLayout(self.form)

        # ── REGISTRATION ──────────────────────────────────────
        layout.addWidget(section_subheader("REGISTRATION"))

        self.registered_geometry_check = QCheckBox("Use Registered Geometry")
        self.registered_geometry_check.setToolTip("Registered geometry not supported by this device")
        layout.addWidget(self.registered_geometry_check)

        self.registration_form = QFormLayout()
        self.registration_form.setSpacing(6)

        self.subframe_spin = QDoubleSpinBox()
        self.subframe_spin.setDecimals(2)
        # Upper bound mirrors RollRegistrationConfig.maximum_subframe_mm (the
        # validated LS-5000 frame-pitch ceiling) — a UI convenience bound
        # only; SaneBackend.scan() is the real enforcement point.
        self.subframe_spin.setRange(0.0, 37.82)
        self.subframe_spin.setSuffix(" mm")
        self.subframe_spin.setEnabled(False)
        self.subframe_spin.setToolTip("Fine transport shift for this frame (RegisteredScanGeometry.subframe_mm)")
        self.registration_form.addRow("Subframe", self.subframe_spin)

        self.br_y_spin = QSpinBox()
        # Upper bound mirrors RollRegistrationConfig.maximum_br_y_device_px
        # (the validated LS-5000 device-pixel ceiling) — a UI convenience
        # bound only; SaneBackend.scan() is the real enforcement point.
        self.br_y_spin.setRange(0, 5958)
        self.br_y_spin.setEnabled(False)
        self.br_y_spin.setToolTip(
            "Inclusive bottom coordinate of the shortened scan window, in device pixels (RegisteredScanGeometry.br_y_device_px)"
        )
        self.registration_form.addRow("BR-Y", self.br_y_spin)

        layout.addLayout(self.registration_form)

        self.load_registration_btn = QPushButton(" Load Registration JSON…")
        self.load_registration_btn.setIcon(qta.icon("fa5s.folder-open", color=THEME.text_primary))
        self.load_registration_btn.setEnabled(False)
        self.load_registration_btn.setToolTip(
            "Load Subframe/BR-Y for the current Frame # from a registration manifest "
            '({"frames": [{"frame": N, "subframe_mm": ..., "br_y": ...}]})'
        )
        layout.addWidget(self.load_registration_btn)

        # ── PROGRESS ────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Scanning… %p%")
        layout.addWidget(self.progress_bar)

        # ── STATUS ──────────────────────────────────────────
        self.status_label = hint_label("")
        layout.addWidget(self.status_label)

        # ── SCAN BUTTON ─────────────────────────────────────
        self.scan_btn = QPushButton(" Scan")
        self.scan_btn.setObjectName("scan_btn")
        self.scan_btn.setFixedHeight(40)
        self.scan_btn.setIcon(qta.icon("fa5s.camera-retro", color=THEME.text_primary))
        layout.addWidget(self.scan_btn)

        layout.addStretch()

        # Pre-fill from persisted settings
        self.fmt_combo.setCurrentText(self._settings.output_format)
        self.folder_edit.setText(self._settings.output_folder)
        self.pattern_edit.setText(self._settings.filename_pattern)
        self.autofocus_check.setChecked(self._settings.autofocus)
        self.ae_check.setChecked(self._settings.auto_exposure)
        self.archival_split_check.setChecked(self._settings.archival_split_capture)

    def _connect_signals(self) -> None:
        self.refresh_btn.clicked.connect(self._on_refresh)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.browse_btn.clicked.connect(self._on_browse)
        self.scan_btn.clicked.connect(self._on_scan)
        self.folder_edit.textChanged.connect(lambda: self._update_settings_from_ui())
        self.pattern_edit.textChanged.connect(lambda: self._update_settings_from_ui())
        self.fmt_combo.currentTextChanged.connect(lambda: self._update_settings_from_ui())
        self.dpi_combo.currentTextChanged.connect(lambda: self._update_settings_from_ui())
        self.depth_combo.currentTextChanged.connect(lambda: self._update_settings_from_ui())
        self.ir_check.toggled.connect(lambda: self._update_settings_from_ui())
        self.autofocus_check.toggled.connect(lambda: self._update_settings_from_ui())
        self.ae_check.toggled.connect(lambda: self._update_settings_from_ui())
        self.samples_combo.currentTextChanged.connect(lambda: self._update_settings_from_ui())
        self.archival_split_check.toggled.connect(self._on_archival_split_toggled)
        self.registered_geometry_check.toggled.connect(self._on_registered_geometry_toggled)
        self.load_registration_btn.clicked.connect(self._on_load_registration_json)

        # Controller signals
        self.controller.scan_devices_ready.connect(self._on_devices_ready)
        self.controller.scan_progress.connect(self._on_scan_progress)
        self.controller.scan_finished.connect(self._on_scan_finished)
        self.controller.scan_error.connect(self._on_scan_error)

    # ── activation hook ───────────────────────────────────────────────

    def on_activated(self) -> None:
        """Called when the Scan tab is switched to."""
        if not self._devices_loaded:
            self._request_devices()

    # ── slots ─────────────────────────────────────────────────────────

    def _request_devices(self) -> None:
        """Request device list from the scan worker thread."""
        if not self._sane_available():
            self._show_sane_missing()
            return
        self.device_combo.clear()
        self.device_combo.addItem("Detecting scanners…", None)
        self.device_combo.setEnabled(False)
        self.status_label.setText("Detecting scanners…")
        self.controller.request_scan_devices()

    @staticmethod
    def _sane_available() -> bool:
        try:
            import sane  # noqa: F401

            return True
        except Exception:
            return False

    def _show_sane_missing(self) -> None:
        import sys

        if sys.platform == "darwin":
            hint = "brew install sane-backends"
        else:
            hint = "sudo apt install libsane  # Debian/Ubuntu\nsudo pacman -S sane  # Arch\nor your distro's sane equivalent"
        self.device_combo.clear()
        self.device_combo.addItem("SANE not available", None)
        self.device_combo.setEnabled(False)
        self.scan_btn.setEnabled(False)
        self.status_label.setText(f"Scanner support requires SANE (libsane).\n\nTo enable:\n{hint}")

    def _on_refresh(self) -> None:
        self._request_devices()

    @pyqtSlot(list)
    def _on_devices_ready(self, devices: list) -> None:
        self._devices = devices
        self._devices_loaded = True
        self.device_combo.clear()
        self.device_combo.setEnabled(True)

        if not devices:
            self.device_combo.addItem("No scanners detected", None)
            self.device_combo.setEnabled(False)
            self.status_label.setText("No scanners detected. Plug in your scanner and click Refresh.")
            self.scan_btn.setEnabled(False)
            return

        for d in devices:
            label_text = f"{d.vendor} {d.model}" if d.vendor else d.model
            self.device_combo.addItem(label_text, d.id)

        # Restore last-used device if present
        if self._settings.last_device_id:
            for i in range(self.device_combo.count()):
                if self.device_combo.itemData(i) == self._settings.last_device_id:
                    self.device_combo.setCurrentIndex(i)
                    break

        self._update_device_caps()

    def _on_device_changed(self, _index: int) -> None:
        self._update_device_caps()

    def _current_device(self) -> ScannerDevice | None:
        device_id = self.device_combo.currentData()
        if not device_id:
            return None
        for d in self._devices:
            if d.id == device_id:
                return d
        return None

    def _update_device_caps(self) -> None:
        device = self._current_device()
        if device is None:
            self.scan_btn.setEnabled(False)
            self.frame_label.setText("")
            self.dpi_combo.setEnabled(False)
            self.depth_combo.setEnabled(False)
            self.ir_check.setEnabled(False)
            self.samples_combo.setEnabled(False)
            self.frame_spin.setEnabled(False)
            self.ae_check.setEnabled(False)
            self.archival_split_check.setEnabled(False)
            self.registered_geometry_check.setEnabled(False)
            self._update_registration_fields_enabled()
            return

        caps = device.capabilities
        self.dpi_combo.setEnabled(True)
        self.depth_combo.setEnabled(True)
        self.ir_check.setEnabled(True)
        self.samples_combo.setEnabled(True)
        self.frame_label.setText(f"Frame: {caps.max_area_mm[0]:.0f} × {caps.max_area_mm[1]:.0f} mm")

        # If no film sources, show banner
        if not caps.sources:
            self.status_label.setText("This scanner reports no film/transparency sources. NegPy v1 supports film scanning only.")
            self.scan_btn.setEnabled(False)
        else:
            self.status_label.setText("")
            self.scan_btn.setEnabled(True)

        self._populate_form(caps)

    def _populate_form(self, caps: ScannerCapabilities) -> None:
        self.dpi_combo.blockSignals(True)
        self.depth_combo.blockSignals(True)
        self.ir_check.blockSignals(True)
        self.samples_combo.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.ae_check.blockSignals(True)
        self.archival_split_check.blockSignals(True)
        self.registered_geometry_check.blockSignals(True)

        # DPI
        self.dpi_combo.clear()
        if caps.supported_dpi:
            for d in caps.supported_dpi:
                self.dpi_combo.addItem(str(d), d)
        if self._settings.dpi:
            idx = self.dpi_combo.findData(self._settings.dpi)
            if idx >= 0:
                self.dpi_combo.setCurrentIndex(idx)
            else:
                self.dpi_combo.setCurrentText(str(self._settings.dpi))

        # Depth
        self.depth_combo.clear()
        if caps.supported_depths:
            for d in caps.supported_depths:
                self.depth_combo.addItem(f"{d}-bit", d)
        if self._settings.depth:
            idx = self.depth_combo.findData(self._settings.depth)
            if idx >= 0:
                self.depth_combo.setCurrentIndex(idx)

        # IR
        self.ir_check.setEnabled(caps.ir_channel)
        if caps.ir_channel:
            self.ir_check.setChecked(self._settings.capture_ir)
            self.ir_check.setToolTip("Scan a separate infrared channel for dust detection")
        else:
            self.ir_check.setChecked(False)
            self.ir_check.setToolTip("IR scanning not supported by this device")

        # Samples/pass (hardware multi-sampling)
        self.samples_combo.setEnabled(caps.multi_sample)
        if caps.multi_sample:
            idx = self.samples_combo.findData(self._settings.samples_per_scan)
            self.samples_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.samples_combo.setToolTip("Hardware multi-sampling passes per line (higher = less noise, slower)")
        else:
            self.samples_combo.setCurrentIndex(0)
            self.samples_combo.setToolTip("Multi-sampling not supported by this device")

        # Frame selection (roll-adapter transport position)
        if caps.adapter_frame_capacity is not None:
            self.frame_spin.setEnabled(True)
            self.frame_spin.setRange(0, caps.adapter_frame_capacity)
            self.frame_spin.setValue(min(self.frame_spin.value(), caps.adapter_frame_capacity))
            self.frame_spin.setToolTip(
                f"Select a specific frame (1–{caps.adapter_frame_capacity}) on the roll adapter before scanning "
                "(“Current” leaves the transport position unchanged)"
            )
        else:
            self.frame_spin.setEnabled(False)
            self.frame_spin.setRange(0, 0)
            self.frame_spin.setValue(0)
            self.frame_spin.setToolTip("Frame selection not supported by this device")

        # Hardware auto-exposure
        self.ae_check.setEnabled(caps.auto_exposure)
        if caps.auto_exposure:
            self.ae_check.setChecked(self._settings.auto_exposure)
            self.ae_check.setToolTip("Meter this positioned frame in hardware before capture")
        else:
            self.ae_check.setChecked(False)
            self.ae_check.setToolTip("Hardware auto-exposure not supported by this device")

        # Archival split-capture (validated RGB4x + IR1x practical-parity recipe)
        archival_supported = caps.ir_channel and caps.multi_sample
        self.archival_split_check.setEnabled(archival_supported)
        if archival_supported:
            self.archival_split_check.setChecked(self._settings.archival_split_capture)
            self.archival_split_check.setToolTip(
                "Validated archival recipe: 4× hardware-multisampled RGB plus a registered single-pass IR channel"
            )
        else:
            self.archival_split_check.setChecked(False)
            self.archival_split_check.setToolTip("Requires both IR and hardware multi-sampling support")

        # Registered geometry (fine transport shift + shortened scan window).
        # Always resets on a capability refresh (device switch or Refresh
        # click) — frame-specific geometry from a previous device/session
        # must never silently carry over to a new one.
        self.registered_geometry_check.setEnabled(caps.registered_geometry)
        self.registered_geometry_check.setChecked(False)
        self.subframe_spin.setValue(0.0)
        self.br_y_spin.setValue(0)
        if caps.registered_geometry:
            self.registered_geometry_check.setToolTip("Position a fine transport shift and shortened scan window for this frame")
        else:
            self.registered_geometry_check.setToolTip("Registered geometry not supported by this device")

        self.dpi_combo.blockSignals(False)
        self.depth_combo.blockSignals(False)
        self.ir_check.blockSignals(False)
        self.samples_combo.blockSignals(False)
        self.frame_spin.blockSignals(False)
        self.ae_check.blockSignals(False)
        self.archival_split_check.blockSignals(False)
        self.registered_geometry_check.blockSignals(False)

        # Cross-widget effects that must run after signals are unblocked
        # (archival split may need to re-lock IR/samples; registration fields
        # follow the "Use Registered Geometry" checkbox's restored state).
        self._apply_archival_split_interlock()
        self._update_registration_fields_enabled()

    # ── new-control interlocks ───────────────────────────────────────────

    def _apply_archival_split_interlock(self) -> None:
        """Force + lock IR/samples to the validated RGB4x+IR1x recipe while
        archival split-capture is active; leaves them alone otherwise."""
        if not (self.archival_split_check.isChecked() and self.archival_split_check.isEnabled()):
            return
        self.ir_check.blockSignals(True)
        self.ir_check.setChecked(True)
        self.ir_check.blockSignals(False)
        idx = self.samples_combo.findData(4)
        if idx >= 0:
            self.samples_combo.blockSignals(True)
            self.samples_combo.setCurrentIndex(idx)
            self.samples_combo.blockSignals(False)
        self.ir_check.setEnabled(False)
        self.samples_combo.setEnabled(False)

    def _on_archival_split_toggled(self, _checked: bool) -> None:
        if not self.archival_split_check.isChecked():
            # Restore IR/samples to whatever the device's own capabilities allow.
            device = self._current_device()
            caps = device.capabilities if device is not None else None
            self.ir_check.setEnabled(caps.ir_channel if caps is not None else False)
            self.samples_combo.setEnabled(caps.multi_sample if caps is not None else False)
        self._apply_archival_split_interlock()
        self._update_settings_from_ui()

    def _on_registered_geometry_toggled(self, _checked: bool) -> None:
        self._update_registration_fields_enabled()

    def _update_registration_fields_enabled(self) -> None:
        active = self.registered_geometry_check.isChecked() and self.registered_geometry_check.isEnabled()
        self.subframe_spin.setEnabled(active)
        self.br_y_spin.setEnabled(active)
        self.load_registration_btn.setEnabled(self.registered_geometry_check.isEnabled())

    def _on_load_registration_json(self) -> None:
        frame_value = self.frame_spin.value()
        if frame_value <= 0:
            self.status_label.setText("Set a Frame # before loading a registration manifest.")
            return

        path, _ = QFileDialog.getOpenFileName(self, "Load Registration JSON", "", "JSON Files (*.json)")
        if not path:
            return

        from negpy.infrastructure.scanners.params import parse_registration_manifest

        try:
            with open(path, encoding="utf-8") as stream:
                data = json.load(stream)
            geometry = parse_registration_manifest(data, frame=frame_value)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.status_label.setText(f"Could not load registration manifest: {exc}")
            return

        self.subframe_spin.setValue(geometry.subframe_mm)
        self.br_y_spin.setValue(geometry.br_y_device_px)
        self.registered_geometry_check.setChecked(True)
        self._update_registration_fields_enabled()
        self.status_label.setText(f"Loaded registration for frame {frame_value} from {os.path.basename(path)}")

    def _on_browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.folder_edit.setText(folder)
            self._update_settings_from_ui()

    def _on_scan(self) -> None:
        if self._scanning:
            # Cancel
            self.controller.cancel_scan()
            return

        # Validate
        device = self._current_device()
        if device is None:
            return

        output_folder = self.folder_edit.text().strip()
        if not output_folder:
            self._on_browse()
            output_folder = self.folder_edit.text().strip()
            if not output_folder:
                return

        # Build ScanRequest
        from negpy.desktop.workers.scan_worker import ScanRequest

        dpi = int(self.dpi_combo.currentData() or self.dpi_combo.currentText() or 3600)
        depth = int(self.depth_combo.currentData() or 16)
        # isChecked() alone is authoritative here (no isEnabled() guard): a
        # device is guaranteed selected at this point (checked above), and
        # ir_check.isChecked() is always kept truthful for it — forced False
        # by _populate_form when the device lacks IR, forced True by the
        # archival split-capture interlock, which also *disables* the box
        # while leaving it checked=True to lock out manual edits. Gating on
        # isEnabled() too would silently drop IR during archival capture.
        capture_ir = self.ir_check.isChecked()
        frame = self.frame_spin.value() or None
        use_registered_geometry = self.registered_geometry_check.isEnabled() and self.registered_geometry_check.isChecked()

        try:
            params = self.controller.build_scan_params(
                dpi=dpi,
                depth=depth,
                capture_ir=capture_ir,
                autofocus=self.autofocus_check.isChecked(),
                samples_per_scan=int(self.samples_combo.currentData() or 1),
                frame=frame,
                auto_exposure=self.ae_check.isEnabled() and self.ae_check.isChecked(),
                subframe_mm=self.subframe_spin.value() if use_registered_geometry else None,
                br_y_device_px=self.br_y_spin.value() if use_registered_geometry else None,
            )
        except ValueError as exc:
            self.status_label.setText(f"Cannot start scan: {exc}")
            return

        req = ScanRequest(
            device_id=device.id,
            params=params,
            output_folder=output_folder,
            filename_pattern=self.pattern_edit.text().strip() or '{{ date }}_{{ "%03d" % seq }}',
            output_format=self.fmt_combo.currentText(),
        )

        self._update_settings_from_ui()
        self._save_settings()

        self.set_scanning(True)
        self.controller.start_scan(req)

    @pyqtSlot(float)
    def _on_scan_progress(self, progress: float) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(int(progress * 100))

    @pyqtSlot(str)
    def _on_scan_finished(self, path: str) -> None:
        self.set_scanning(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Scanned: {path}")

    @pyqtSlot(str)
    def _on_scan_error(self, msg: str) -> None:
        self.set_scanning(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {msg}")

    # ── state helpers ─────────────────────────────────────────────────

    def set_scanning(self, active: bool) -> None:
        self._scanning = active
        if active:
            self.scan_btn.setText(" Stop")
            self.scan_btn.setIcon(qta.icon("fa5s.stop", color=THEME.text_primary))
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
        else:
            self.scan_btn.setText(" Scan")
            self.scan_btn.setIcon(qta.icon("fa5s.camera-retro", color=THEME.text_primary))

    def _update_settings_from_ui(self) -> None:
        dpi_text = self.dpi_combo.currentData() or self.dpi_combo.currentText()
        depth_text = self.depth_combo.currentData() or 16
        try:
            dpi = int(dpi_text)
        except (ValueError, TypeError):
            dpi = 3600
        try:
            depth = int(depth_text)
        except (ValueError, TypeError):
            depth = 16

        device = self._current_device()
        self.settings = ScannerSettings(
            last_device_id=device.id if device else self._settings.last_device_id,
            dpi=dpi,
            depth=depth,
            # See the matching comment in _on_scan(): isChecked() alone is
            # authoritative — an isEnabled() guard would misread the archival
            # split-capture interlock's locked-but-checked state as "off".
            capture_ir=self.ir_check.isChecked(),
            autofocus=self.autofocus_check.isChecked(),
            samples_per_scan=int(self.samples_combo.currentData() or 1),
            auto_exposure=self.ae_check.isChecked() and self.ae_check.isEnabled(),
            archival_split_capture=self.archival_split_check.isChecked() and self.archival_split_check.isEnabled(),
            output_folder=self.folder_edit.text().strip(),
            output_format=self.fmt_combo.currentText(),
            filename_pattern=self.pattern_edit.text().strip() or '{{ date }}_{{ "%03d" % seq }}',
        )


class _ScanUnsupportedPlaceholder(QWidget):
    """Shown on Windows where SANE is not available."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        label = QLabel("Scanner support not yet available on Windows.")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_base}px; padding: 20px;")
        layout.addWidget(label)
