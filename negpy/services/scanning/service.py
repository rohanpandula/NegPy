import os
import threading
from typing import Callable

from negpy.infrastructure.scanners.base import ScannerBackend, ScannerDevice, ScannerSession
from negpy.infrastructure.scanners.params import ScanParams
from negpy.infrastructure.scanners.per_frame_roll import PerFrameRollSession
from negpy.infrastructure.scanners.result import ScanResult
from negpy.infrastructure.scanners.roll import RollSession
from negpy.kernel.system.logging import get_logger
from negpy.services.scanning.templating import render_scan_filename

logger = get_logger(__name__)


class ScannerService:
    """Orchestrates device enumeration, scan execution, and file writing."""

    def __init__(self, backend: ScannerBackend | None = None, backend_id: str | None = None) -> None:
        self._backend = backend
        self._backend_id = backend_id

    def _get_backend(self) -> ScannerBackend:
        if self._backend is None:
            from negpy.infrastructure.scanners.registry import DEFAULT_BACKEND_ID, create_backend

            self._backend = create_backend(self._backend_id or DEFAULT_BACKEND_ID)
        return self._backend

    def list_devices(self) -> list[ScannerDevice]:
        return self._get_backend().list_devices()

    def refresh_devices(self) -> list[ScannerDevice]:
        backend = self._get_backend()
        refresh = getattr(backend, "refresh_devices", None)
        if callable(refresh):
            return refresh()
        return backend.list_devices()

    def probe_device(self, device_id: str) -> ScannerDevice:
        """Return one device from a fresh backend enumeration."""

        backend = self._get_backend()
        try:
            strict_probe = getattr(backend, "probe_device", None)
            if callable(strict_probe):
                device = strict_probe(device_id)
                if device is not None:
                    return device
                devices: list[ScannerDevice] = []
            else:
                devices = self.refresh_devices()
        except Exception as exc:
            raise RuntimeError(f"Could not probe scanner device {device_id!r}: fresh enumeration failed: {exc}") from exc

        for device in devices:
            if device.id == device_id:
                return device
        raise RuntimeError(f"Scanner device {device_id!r} was not found during fresh enumeration")

    def run_scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[[float], None],
        cancel: threading.Event,
    ) -> ScanResult:
        backend = self._get_backend()
        return backend.scan(device_id, params, progress, cancel)

    def open_session(self, device_id: str) -> ScannerSession:
        """Open an exclusive session on a session-capable backend.

        Optional-method pattern like refresh_devices: a backend without
        session support is addressed through one-shot scan()/eject().
        """
        backend = self._get_backend()
        open_session = getattr(backend, "open_session", None)
        if not callable(open_session):
            raise RuntimeError(f"Backend {type(backend).__name__} does not support exclusive sessions")
        return open_session(device_id)

    def open_roll(self, device: ScannerDevice, *, dpi: int) -> RollSession:
        """Open a whole-strip roll session over the device's backend."""
        return PerFrameRollSession(self._get_backend(), device, dpi=dpi)

    def eject(self, device_id: str) -> bool:
        """Trigger a capability-gated film eject; False when unsupported.

        Mirrors the optional-method pattern in refresh_devices/probe_device
        above — only SaneBackend implements this today.
        """
        backend = self._get_backend()
        eject = getattr(backend, "eject", None)
        if not callable(eject):
            return False
        return bool(eject(device_id))

    def write_result(
        self,
        result: ScanResult,
        output_folder: str,
        filename_pattern: str,
        output_format: str = "TIFF",
        seq: int | None = None,
    ) -> str:
        """Write ScanResult to disk. Returns path to the RGB file.

        Filename pattern is a Jinja2 template with variables: date, seq. `seq`
        seeds the collision search: single scans pass None (start at 1); a range
        batch passes the frame number so masters are frame-numbered.
        """
        from datetime import date as dt_date

        from negpy.services.scanning.writer import write_dng_linear, write_tiff_16bit

        os.makedirs(output_folder, exist_ok=True)

        date_str = dt_date.today().strftime("%Y%m%d")
        ext = ".dng" if output_format.upper() == "DNG" else ".tif"

        seq = seq or 1
        while True:
            basename = render_scan_filename(filename_pattern, date_str, seq)
            rgb_path = os.path.join(output_folder, basename)
            if not os.path.exists(rgb_path + ext):
                break
            seq += 1

        if output_format.upper() == "DNG":
            rgb_path = write_dng_linear(result, rgb_path)
        else:
            rgb_path = write_tiff_16bit(result, rgb_path)

        return rgb_path
