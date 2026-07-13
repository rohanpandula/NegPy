"""negpy/features/nikonlook/processor.py -- thin adapter wiring negfit's
"Nikon Scan look" bundle system (the external `nikonlook_core` package)
into NegPy. ALL color math (matrix multiply, curve application, gain
estimation) lives in `nikonlook_core` -- this file only:

  1. resolves which bundle to use (config field / env var / package
     default),
  2. loads + caches it,
  3. runs NegPy's OWN existing IR dust-removal ahead of the color model
     (reused unmodified -- see `apply_ir_dust_removal` in
     negpy/features/retouch/logic.py -- per
     NEGPY-INTEGRATION-PLAN.md's processing order: scan -> IR heal ->
     nikonlook core -> export),
  4. calls the three FROZEN `nikonlook_core` functions
     (`load_bundle`/`estimate_gains`/`apply`).

No color science is reimplemented here. See
digital-ice-2026/negfit/NEGPY-INTEGRATION-PLAN.md for the full design and
digital-ice-2026/negfit/nikonlook_core.py for the frozen API contract.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.nikonlook.models import NikonLookConfig
from negpy.features.retouch.logic import apply_ir_dust_removal
from negpy.features.retouch.models import RetouchConfig
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

NEGFIT_BUNDLE_ENV_VAR = "NEGFIT_BUNDLE"


class NikonLookUnavailable(RuntimeError):
    """Raised when Nikon Scan look mode is enabled but cannot actually run
    -- either `nikonlook_core` isn't importable (editable install missing;
    see NEGPY-INTEGRATION-PLAN.md's `uv pip install -e
    ~/Downloads/digital-ice-2026` step) or no bundle directory resolves.
    Deliberately NOT caught anywhere in this module: a clear, immediate
    failure is far less confusing than silently falling back to normal
    C41/BW/E6 inversion when a user has explicitly opted into this mode.
    """


def _import_nikonlook_core():
    """Deferred import -- `nikonlook_core` is an optional, external,
    editable-installed dependency (see module docstring). Every OTHER
    NegPy module must keep working (and the ~1345-test suite must stay
    green) whether or not it happens to be installed in a given venv;
    only code paths that actually exercise Nikon Scan look mode should
    ever require it."""
    try:
        import nikonlook_core
    except ImportError as exc:
        raise NikonLookUnavailable(
            "nikonlook_core is not importable. Nikon Scan look (beta) requires an "
            "editable install of the negfit-core package into this venv: "
            "`uv pip install -e ~/Downloads/digital-ice-2026` -- see "
            "digital-ice-2026/negfit/NEGPY-INTEGRATION-PLAN.md."
        ) from exc
    return nikonlook_core


def default_bundle_path() -> Optional[Path]:
    """The bundle shipped alongside nikonlook_core's OWN editable-install
    location: negfit/bundles/nikonlook-v1, resolved from
    `nikonlook_core.__file__` -- NOT a hardcoded absolute path, so this
    resolves correctly regardless of where digital-ice-2026 happens to be
    checked out on a given machine (the whole point of an editable
    install). Returns None if nikonlook_core isn't importable at all (a
    "no default available" probe used by `resolve_bundle_path`'s fallback
    chain -- unlike that function's own hard-failure callers, this one
    should never raise)."""
    try:
        nikonlook_core = _import_nikonlook_core()
    except NikonLookUnavailable:
        return None
    return Path(nikonlook_core.__file__).resolve().parent / "bundles" / "nikonlook-v1"


def resolve_bundle_path(configured: Optional[str]) -> Optional[Path]:
    """Resolution order: `configured` (settings.nikonlook.nikonlook_bundle_path)
    wins if set; else the `NEGFIT_BUNDLE` env var; else the
    package-relative default (see `default_bundle_path`). Returns None if
    nothing resolves -- callers decide how to handle that (
    `NikonLookProcessor.available()` treats it as "capability off";
    `NikonLookProcessor._load()` raises `NikonLookUnavailable`)."""
    if configured:
        return Path(configured).expanduser().resolve()
    env_path = os.getenv(NEGFIT_BUNDLE_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()
    return default_bundle_path()


class NikonLookProcessor:
    """Thin adapter: IR heal (existing NegPy logic, reused unmodified),
    then the negfit color model (nikonlook_core, imported not
    reimplemented). One instance per caller; the loaded Bundle is cached
    on the instance so repeated `.process()` calls (e.g. a preview
    followed by an export of the same file) don't re-parse the bundle's
    JSON every time.
    """

    def __init__(self, config: NikonLookConfig):
        self.config = config
        self._nikonlook_core = None
        self._bundle = None
        self._bundle_path: Optional[Path] = None

    @property
    def bundle_path(self) -> Optional[Path]:
        return resolve_bundle_path(self.config.nikonlook_bundle_path)

    def available(self) -> bool:
        """True if a bundle resolves to an existing directory AND
        nikonlook_core is importable -- the capability gate
        NEGPY-INTEGRATION-PLAN.md asks for ("mode visible only when a
        bundle path is configured"). Never raises -- safe to call from UI
        / status-reporting code that just wants a yes/no."""
        try:
            path = self.bundle_path
        except NikonLookUnavailable:
            return False
        return path is not None and path.is_dir()

    def _load(self):
        nikonlook_core = _import_nikonlook_core()
        path = self.bundle_path
        if path is None or not path.is_dir():
            raise NikonLookUnavailable(
                f"Nikon Scan look is enabled but no bundle was found. Checked: "
                f"settings.nikonlook.nikonlook_bundle_path={self.config.nikonlook_bundle_path!r}, "
                f"${NEGFIT_BUNDLE_ENV_VAR}={os.getenv(NEGFIT_BUNDLE_ENV_VAR)!r}, "
                f"package default={default_bundle_path()!r}. Point one of these at "
                f"a negfit bundle directory (e.g. negfit/bundles/nikonlook-v1)."
            )
        if self._bundle is None or self._bundle_path != path:
            self._bundle = nikonlook_core.load_bundle(path)
            self._bundle_path = path
            self._nikonlook_core = nikonlook_core
            logger.info(
                "NikonLookProcessor: loaded bundle %s (quality_tier=%s)",
                path,
                self._bundle.manifest.get("quality_tier"),
            )
        return self._nikonlook_core, self._bundle

    def process(
        self,
        raw_sensor_rgb: ImageBuffer,
        ir_buffer: Optional[np.ndarray],
        retouch_config: RetouchConfig,
        scale_factor: float,
    ) -> Tuple[ImageBuffer, Optional[bytes]]:
        """Runs IR heal (only if `retouch_config.ir_dust_remove` -- honors
        the user's existing retouch settings rather than forcing it on)
        then the negfit color model, on RAW LINEAR sensor RGB (pre-
        inversion, pre-white-balance -- see nikonlook_core's module
        docstring for the exact convention it expects; NegPy callers must
        supply this with `settings.process.linear_raw=True` decoding, see
        DarkroomEngine.process_nikonlook's guard).

        Only the IR-based heal is used here (`apply_ir_dust_removal`) --
        NOT the full `apply_dust_removal` composite, whose OTHER two
        mechanisms (luminance-based "auto" dust detection and manual heal
        strokes) are perceptually tuned for an already-inverted, photo-
        like positive image (both bracket through the working-space OETF
        -- see negpy CLAUDE.md's "Retouch runs in the linear island" /
        "dust *detection* is perceptual" note) and would misbehave on raw,
        un-inverted negative sensor data.

        Returns (device_rgb01, icc_bytes_or_None). `device_rgb01` is the
        FINAL look, already gamma-encoded in the bundle's own ICC space --
        callers must NOT run it through NegPy's normal working-space color
        management again (see image_processor.py's nikonlook export
        branch, which tags `icc_bytes` directly instead).
        """
        nikonlook_core, bundle = self._load()

        img = raw_sensor_rgb
        if retouch_config.ir_dust_remove and ir_buffer is not None:
            img, _mask = apply_ir_dust_removal(
                img,
                ir_buffer,
                # Same UI-sensitivity -> raw-threshold inversion
                # RetouchProcessor.process() applies (negpy/features/retouch/
                # processor.py) -- reproduced here, not re-derived, so this
                # mode's IR healing behaves identically to the normal pipeline's
                # for the same retouch settings.
                threshold=1.0 - retouch_config.ir_threshold,
                inpaint_radius=retouch_config.ir_inpaint_radius,
                scale_factor=scale_factor,
            )

        raw01 = np.asarray(img, dtype=np.float64)
        k = nikonlook_core.estimate_gains(raw01, bundle)
        device_rgb01 = nikonlook_core.apply(raw01, k, bundle)
        return device_rgb01.astype(np.float32), bundle.icc_bytes
