"""tests/test_nikonlook_golden.py -- golden-regression plumbing tripwire
for the "Nikon Scan look (beta)" adapter (negpy/features/nikonlook/). See
digital-ice-2026/negfit/NEGPY-INTEGRATION-PLAN.md P3.

This is NOT a color-quality gate (negfit's own fit37/GATE7-RESULTS.json is
that -- no gate numerically passes yet, see bundle v1's manifest.json). It
asserts NegPy's OWN wiring around nikonlook_core (bundle load,
DarkroomEngine.process_nikonlook's geometry+adapter dispatch,
NikonLookProcessor's estimate_gains/apply calls, WorkspaceConfig
(de)serialization) still reproduces a frozen reference output for a frozen
input. A failure here means the PLUMBING broke (a signature changed, a
stage got reordered, a shape/dtype slipped, a config field collided) -- see
tests/fixtures/nikonlook/README.md for the fixture's provenance and how to
regenerate it if bundle v1's math changes ON PURPOSE.

Requires an editable install of negfit-core in this venv:
    uv pip install -e ~/Downloads/digital-ice-2026
(skipped, not failed, if that hasn't been done -- see the importorskip below).
"""

from pathlib import Path

import numpy as np
import pytest

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.geometry.models import GeometryConfig
from negpy.features.nikonlook.models import NikonLookConfig
from negpy.features.process.models import ProcessConfig
from negpy.features.retouch.models import RetouchConfig
from negpy.services.rendering.engine import DarkroomEngine

nikonlook_core = pytest.importorskip(
    "nikonlook_core",
    reason=(
        "nikonlook_core (negfit-core) is not installed in this venv -- "
        "run `uv pip install -e ~/Downloads/digital-ice-2026` (see "
        "digital-ice-2026/negfit/NEGPY-INTEGRATION-PLAN.md)"
    ),
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "nikonlook" / "frame003_crop512.npz"
BUNDLE_PATH = Path(nikonlook_core.__file__).resolve().parent / "bundles" / "nikonlook-v1"


# ---------------------------------------------------------------------------
# Minimal, INDEPENDENT CIEDE2000 (Sharma, Wu & Dalal 2005) + Adobe RGB (1998)
# colorimetry -- reimplemented here rather than imported from
# negfit/color.py. Two reasons: (a) negfit/color.py is not part of
# nikonlook_core's packaged/importable surface (digital-ice-2026/
# pyproject.toml ships only nikonlook_core.py -- see that file's module
# docstring on why it's kept dependency-minimal), and (b) a test oracle is
# better kept independent of the production code path it's checking, even
# when it could technically import it. Same published formula/constants
# negfit/color.py uses (see that file for the full derivation/citations
# and its own exhaustive unit test against the Sharma-Wu-Dalal reference
# table); this is a condensed, test-only port.
# ---------------------------------------------------------------------------

_ADOBE_RGB_GAMMA = 563.0 / 256.0
_ADOBE_RGB_TO_XYZ_D65 = np.array(
    [
        [0.5766690429, 0.1855582379, 0.1882286462],
        [0.2973449753, 0.6273635663, 0.0752914585],
        [0.0270313614, 0.0706888525, 0.9913375368],
    ]
)
_D65_WHITE_XYZ = np.array([0.9504559271, 1.0, 1.0890577508])


def _adobe_rgb_to_lab(device_rgb01: np.ndarray) -> np.ndarray:
    linear = np.clip(device_rgb01, 0.0, None) ** _ADOBE_RGB_GAMMA
    xyz = linear @ _ADOBE_RGB_TO_XYZ_D65.T
    r = xyz / _D65_WHITE_XYZ
    delta = 6.0 / 29.0
    f = np.where(r > delta**3, np.cbrt(r), r / (3 * delta**2) + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def _ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cbar7 = (0.5 * (C1 + C2)) ** 7
    G = 0.5 * (1.0 - np.sqrt(Cbar7 / (Cbar7 + 25.0**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    dLp = L2 - L1
    dCp = C2p - C1p
    C1C2p = C1p * C2p
    dhp_raw = h2p - h1p
    dhp = np.where(
        C1C2p == 0,
        0.0,
        np.where(dhp_raw > 180, dhp_raw - 360, np.where(dhp_raw < -180, dhp_raw + 360, dhp_raw)),
    )
    dHp = 2 * np.sqrt(C1C2p) * np.sin(np.radians(dhp) / 2.0)
    Lbarp = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)
    hsum, habs = h1p + h2p, np.abs(h1p - h2p)
    hbarp = np.where(
        C1C2p == 0,
        hsum,
        np.where(habs > 180, np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2), hsum / 2),
    )
    T = (
        1
        - 0.17 * np.cos(np.radians(hbarp - 30))
        + 0.24 * np.cos(np.radians(2 * hbarp))
        + 0.32 * np.cos(np.radians(3 * hbarp + 6))
        - 0.20 * np.cos(np.radians(4 * hbarp - 63))
    )
    d_theta = 30.0 * np.exp(-(((hbarp - 275.0) / 25.0) ** 2))
    Cbarp7 = Cbarp**7
    RC = 2 * np.sqrt(Cbarp7 / (Cbarp7 + 25.0**7))
    SL = 1 + (0.015 * (Lbarp - 50) ** 2) / np.sqrt(20 + (Lbarp - 50) ** 2)
    SC = 1 + 0.045 * Cbarp
    SH = 1 + 0.015 * Cbarp * T
    RT = -np.sin(np.radians(2 * d_theta)) * RC
    termL, termC, termH = dLp / SL, dCp / SC, dHp / SH
    return np.sqrt(np.clip(termL**2 + termC**2 + termH**2 + RT * termC * termH, 0.0, None))


def _load_fixture():
    d = np.load(FIXTURE_PATH)
    raw01 = d["raw_crop_u16"].astype(np.float64) / float(d["full_scale"])
    expected01 = d["expected_device_rgb_u16"].astype(np.float64) / 65535.0
    return raw01, expected01, d["expected_k"]


def _identity_nikonlook_config() -> WorkspaceConfig:
    """A WorkspaceConfig with nikonlook enabled and everything else at an
    identity/no-op setting (no rotation, no crop, IR healing off) so the
    fixture's raw crop passes through DarkroomEngine.process_nikonlook
    unchanged except for the color model itself -- exactly what the golden
    fixture's `expected_device_rgb_u16` was computed against."""
    return WorkspaceConfig(
        process=ProcessConfig(linear_raw=True),
        geometry=GeometryConfig(rotation=0, fine_rotation=0.0),
        retouch=RetouchConfig(ir_dust_remove=False),
        nikonlook=NikonLookConfig(nikonlook_enabled=True, nikonlook_bundle_path=str(BUNDLE_PATH)),
    )


def test_fixture_and_bundle_present():
    assert FIXTURE_PATH.exists(), f"golden fixture missing: {FIXTURE_PATH}"
    assert BUNDLE_PATH.is_dir(), f"bundle v1 missing: {BUNDLE_PATH}"


def test_estimate_gains_matches_golden():
    """nikonlook_core.estimate_gains() itself is stable for this input --
    isolates Layer A from Layer B before the combined adapter test below."""
    raw01, _expected01, expected_k = _load_fixture()
    bundle = nikonlook_core.load_bundle(BUNDLE_PATH)
    k = nikonlook_core.estimate_gains(raw01, bundle)
    np.testing.assert_allclose(k, expected_k, rtol=1e-9)


def test_nikonlook_adapter_reproduces_golden_output():
    """The actual plumbing tripwire: run the crop through
    DarkroomEngine.process_nikonlook -- NegPy's real integration point,
    exactly what ImageProcessor.run_pipeline/process_export dispatch to --
    and check the result against the frozen expected output within a
    tight DE00 tolerance. Not a color-quality gate (see module docstring);
    0.1 is a plumbing tolerance, not a perceptual-difference target."""
    raw01, expected01, _expected_k = _load_fixture()
    config = _identity_nikonlook_config()
    context = PipelineContext(
        scale_factor=1.0,
        original_size=raw01.shape[:2],
        process_mode=config.process.process_mode,
    )
    engine = DarkroomEngine()
    out = engine.process_nikonlook(raw01.astype(np.float32), config, context)

    assert out.shape == expected01.shape
    assert np.all(np.isfinite(out))

    de00 = _ciede2000(_adobe_rgb_to_lab(out.astype(np.float64)), _adobe_rgb_to_lab(expected01))
    median_de00 = float(np.median(de00))
    assert median_de00 < 0.1, f"nikonlook adapter output drifted from golden fixture: median DE00={median_de00:.4f}"

    # The export encoder needs this side-channel (image_processor.py's
    # _encode_export_nikonlook) -- verify process_nikonlook actually sets it.
    assert context.metrics.get("nikonlook_icc_bytes") is not None


def test_process_export_nikonlook_end_to_end(monkeypatch):
    """Exercises ImageProcessor.process_export's OWN new wiring (the
    _is_nikonlook branch, pipeline_metrics capture, _encode_export_nikonlook's
    TIFF+ICC-passthrough encode) -- the OTHER half of the plumbing that
    test_nikonlook_adapter_reproduces_golden_output (DarkroomEngine.
    process_nikonlook called directly) doesn't reach. Monkeypatches
    `_load_source_f32` (file decode -- pre-existing NegPy infrastructure,
    not part of this feature, out of scope here) to hand back the golden
    fixture directly, so this test verifies MY code's wiring without
    depending on how any particular file format gets decoded upstream.
    """
    import io

    import tifffile

    from negpy.domain.models import ExportConfig, ExportFormat

    raw01, expected01, _expected_k = _load_fixture()
    config = _identity_nikonlook_config()
    bundle = nikonlook_core.load_bundle(BUNDLE_PATH)

    from negpy.services.rendering.image_processor import ImageProcessor

    proc = ImageProcessor()
    monkeypatch.setattr(
        proc,
        "_load_source_f32",
        lambda file_path, params, fast_decode=False: (raw01.astype(np.float32), None, "Adobe RGB"),
    )

    export_settings = ExportConfig(export_fmt=ExportFormat.TIFF)
    data, fmt = proc.process_export("fake_frame003_crop.tif", config, export_settings, source_hash="test-hash")

    assert fmt == "tiff"
    assert data is not None, f"process_export failed: {fmt}"

    out_u16 = tifffile.imread(io.BytesIO(data))
    assert out_u16.shape == expected01.shape

    with tifffile.TiffFile(io.BytesIO(data)) as tf:
        icc_bytes = bytes(tf.pages[0].tags["InterColorProfile"].value)
    assert icc_bytes == bundle.icc_bytes, "exported TIFF's embedded ICC does not match the bundle's ICC exactly"

    out01 = out_u16.astype(np.float64) / 65535.0
    de00 = _ciede2000(_adobe_rgb_to_lab(out01), _adobe_rgb_to_lab(expected01))
    median_de00 = float(np.median(de00))
    assert median_de00 < 0.1, f"process_export nikonlook output drifted from golden fixture: median DE00={median_de00:.4f}"


def test_process_export_nikonlook_rejects_non_tiff(monkeypatch):
    """Honest scope guard: non-TIFF export formats must fail clearly (not
    silently mis-encode or embed no ICC) -- see _encode_export_nikonlook's
    docstring for why (DNG's writer has no ICC-tag support today)."""
    from negpy.domain.models import ExportConfig, ExportFormat

    from negpy.services.rendering.image_processor import ImageProcessor

    raw01, _expected01, _expected_k = _load_fixture()
    config = _identity_nikonlook_config()

    proc = ImageProcessor()
    monkeypatch.setattr(
        proc,
        "_load_source_f32",
        lambda file_path, params, fast_decode=False: (raw01.astype(np.float32), None, "Adobe RGB"),
    )

    export_settings = ExportConfig(export_fmt=ExportFormat.JPEG)
    data, error = proc.process_export("fake.jpg", config, export_settings, source_hash="test-hash")
    assert data is None
    assert "TIFF" in error


def test_nikonlook_requires_linear_raw():
    """Fail-loud guard: this mode assumes raw, non-white-balanced sensor
    RGB (see nikonlook_core's module docstring) -- linear_raw=False must
    raise, not silently produce wrong colors."""
    config = WorkspaceConfig(
        process=ProcessConfig(linear_raw=False),
        nikonlook=NikonLookConfig(nikonlook_enabled=True, nikonlook_bundle_path=str(BUNDLE_PATH)),
    )
    context = PipelineContext(scale_factor=1.0, original_size=(4, 4), process_mode=config.process.process_mode)
    engine = DarkroomEngine()
    with pytest.raises(ValueError, match="linear_raw"):
        engine.process_nikonlook(np.zeros((4, 4, 3), dtype=np.float32), config, context)


def test_nikonlook_disabled_by_default():
    """WorkspaceConfig() with no explicit nikonlook settings must have the
    mode off -- existing workspaces/edits never silently pick this up."""
    assert WorkspaceConfig().nikonlook.nikonlook_enabled is False


def test_workspace_config_roundtrip_nikonlook_fields():
    """Also guards against the enabled/bundle_path flat-key collision with
    RgbScanConfig's own `enabled` field (see NikonLookConfig's docstring)."""
    cfg = WorkspaceConfig(
        nikonlook=NikonLookConfig(nikonlook_enabled=True, nikonlook_bundle_path="/tmp/some-bundle"),
    )
    flat = cfg.to_dict()
    assert flat["nikonlook_enabled"] is True
    assert flat["nikonlook_bundle_path"] == "/tmp/some-bundle"
    assert flat["enabled"] is False  # RgbScanConfig's own field, untouched

    restored = WorkspaceConfig.from_flat_dict(flat)
    assert restored.nikonlook.nikonlook_enabled is True
    assert restored.nikonlook.nikonlook_bundle_path == "/tmp/some-bundle"
    assert restored.rgbscan.enabled is False


def test_workspace_config_backcompat_without_nikonlook_fields():
    """Old config dicts (saved before this feature existed) must
    deserialize with the mode off, not raise."""
    cfg = WorkspaceConfig.from_flat_dict({})
    assert cfg.nikonlook.nikonlook_enabled is False
    assert cfg.nikonlook.nikonlook_bundle_path is None


def test_nikonlook_processor_unavailable_without_bundle(tmp_path):
    from negpy.features.nikonlook.processor import NikonLookProcessor, NikonLookUnavailable

    missing = tmp_path / "does-not-exist"
    proc = NikonLookProcessor(NikonLookConfig(nikonlook_enabled=True, nikonlook_bundle_path=str(missing)))
    assert proc.available() is False
    with pytest.raises(NikonLookUnavailable):
        proc.process(np.zeros((4, 4, 3), dtype=np.float32), None, RetouchConfig(), 1.0)


def test_nikonlook_processor_available_with_real_bundle():
    from negpy.features.nikonlook.processor import NikonLookProcessor

    proc = NikonLookProcessor(NikonLookConfig(nikonlook_enabled=True, nikonlook_bundle_path=str(BUNDLE_PATH)))
    assert proc.available() is True
