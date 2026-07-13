"""negpy/features/nikonlook/models.py -- config for the "Nikon Scan look
(beta)" conversion mode. See negpy/features/nikonlook/processor.py and
digital-ice-2026/negfit/NEGPY-INTEGRATION-PLAN.md for the full design.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class NikonLookConfig:
    """ADDITIVE, opt-in conversion mode: replaces NegPy's own inversion
    (normalization + print curve) with an externally-trained color model
    (a negfit "bundle": 3x3 matrix + monotone curves + per-frame exposure
    gain), applied via the `nikonlook_core` package (editable-installed
    from a sibling repo -- `uv pip install -e ~/Downloads/digital-ice-2026`,
    see NEGPY-INTEGRATION-PLAN.md). Off by default (`nikonlook_enabled=
    False`) so no existing workspace/edit ever changes behavior just by
    this field existing. Existing inversion modes (C41/BW/E6, the Flat
    render intent, RGB Scan) are completely untouched by this feature.

    Field names are prefixed `nikonlook_` (not bare `enabled`/`bundle_path`)
    because `WorkspaceConfig.to_dict()`/`from_flat_dict()` flatten every
    feature config into ONE shared key namespace (see negpy CLAUDE.md's
    "Data model" section) -- `RgbScanConfig` already owns a bare `enabled`
    field, so an unprefixed one here would silently collide and clobber it
    on (de)serialization. Mirrors `ExportConfig`'s own `contact_sheet_*`
    prefixing for the identical reason.

    `nikonlook_bundle_path`: None resolves via the `NEGFIT_BUNDLE` env var,
    else the package-relative default bundle shipped alongside
    `nikonlook_core` (negfit/bundles/nikonlook-v1) -- see
    `negpy.features.nikonlook.processor.resolve_bundle_path`. Set
    explicitly to pin a specific bundle version/location (tests do this).

    Quality note (mirrors the bundle's own manifest.json): ships as BETA.
    No shipping quality gate passes yet -- see the resolved bundle's
    manifest.json (`quality_tier`/`quality_note`/`gates` fields) for the
    honest, current numbers. There is currently no GUI surface for this
    mode (see NEGPY-INTEGRATION-PLAN.md's "honest gap" note) -- it is
    activated purely by constructing/persisting a WorkspaceConfig with
    `nikonlook.nikonlook_enabled=True`.
    """

    nikonlook_enabled: bool = False
    nikonlook_bundle_path: Optional[str] = None
