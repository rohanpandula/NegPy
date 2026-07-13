# nikonlook golden-regression fixture

`frame003_crop512.npz` is a 512x512 raw crop taken from
digital-ice-2026/negfit's frame003.pnm (native scanner-array rows
[2200:2712], cols [2600:3112], pre-upright -- orientation
doesn't matter for this fixture, see below), plus its EXPECTED
`nikonlook_core` output (bundle `nikonlook-v1`), used
by `tests/test_nikonlook_golden.py` as a plumbing tripwire: "did wiring
nikonlook_core into NegPy (bundle load, gain estimate, matrix+curve apply)
break," NOT a color-quality gate (negfit's own fit37/GATE7-RESULTS.json is
the real quality gate).

Regenerate via `python3 negfit/build_negpy_golden_fixture.py` in the negfit
repo (digital-ice-2026) -- only when bundle v1's math changes ON PURPOSE (a
new bundle version); the entire point of this fixture is to catch
UNINTENTIONAL drift between the two repos.

Arrays (npz keys):
- `raw_crop_u16` (H,W,3) uint16 -- raw scanner counts, same convention as a
  `.pnm` capture's payload (native scanner-array orientation; the NegPy
  test runs this through `DarkroomEngine.process_nikonlook` with identity
  geometry -- `rotation=0` -- so orientation is irrelevant to what's under
  test: the adapter plumbing, not scan geometry).
- `full_scale` scalar float64 -- divisor to reach raw01 = raw_crop_u16 /
  full_scale (65535.0 for this 16-bit capture).
- `expected_device_rgb_u16` (H,W,3) uint16 -- nikonlook_core.apply()'s
  output on this crop (estimate_gains -> apply, bundle v1), scaled to
  [0,65535].
- `expected_k` (3,) float64 -- nikonlook_core.estimate_gains()'s output on
  this crop (its own small plumbing check, independent of `apply`).
- `bundle_version`, `source_frame`, `crop_row0`, `crop_col0` -- provenance.
