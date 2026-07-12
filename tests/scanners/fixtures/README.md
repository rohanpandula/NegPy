# LS-5000 roll-registration regression fixture

`ls5000_roll_registration_frame03.npz` is a compact replay fixture derived
from the successful 24-frame Nikon Super Coolscan LS-5000 ED sweep captured on
2026-07-11 at 400 dpi.

It retains every scanner row but only 64 evenly spaced columns, quantized from
16 to 8 bits. The selected frames cover the cut first exposure, frames 02/03,
the first directly visible transport stop, the dark frame 21, and late-roll
stops used by the robust tail model. The aligned frame-03 sample comes from the
hardware-verified closed-loop rescan.

This reduction keeps the edge, film-base, and repeated-row transport signals
needed by the registration code while holding the fixture below 0.5 MB. It is
regression data, not a display-quality reproduction of the photographs.
