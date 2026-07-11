import io
import os
import struct
import tempfile

import numpy as np
import tifffile

from negpy.infrastructure.scanners.result import ScanResult
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


def _to_uint16(arr: np.ndarray) -> np.ndarray:
    """Convert array to uint16. For uint8, replicate byte (x<<8 | x) so 8-bit
    values span the full 16-bit range instead of being capped at 255."""
    if arr.dtype == np.uint16:
        return arr
    if arr.dtype == np.uint8:
        a16 = arr.astype(np.uint16)
        return (a16 << 8) | a16
    return arr.astype(np.uint16)


def _write_temp_tiff(data: np.ndarray, target_path: str, *, photometric: str) -> str:
    """Write `data` to a temp TIFF next to `target_path`. Returns the temp path.

    Caller commits it (os.replace to the real path) and is responsible for
    cleaning up the temp file if anything downstream fails. On failure here,
    the temp file is cleaned up before re-raising and `target_path` itself is
    never touched.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".tif", dir=os.path.dirname(target_path) or ".")
    os.close(fd)
    try:
        tifffile.imwrite(tmp_path, data, photometric=photometric, compression="lzw")
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return tmp_path


def write_tiff_16bit(result: ScanResult, path: str) -> str:
    """Write ScanResult to 16-bit TIFF. IR written as sidecar `<basename>_IR.tif`.

    Transactional as a unit: RGB and (if present) IR are both written to temp
    files first, and the real `path` / `ir_path` are only touched (via
    os.replace) once every payload is fully on disk. If either temp write
    fails, neither final path is touched. If the IR temp write succeeds but
    its commit (os.replace) fails after the RGB commit already landed, the
    RGB commit is rolled back too — this never leaves an orphan RGB file with
    no IR when IR was requested. When this write has no IR, a stale
    `<basename>_IR.tif` sidecar left over from a previous IR-enabled write of
    the same target is removed so it can't be silently misattributed to the
    new IR-less RGB frame (TiffLoader auto-discovers `_IR.tif` by filename
    alone, with no cross-check against the RGB it's paired with).

    Returns final RGB path.
    """
    if not path.lower().endswith((".tif", ".tiff")):
        path = path + ".tif"

    rgb = _to_uint16(result.rgb)
    base = os.path.splitext(path)[0]
    ir_path = f"{base}_IR.tif"
    has_ir = result.ir is not None

    # Phase 1: write both payloads to temp files. Nothing under `path` or
    # `ir_path` is touched here, so a failure at this stage (bad array,
    # codec error, disk full) leaves the filesystem exactly as it was.
    tmp_rgb = _write_temp_tiff(rgb, path, photometric="rgb")
    tmp_ir = None
    if has_ir:
        try:
            ir_data = _to_uint16(result.ir)
            tmp_ir = _write_temp_tiff(ir_data, ir_path, photometric="minisblack")
        except Exception:
            os.unlink(tmp_rgb)
            raise

    # Phase 2: commit. Both temp files are known-good at this point; only a
    # filesystem-level rename can fail from here.
    try:
        os.replace(tmp_rgb, path)
    except Exception:
        if os.path.exists(tmp_rgb):
            os.unlink(tmp_rgb)
        if tmp_ir and os.path.exists(tmp_ir):
            os.unlink(tmp_ir)
        raise

    if tmp_ir is not None:
        try:
            
            os.replace(tmp_ir, ir_path)
        except Exception:
            # RGB already committed but its IR pair didn't make it — undo
            # the RGB commit rather than leave an orphan (IR was requested).
            if os.path.exists(path):
                os.unlink(path)
            if os.path.exists(tmp_ir):
                os.unlink(tmp_ir)
            raise
    elif os.path.exists(ir_path):
        # No IR this write; drop a stale sidecar from a previous IR-enabled
        # write of this same target so it isn't silently misattributed to
        # the new IR-less RGB frame.
        os.unlink(ir_path)

    # Mask marking which IR pixels the scanner actually sampled. The loader
    # fails closed on a malformed one, so write {0,255} the reader accepts.
    if result.ir_valid_mask is not None:
        base = os.path.splitext(path)[0]
        valid_path = f"{base}_IR_VALID.tif"
        valid_data = np.asarray(result.ir_valid_mask).astype(np.uint8) * 255
        fd_v, tmp_v = tempfile.mkstemp(suffix=".tif", dir=os.path.dirname(valid_path) or ".")
        os.close(fd_v)
        try:
            tifffile.imwrite(tmp_v, valid_data, photometric="minisblack", compression="zlib", predictor=True)
            os.replace(tmp_v, valid_path)
        except Exception:
            if os.path.exists(tmp_v):
                os.unlink(tmp_v)
            raise

    return path


def write_dng_linear(result: ScanResult, path: str) -> str:
    """Write ScanResult to an uncompressed 16-bit LinearRaw DNG via tifffile.

    A LinearRaw DNG is a single-IFD TIFF plus a few DNG tags. If result.ir is
    present it is stacked as an extra sample. Atomic write; returns final path.
    """
    if not path.lower().endswith(".dng"):
        path = path + ".dng"

    rgb = _to_uint16(result.rgb)

    if result.ir is not None:
        ir = result.ir
        if ir.ndim == 2:
            ir = ir[:, :, np.newaxis]
        ir = _to_uint16(ir)
        full_array = np.dstack([rgb, ir])
    else:
        full_array = np.ascontiguousarray(rgb)

    model = result.device_model
    # (code, dtype, count, value, writeonce); NewSubfileType=0 is required or LibRaw rejects the DNG.
    extratags = [
        (254, 4, 1, 0, True),  # NewSubfileType
        (50706, 1, 4, (1, 4, 0, 0), True),  # DNGVersion
        (50707, 1, 4, (1, 0, 0, 0), True),  # DNGBackwardVersion
        (274, 3, 1, 1, True),  # Orientation
        (271, 2, len(model) + 1, model, True),  # Make
        (272, 2, len(model) + 1, model, True),  # Model
    ]
    payload = _encode_dng(full_array, extratags)

    fd, tmp_path = tempfile.mkstemp(suffix=".dng", dir=os.path.dirname(path) or ".")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(payload)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return path


def _encode_dng(full_array: np.ndarray, extratags: list) -> bytes:
    """Encode an RGB(+IR) uint16 array as LinearRaw DNG bytes.

    RGB is written with the RGB photometric so tifffile emits a clean 3 *color*
    samples with no ExtraSamples (matching pidng); the PhotometricInterpretation
    tag is then patched to LinearRaw (34892), which DNG requires. Marking colour
    planes as ExtraSamples instead makes some raw processors treat the file as a
    1-channel sensor + aux planes and mis-demosaic it.

    The IR (4-sample) case keeps the LINEAR_RAW photometric with the extra planes
    declared as extra samples — there the 4th plane genuinely is infrared, and
    tifffile has no clean 4-colour-sample form.
    """
    buf = io.BytesIO()
    if full_array.shape[-1] == 3:
        tifffile.imwrite(buf, full_array, photometric=tifffile.PHOTOMETRIC.RGB, compression=None, metadata=None, extratags=extratags)
        data = bytearray(buf.getvalue())
        with tifffile.TiffFile(io.BytesIO(bytes(data))) as tf:
            offset = tf.pages[0].tags["PhotometricInterpretation"].valueoffset
            byteorder = tf.byteorder
        struct.pack_into(byteorder + "H", data, offset, 34892)  # RGB(2) → LinearRaw(34892)
        return bytes(data)

    extrasamples = (0,) * (full_array.shape[-1] - 1)
    tifffile.imwrite(
        buf,
        full_array,
        photometric=tifffile.PHOTOMETRIC.LINEAR_RAW,
        compression=None,
        metadata=None,
        extrasamples=extrasamples,
        extratags=extratags,
    )
    return buf.getvalue()
