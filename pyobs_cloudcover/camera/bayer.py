"""Bayer patterns: where they come from (headers), how orientation changes them, and debayering."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.camera.orientation import Orientation

log = logging.getLogger(__name__)

BAYER_PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")

# OpenCV names Bayer patterns after the second row's 2nd/3rd pixels, so its "BG" is our RGGB
_CV2_CODES = {
    "RGGB": cv2.COLOR_BayerBG2RGB,
    "BGGR": cv2.COLOR_BayerRG2RGB,
    "GRBG": cv2.COLOR_BayerGB2RGB,
    "GBRG": cv2.COLOR_BayerGR2RGB,
}


def _cell(pattern: str) -> npt.NDArray[np.str_]:
    return np.array([[pattern[0], pattern[1]], [pattern[2], pattern[3]]])


def _transform(pattern: str, height: int, width: int, op: Any) -> str:
    """Apply an array operation to a mosaic of the given size and read back the pattern at the origin.

    Only the parity of the size matters, so a small proxy with the same parity stands in for the image.
    """
    proxy = np.tile(_cell(pattern), (2, 2))[: 2 + height % 2, : 2 + width % 2]
    out = op(proxy)
    return "".join(out[:2, :2].ravel())


def flip_rows(pattern: str, height: int) -> str:
    return _transform(pattern, height, 2, lambda a: a[::-1, :])


def shift(pattern: str, x_offset: int, y_offset: int) -> str:
    """Pattern seen from pixel (x_offset, y_offset) of a mosaic with ``pattern`` at its origin."""
    return "".join(np.roll(_cell(pattern), (-y_offset, -x_offset), axis=(0, 1)).ravel())


def to_canonical(pattern: str, orientation: Orientation, height: int, width: int) -> str:
    """Pattern at the origin of the canonical image, given the pattern at the origin of the raw image."""
    return _transform(pattern, height, width, orientation.to_canonical)


def pattern_from_header(header: Mapping[str, Any], height: int) -> str | None:
    """Bayer pattern of the data array's row 0, as stored in the file, from ``BAYERPAT``/``COLORTYP``.

    ``BAYERPAT`` describes the image displayed top-down. Files with ``ROWORDER = 'BOTTOM-UP'`` store the rows
    the other way round, so the pattern of the stored array is the flipped one. ``XBAYROFF``/``YBAYROFF`` shift
    the pattern.
    """
    value = None
    for key in ("BAYERPAT", "COLORTYP"):
        if key in header and str(header[key]).strip().upper() in BAYER_PATTERNS:
            value = str(header[key]).strip().upper()
            break
    if value is None:
        return None
    value = shift(value, int(header.get("XBAYROFF", 0) or 0), int(header.get("YBAYROFF", 0) or 0))
    if str(header.get("ROWORDER", "TOP-DOWN")).strip().upper() == "BOTTOM-UP":
        value = flip_rows(value, height)
    return value


def debayer(image: npt.NDArray[Any], pattern: str) -> npt.NDArray[np.float32]:
    """Bilinear demosaicing to an (H, W, 3) RGB float image."""
    if pattern not in _CV2_CODES:
        raise ValueError(f"Unknown Bayer pattern {pattern}")
    data = image if image.dtype in (np.uint8, np.uint16) else np.clip(image, 0, 65535).astype(np.uint16)
    return np.asarray(cv2.cvtColor(np.ascontiguousarray(data), _CV2_CODES[pattern]), dtype=np.float32)
