"""Reading image files written by capture software."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from astropy.io import fits

log = logging.getLogger(__name__)

FITS_SUFFIXES = (".fits", ".fit", ".fts", ".fits.gz", ".fits.fz", ".fz")
PIL_SUFFIXES = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
RAW_SUFFIXES = (".dng",)

_warned_lossy: set[str] = set()


@dataclass
class RawImage:
    """Image data as stored in the file: (H, W) mono or Bayer mosaic, or (H, W, 3) RGB."""

    data: npt.NDArray[Any]
    header: dict[str, Any] = field(default_factory=dict)
    path: str = ""
    mtime: float | None = None
    format: str = "fits"
    bit_depth: int | None = None  # what the file format says, if anything

    @property
    def is_rgb(self) -> bool:
        return self.data.ndim == 3


def read_image(path: str | os.PathLike[str]) -> RawImage:
    """Read FITS (mono, raw Bayer or 3-plane RGB), PNG/TIFF/JPEG (8 or 16 bit) or, with ``rawpy``, DNG."""
    path = Path(path)
    name = path.name.lower()
    mtime = path.stat().st_mtime
    if name.endswith(FITS_SUFFIXES):
        image = _read_fits(path)
    elif name.endswith(PIL_SUFFIXES):
        image = _read_pil(path)
    elif name.endswith(RAW_SUFFIXES):
        image = _read_dng(path)
    else:
        raise ValueError(f"Unsupported image format: {path.name}")
    image.path, image.mtime = str(path), mtime
    return image


def from_fits_hdu(data: npt.NDArray[Any], header: Any) -> RawImage:
    """Wrap FITS data + header (e.g. from a pyobs Image) like :func:`read_image` would."""
    data = np.asarray(data)
    if data.ndim == 3:
        if data.shape[0] == 3:
            data = np.moveaxis(data, 0, -1)
        elif data.shape[-1] != 3:
            raise ValueError(f"Can't interpret a FITS cube of shape {data.shape} as an image")
    elif data.ndim != 2:
        raise ValueError(f"Can't interpret FITS data of shape {data.shape} as an image")
    bitpix = int(header.get("BITPIX", 0)) if header is not None else 0
    return RawImage(data=data, header=dict(header or {}), format="fits",
                    bit_depth={8: 8, 16: 16}.get(bitpix))


def _read_fits(path: Path) -> RawImage:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                header = fits.Header(hdul[0].header)
                if hdu is not hdul[0]:
                    header.update(hdu.header)
                return from_fits_hdu(np.array(hdu.data), header)
    raise ValueError(f"No image data in {path}")


def _warn_lossy(path: Path, reason: str) -> None:
    key = f"{path.suffix}:{reason}"
    if key not in _warned_lossy:
        _warned_lossy.add(key)
        log.warning(
            "%s is %s. Stretched or compressed images lose the faint stars the night pipeline relies on; "
            "use raw FITS (or 16-bit PNG/TIFF) where the capture software allows it.", path.name, reason
        )


def _read_pil(path: Path) -> RawImage:
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        mode = str(im.mode)
        if mode in ("I;16", "I;16B", "I;16L", "I"):
            data = np.asarray(im, dtype=np.uint16 if mode.startswith("I;16") else np.int32)
            bits = 16
        elif mode in ("RGB", "L"):
            data = np.asarray(im)
            bits = 8
        else:
            data = np.asarray(im.convert("RGB"))
            bits = 8
        info = {str(k).upper(): v for k, v in im.info.items() if isinstance(v, (str, int, float))}
    if path.suffix.lower() in (".jpg", ".jpeg"):
        _warn_lossy(path, "a JPEG (lossy, usually stretched)")
    elif bits == 8:
        _warn_lossy(path, "8-bit")
    return RawImage(data=data, header=info, format=path.suffix.lower().lstrip("."), bit_depth=bits)


def _read_dng(path: Path) -> RawImage:
    try:
        import rawpy
    except ImportError:
        raise ValueError(f"Reading {path.name} needs the optional 'rawpy' package") from None
    with rawpy.imread(str(path)) as raw:
        data = np.array(raw.raw_image_visible, dtype=np.uint16)
        pattern = "".join(raw.color_desc.decode()[i] for i in raw.raw_pattern.ravel())
    header = {"BAYERPAT": pattern} if set(pattern) <= set("RGB") and len(pattern) == 4 else {}
    return RawImage(data=data, header=header, format="dng", bit_depth=None)
