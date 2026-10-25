"""Camera profile: everything camera-specific, so the pipelines can stay camera agnostic."""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from astropy.io import fits
from pydantic import BaseModel, ConfigDict, Field, field_validator

from pyobs_cloudcover.camera import bayer as bayer_utils
from pyobs_cloudcover.camera.orientation import Orientation
from pyobs_cloudcover.camera.reader import RawImage
from pyobs_cloudcover.obs_time import parse_time, time_from_header

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

_COMMON_MAXIMA = (255, 1023, 4095, 16383, 65535)


class TimestampConfig(BaseModel):
    """Where the observation time comes from.

    ``auto`` tries the header keywords, then ``filename_format`` (if given), then the file's modification time.
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["auto", "header", "filename", "mtime"] = "auto"
    keywords: list[str] = Field(default_factory=lambda: ["DATE-OBS", "DATE_OBS"])
    filename_format: str | None = Field(
        default=None, description="strptime format of the file name, e.g. 'allsky_%Y%m%d-%H%M%S.fits'"
    )
    timezone: str = Field(default="UTC", description="Zone of timestamps that don't carry one (e.g. 'Europe/Berlin')")
    offset: float = Field(default=0.0, description="Seconds added to the timestamp (e.g. half the exposure time)")


class OrientationConfig(BaseModel):
    """Raw -> canonical: flip left/right, then up/down, then rotate counter-clockwise by ``rotate`` degrees."""

    model_config = ConfigDict(extra="forbid")

    flip_x: bool = False
    flip_y: bool = False
    rotate: Literal[0, 90, 180, 270] = 0

    def to_orientation(self) -> Orientation:
        return Orientation(self.flip_x, self.flip_y, self.rotate)


class CameraConfig(BaseModel):
    """The camera profile section of the module config. All paths may be empty."""

    model_config = ConfigDict(extra="forbid")

    bayer: str = Field(
        default="auto",
        description="'auto' (from BAYERPAT/COLORTYP + ROWORDER), 'none' for mono, or the pattern of the stored "
        "array's first row: RGGB, BGGR, GRBG, GBRG",
    )
    orientation: OrientationConfig = Field(default_factory=OrientationConfig)
    timestamp: TimestampConfig = Field(default_factory=TimestampConfig)
    saturation: float | None = Field(default=None, description="Saturation level in ADU; default: guessed")
    bit_depth: int | None = Field(default=None, description="ADC bits, used for the saturation level if given")
    dark: str = Field(default="", description="Master dark (FITS or .npy), in raw orientation")
    hot_pixels: str = Field(default="", description="Hot-pixel map (.npy or FITS, nonzero = hot), raw orientation")
    mask: str = Field(default="", description="Sky mask (.npy or image, nonzero = sky), canonical orientation")
    world_model: dict[str, Any] | None = Field(
        default=None, description="World model in canonical full-resolution pixel coordinates"
    )

    @field_validator("bayer")
    @classmethod
    def _check_bayer(cls, value: str) -> str:
        value = value.strip()
        if value.lower() in ("auto", "none"):
            return value.lower()
        if value.upper() not in bayer_utils.BAYER_PATTERNS:
            raise ValueError(f"bayer must be auto, none or one of {bayer_utils.BAYER_PATTERNS}")
        return value.upper()


@dataclass
class Frame:
    """An image in the canonical orientation, ready for the pipelines.

    ``data`` is (H, W) (mono or Bayer mosaic with pattern ``bayer`` at its origin) or (H, W, 3) RGB.
    """

    data: FloatArray
    obs_time: datetime.datetime
    bayer: str | None = None
    saturation: float = 65535.0
    exposure: float | None = None
    path: str = ""
    header: dict[str, Any] = field(default_factory=dict)

    @property
    def is_rgb(self) -> bool:
        return self.data.ndim == 3

    @property
    def is_color(self) -> bool:
        return self.is_rgb or self.bayer is not None

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.data.shape[0]), int(self.data.shape[1])


def load_array(path: str) -> npt.NDArray[Any]:
    """A 2D array from .npy, FITS or a PNG/TIFF image."""
    if path.lower().endswith(".npy"):
        return np.asarray(np.load(path))
    if path.lower().endswith((".fits", ".fit", ".fts", ".fits.gz", ".fz")):
        return np.asarray(fits.getdata(path))
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        return np.asarray(im.convert("L") if im.mode not in ("L", "I;16", "I") else im)


class CameraProfile:
    """Turns raw images of one camera into canonical :class:`Frame` objects."""

    def __init__(self, config: CameraConfig | dict[str, Any] | None = None) -> None:
        self.config = config if isinstance(config, CameraConfig) else CameraConfig.model_validate(config or {})
        self.orientation = self.config.orientation.to_orientation()
        self._dark = load_array(self.config.dark).astype(np.float64) if self.config.dark else None
        self._dark_exposure: float | None = None
        if self.config.dark and not self.config.dark.lower().endswith(".npy"):
            exptime = fits.getheader(self.config.dark).get("EXPTIME")
            self._dark_exposure = float(exptime) if exptime else None
        self._hot = load_array(self.config.hot_pixels).astype(bool) if self.config.hot_pixels else None
        self._mask = load_array(self.config.mask).astype(bool) if self.config.mask else None
        self._warned: set[str] = set()

    @property
    def mask(self) -> npt.NDArray[np.bool_] | None:
        """Sky mask in canonical orientation (True = sky), if configured."""
        return self._mask

    def _warn_once(self, key: str, msg: str, *args: Any) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(msg, *args)

    # ------------------------------------------------------------------ time

    def obs_time(self, raw: RawImage) -> datetime.datetime:
        cfg = self.config.timestamp
        name = Path(raw.path).name
        time: datetime.datetime | None = None
        if cfg.source in ("auto", "header"):
            time = time_from_header(raw.header, tuple(cfg.keywords), cfg.timezone)
            if time is None and cfg.source == "header":
                raise ValueError(f"No timestamp ({', '.join(cfg.keywords)}) in the header of {name or 'image'}")
        if time is None and cfg.source in ("auto", "filename") and cfg.filename_format:
            try:
                naive = datetime.datetime.strptime(name, cfg.filename_format)
            except ValueError:
                if cfg.source == "filename":
                    raise ValueError(f"File name {name} doesn't match {cfg.filename_format}") from None
            else:
                time = parse_time(naive.isoformat(), cfg.timezone)
        if time is None:
            if raw.mtime is None:
                raise ValueError(f"No timestamp for {name or 'image'}")
            self._warn_once("mtime", "Using file modification times as observation times (no timestamp found).")
            time = datetime.datetime.fromtimestamp(raw.mtime, tz=datetime.UTC)
        return time + datetime.timedelta(seconds=cfg.offset)

    # ------------------------------------------------------------------ calibration

    def saturation(self, raw: RawImage) -> float:
        if self.config.saturation is not None:
            return float(self.config.saturation)
        if self.config.bit_depth is not None:
            return float(2**self.config.bit_depth - 1)
        if raw.data.dtype == np.uint8 or raw.bit_depth == 8:
            return 255.0
        if np.issubdtype(raw.data.dtype, np.floating):
            return float(np.nanmax(raw.data))
        if raw.data.dtype == np.uint16:
            # ZWO & co. store 12/14-bit data left-shifted, so every value is a multiple of 4 or 16
            sample = raw.data.ravel()[:: max(1, raw.data.size // 100000)]
            if np.any(sample) and np.all(sample % 4 == 0):
                return 65535.0
        # smallest common ADC range that holds the data (12-bit data stored unshifted in 16-bit files, ...)
        peak = float(np.max(raw.data))
        return float(next((m for m in _COMMON_MAXIMA if peak <= m), peak))

    def _calibrate(self, raw: RawImage) -> FloatArray:
        data = raw.data.astype(np.float64)
        if self._dark is not None:
            if self._dark.shape != data.shape[:2]:
                raise ValueError(f"Dark {self._dark.shape} doesn't match the image {data.shape[:2]}")
            dark = self._dark
            exposure = _exposure(raw.header)
            if self._dark_exposure and exposure:
                dark = dark * (exposure / self._dark_exposure)
            data = data - (dark[..., None] if data.ndim == 3 else dark)
        if self._hot is not None:
            if self._hot.shape != data.shape[:2]:
                raise ValueError(f"Hot-pixel map {self._hot.shape} doesn't match the image {data.shape[:2]}")
            data = _replace_hot_pixels(data, self._hot)
        return data

    def bayer_pattern(self, raw: RawImage) -> str | None:
        """Pattern at the origin of the raw array, or None for mono/RGB."""
        if raw.is_rgb or self.config.bayer == "none":
            return None
        from_header = bayer_utils.pattern_from_header(raw.header, raw.data.shape[0])
        if self.config.bayer == "auto":
            return from_header
        if from_header is not None and from_header != self.config.bayer:
            self._warn_once(
                "bayer", "Configured Bayer pattern %s overrides %s from the header.", self.config.bayer, from_header
            )
        return self.config.bayer

    # ------------------------------------------------------------------ main entry

    def prepare(self, raw: RawImage) -> Frame:
        pattern = self.bayer_pattern(raw)
        height, width = raw.data.shape[:2]
        data = self._calibrate(raw)
        canonical = np.ascontiguousarray(self.orientation.to_canonical(data))
        if pattern is not None:
            pattern = bayer_utils.to_canonical(pattern, self.orientation, height, width)
        if self._mask is not None and self._mask.shape != canonical.shape[:2]:
            raise ValueError(
                f"Mask {self._mask.shape} doesn't match the image {canonical.shape[:2]} (canonical orientation)"
            )
        return Frame(
            data=canonical,
            obs_time=self.obs_time(raw),
            bayer=pattern,
            saturation=self.saturation(raw),
            exposure=_exposure(raw.header),
            path=raw.path,
            header=raw.header,
        )


def _exposure(header: dict[str, Any]) -> float | None:
    for key in ("EXPTIME", "EXPOSURE"):
        if key in header:
            try:
                return float(header[key])
            except (TypeError, ValueError):
                pass
    return None


def _replace_hot_pixels(data: FloatArray, hot: npt.NDArray[np.bool_]) -> FloatArray:
    """Replace hot pixels by the median of their same-colour neighbours (2 pixels away, so Bayer-safe)."""
    data = data.copy()
    ys, xs = np.nonzero(hot)
    if len(ys) == 0:
        return data
    h, w = hot.shape
    neighbours = []
    for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        ny, nx = np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)
        values = data[ny, nx].astype(np.float64)
        values[hot[ny, nx]] = np.nan
        neighbours.append(values)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            replacement = np.nanmedian(np.stack(neighbours), axis=0)
    data[ys, xs] = np.where(np.isnan(replacement), data[ys, xs], replacement)
    return data
