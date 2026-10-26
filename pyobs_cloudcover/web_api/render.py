"""Render sky maps as PNG images (polar alt/az view, north up, east left, as seen looking up)."""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from pyobs_cloudcover.camera.bayer import debayer
from pyobs_cloudcover.camera.profile import Frame
from pyobs_cloudcover.sky_query import SkyPixelQuery, altaz_to_xyz, angle_to_chord

# dark blue clear sky, white clouds, black where there is no information
CLOUD_COLOURS = np.array([[25, 45, 100], [230, 230, 230]], dtype=np.uint8)
UNKNOWN = np.array([0, 0, 0], dtype=np.uint8)
# perceptually ordered ramp (dark purple -> yellow) for continuous values
RAMP = np.array([[48, 18, 59], [65, 100, 220], [30, 180, 190], [120, 220, 80], [250, 210, 40]], dtype=np.float64)


def _canvas_altaz(
    size: int, min_alt: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    c = (size - 1) / 2
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    dx, dy = x - c, y - c
    r = np.hypot(dx, dy) / c  # 0 at the zenith, 1 at min_alt
    alt = np.deg2rad(90 - r * (90 - min_alt))
    # north up, east left
    az = np.mod(np.arctan2(-dx, -dy), 2 * np.pi)
    return alt, az, r <= 1


def _sample(
    query: SkyPixelQuery,
    alt: npt.NDArray[np.float64],
    az: npt.NDArray[np.float64],
    inside: npt.NDArray[np.bool_],
    max_distance: float,
) -> npt.NDArray[np.float64]:
    values = np.full(alt.shape, np.nan)
    if len(query.grid) == 0:
        return values
    dist, idx = query.grid.tree.query(altaz_to_xyz(alt[inside], az[inside]), distance_upper_bound=max_distance)
    pixels = query.get_pixels()
    ok = np.isfinite(dist)
    sampled = np.full(len(dist), np.nan)
    sampled[ok] = pixels[idx[ok]]
    values[inside] = sampled
    return values


def _decorate(image: Image.Image, size: int, min_alt: float) -> None:
    draw = ImageDraw.Draw(image)
    c = (size - 1) / 2
    for alt in (30, 60):
        if alt > min_alt:
            r = c * (90 - alt) / (90 - min_alt)
            draw.ellipse([c - r, c - r, c + r, c + r], outline=(120, 120, 120))
    draw.ellipse([0, 0, size - 1, size - 1], outline=(160, 160, 160))
    for label, (x, y) in {"N": (c, 2), "S": (c, size - 14), "E": (4, c - 6), "W": (size - 12, c - 6)}.items():
        draw.text((x, y), label, fill=(255, 80, 80))


def render_cloud_map(query: SkyPixelQuery, size: int = 480, min_alt: float = 0.0) -> bytes:
    alt, az, inside = _canvas_altaz(size, min_alt)
    values = _sample(query, alt, az, inside, angle_to_chord(np.deg2rad(2.0)))
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[:] = UNKNOWN
    known = np.isfinite(values)
    rgb[known] = CLOUD_COLOURS[(values[known] > 0.5).astype(int)]
    image = Image.fromarray(rgb)
    _decorate(image, size, min_alt)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def render_value_map(query: SkyPixelQuery, low: float, high: float, size: int = 480, min_alt: float = 0.0) -> bytes:
    """Continuous map (e.g. limiting magnitude), ``low`` dark to ``high`` bright."""
    alt, az, inside = _canvas_altaz(size, min_alt)
    values = _sample(query, alt, az, inside, angle_to_chord(np.deg2rad(2.0)))
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    known = np.isfinite(values)
    t = np.clip((values[known] - low) / (high - low), 0, 1) * (len(RAMP) - 1)
    i = np.minimum(t.astype(int), len(RAMP) - 2)
    f = (t - i)[:, None]
    rgb[known] = (RAMP[i] * (1 - f) + RAMP[i + 1] * f).astype(np.uint8)
    image = Image.fromarray(rgb)
    _decorate(image, size, min_alt)
    draw = ImageDraw.Draw(image)
    draw.text((4, size - 14), f"{low:g} .. {high:g}", fill=(200, 200, 200))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def render_frame(frame: Frame, size: int = 720) -> bytes:
    """The camera image as the pipelines see it (canonical orientation), debayered, scaled to at most ``size``
    pixels and stretched for display: black at the 1st percentile of each channel, white at the 99.5th percentile,
    square-root stretch in between."""
    data: npt.NDArray[Any] = np.asarray(frame.data)
    if frame.bayer is not None and data.ndim == 2:
        data = debayer(data, frame.bayer)
    data = data.astype(np.float32)
    scale = size / max(data.shape[:2])
    if scale < 1:
        data = cv2.resize(data, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if data.ndim == 2:
        data = data[..., None]
    data = data - np.percentile(data, 1, axis=(0, 1))
    white = float(np.percentile(data, 99.5))
    stretched = np.sqrt(np.clip(data / white, 0, 1)) if white > 0 else np.zeros_like(data)
    rgb = (255 * stretched).astype(np.uint8)
    image = Image.fromarray(rgb[..., 0] if rgb.shape[2] == 1 else rgb)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()
