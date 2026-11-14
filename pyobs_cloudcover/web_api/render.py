"""Render sky maps as PNG images (polar alt/az view, north up, east left, as seen looking up)."""

from __future__ import annotations

import datetime
import io
from functools import cache
from importlib import metadata
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from pyobs_cloudcover.camera.bayer import debayer
from pyobs_cloudcover.camera.profile import Frame
from pyobs_cloudcover.sky_query import SkyPixelQuery, altaz_to_xyz, angle_to_chord
from pyobs_cloudcover.world_model.world_model import WorldModel

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


@cache
def credit_lines() -> tuple[str, str]:
    """Name and version of this package, drawn into every rendered image so a saved map says where it came from."""
    try:
        version = metadata.version("pyobs-allsky-cloudcover")
    except metadata.PackageNotFoundError:
        version = "unknown"
    return "pyobs-allsky-cloudcover", f"v{version}"


def _credit(image: Image.Image, box: bool = False) -> None:
    """Name and version, small, in the bottom-right corner (outside the sky circle on maps). ``box``: on a dark
    background box, for camera images."""
    draw = ImageDraw.Draw(image)
    lines = credit_lines()
    width, height = image.size
    widths = [draw.textlength(line) for line in lines]
    line_height = 11
    x_right, y = width - 4, height - 4 - line_height * len(lines)
    if box:
        draw.rectangle([x_right - max(widths) - 4, y - 3, width - 1, height - 1], fill=(0, 0, 0))
    for line, w in zip(lines, widths):
        draw.text((x_right - w, y), line, fill=(140, 140, 140))
        y += line_height


def _timestamp(image: Image.Image, time: datetime.datetime | None, box: bool = False) -> None:
    """Observation time (UT), small, in the top-left corner (outside the sky circle on maps)."""
    if time is None:
        return
    draw = ImageDraw.Draw(image)
    text = time.astimezone(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UT")
    if box:
        draw.rectangle([0, 0, draw.textlength(text) + 8, 17], fill=(0, 0, 0))
    draw.text((4, 4), text, fill=(200, 200, 200))


def _label(image: Image.Image, lines: list[str], box: bool = False) -> None:
    """What the image shows, small, in the bottom-left corner (outside the sky circle on maps); several lines are
    stacked upwards from the corner. ``box``: on a dark background box, for camera images."""
    lines = [line for line in lines if line]
    if not lines:
        return
    draw = ImageDraw.Draw(image)
    line_height = 11
    y = image.size[1] - 4 - line_height * len(lines)
    if box:
        width = max(draw.textlength(line) for line in lines)
        draw.rectangle([0, y - 3, width + 8, image.size[1] - 1], fill=(0, 0, 0))
    for line in lines:
        draw.text((4, y), line, fill=(200, 200, 200))
        y += line_height


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
    _credit(image)


def render_cloud_map(
    query: SkyPixelQuery,
    size: int = 480,
    min_alt: float = 0.0,
    time: datetime.datetime | None = None,
    label: str = "Cloud map",
) -> bytes:
    alt, az, inside = _canvas_altaz(size, min_alt)
    values = _sample(query, alt, az, inside, angle_to_chord(np.deg2rad(2.0)))
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[:] = UNKNOWN
    known = np.isfinite(values)
    rgb[known] = CLOUD_COLOURS[(values[known] > 0.5).astype(int)]
    image = Image.fromarray(rgb)
    _decorate(image, size, min_alt)
    _timestamp(image, time)
    _label(image, [label])
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def render_value_map(
    query: SkyPixelQuery,
    low: float,
    high: float,
    size: int = 480,
    min_alt: float = 0.0,
    time: datetime.datetime | None = None,
    label: str = "",
    unit: str = "",
) -> bytes:
    """Continuous map (e.g. limiting magnitude), ``low`` dark to ``high`` bright, with a colour bar (in ``unit``)
    top right and the label bottom left."""
    alt, az, inside = _canvas_altaz(size, min_alt)
    values = _sample(query, alt, az, inside, angle_to_chord(np.deg2rad(2.0)))
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    known = np.isfinite(values)
    rgb[known] = _ramp_colours((values[known] - low) / (high - low))
    image = Image.fromarray(rgb)
    _decorate(image, size, min_alt)
    _timestamp(image, time)
    _colour_bar(image, low, high, unit)
    _label(image, [label])
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _ramp_colours(t: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    """Colours of ``RAMP`` for positions ``t`` (0 = low end, 1 = high end; clipped)."""
    t = np.clip(t, 0, 1) * (len(RAMP) - 1)
    i = np.minimum(t.astype(int), len(RAMP) - 2)
    f = (t - i)[..., None]
    return np.asarray(RAMP[i] * (1 - f) + RAMP[i + 1] * f, dtype=np.uint8)


def tick_values(low: float, high: float, max_ticks: int = 6) -> list[float]:
    """Round values (steps of 1, 2 or 5 times a power of ten) from ``low`` to ``high``, at most ``max_ticks``."""
    if not high > low:
        return [low]
    span = high - low
    for exponent in range(int(np.floor(np.log10(span / max_ticks))), int(np.ceil(np.log10(span))) + 1):
        for factor in (1, 2, 5):
            step = factor * 10.0**exponent
            first = np.ceil(low / step - 1e-9) * step
            ticks = list(np.arange(first, high + step * 1e-9, step))
            if len(ticks) <= max_ticks:
                return [round(float(v), 10) for v in ticks]
    return [low, high]


def _colour_bar(image: Image.Image, low: float, high: float, unit: str) -> None:
    """Colour scale of a value map in the top-right corner (outside the sky circle): the ramp from ``low`` (left)
    to ``high`` (right) with round tick values below it and the unit to its left. Values outside the range are
    drawn in the end colours."""
    width, _ = image.size
    length = max(60, round(0.2 * width))
    x0, x1, y0, y1 = width - 8 - length, width - 8, 6, 14
    ramp = _ramp_colours(np.linspace(0, 1, length))
    image.paste(Image.fromarray(np.repeat(ramp[None, :, :], y1 - y0, axis=0)), (x0, y0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([x0 - 1, y0 - 1, x1, y1], outline=(160, 160, 160))
    for value in tick_values(low, high):
        x = x0 + (value - low) / (high - low) * (length - 1) if high > low else x0
        draw.line([x, y1, x, y1 + 3], fill=(200, 200, 200))
        text = f"{value:g}"
        tw = draw.textlength(text)
        draw.text((min(max(x - tw / 2, x0 - 4), width - 2 - tw), y1 + 4), text, fill=(200, 200, 200))
    if unit:
        draw.text((x0 - 6 - draw.textlength(unit), y0 - 2), unit, fill=(200, 200, 200))


def _display_rgb(frame: Frame, max_size: int) -> npt.NDArray[np.uint8]:
    """The frame debayered, scaled to at most ``max_size`` pixels and made displayable (H, W, 3).

    8-bit images (JPEG/PNG from capture software) are already processed for display and shown as they are; a
    contrast stretch would shift their colours. Deeper, linear data are stretched: black at the 1st percentile of
    each channel, white at the 99.5th percentile, square root in between."""
    data: npt.NDArray[Any] = np.asarray(frame.data)
    if frame.bayer is not None and data.ndim == 2:
        data = debayer(data, frame.bayer)
    data = data.astype(np.float32)
    scale = max_size / max(data.shape[:2])
    if scale < 1:
        data = cv2.resize(data, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if data.ndim == 2:
        data = data[..., None]
    if frame.saturation <= 255:
        rgb = np.clip(data, 0, 255).astype(np.uint8)
    else:
        data = data - np.percentile(data, 1, axis=(0, 1))
        white = float(np.percentile(data, 99.5))
        stretched = np.sqrt(np.clip(data / white, 0, 1)) if white > 0 else np.zeros_like(data)
        rgb = (255 * stretched).astype(np.uint8)
    if rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    return np.ascontiguousarray(rgb)


def _png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def render_frame(frame: Frame, size: int = 720, label: str = "Camera image") -> bytes:
    """The camera image as the pipelines see it (canonical orientation), scaled to at most ``size`` pixels, with
    its observation time."""
    image = Image.fromarray(_display_rgb(frame, size))
    _timestamp(image, frame.obs_time, box=True)
    _credit(image, box=True)
    _label(image, [label], box=True)
    return _png(image)


def render_sky_image(
    frame: Frame, model: WorldModel, size: int = 720, min_alt: float = 0.0, label: str = "Sky (north up)"
) -> bytes:
    """The camera image reprojected onto the sky, in the same polar view as the maps (north up, east left, zenith
    in the centre, ``min_alt`` at the edge), so it can be laid next to or over a cloud map. ``model`` is the camera's
    world model in canonical full-resolution pixels; sky the camera doesn't see is black."""
    alt, az, inside = _canvas_altaz(size, min_alt)
    h, w = np.asarray(frame.data).shape[:2]
    # sample from a copy at roughly the canvas's resolution, so the downscaling averages instead of aliasing
    rgb = _display_rgb(frame, max(size * 2, 1))
    factor = rgb.shape[1] / w
    with np.errstate(invalid="ignore"):
        x, y = (np.asarray(v, dtype=np.float64) for v in model.altaz_to_pix(alt[inside], az[inside]))
    map_x = np.full(alt.shape, -1.0, dtype=np.float32)
    map_y = np.full(alt.shape, -1.0, dtype=np.float32)
    ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    sel = np.flatnonzero(inside.ravel())[ok]
    # pixel centres: full-resolution pixel i covers [i - 0.5, i + 0.5]
    map_x.ravel()[sel] = (x[ok] + 0.5) * factor - 0.5
    map_y.ravel()[sel] = (y[ok] + 0.5) * factor - 0.5
    sky = cv2.remap(rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    image = Image.fromarray(sky)
    _decorate(image, size, min_alt)
    _timestamp(image, frame.obs_time)
    _label(image, [label])
    return _png(image)


def _wrap(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, width: float
) -> list[str]:
    lines: list[str] = []
    for word in text.split():
        if lines and draw.textlength(lines[-1] + " " + word, font=font) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines


def render_placeholder(
    title: str,
    detail: str,
    width: int,
    height: int,
    polar: bool = False,
    time: datetime.datetime | None = None,
    label: str = "",
) -> bytes:
    """A stand-in image of the given size saying why there is no real one (see :mod:`placeholders`). ``polar``:
    draw the empty sky circle of the maps behind the text. ``time``: when this was decided (shown top left).
    ``label``: what the real image would be (bottom left). Line breaks in ``detail`` start new paragraphs."""
    image = Image.new("RGB", (width, height), (24, 27, 31))
    if polar:
        _decorate(image, width, 0.0)
    else:
        _credit(image)
    draw = ImageDraw.Draw(image)
    scale = min(width, height) / 480
    title_font = ImageFont.load_default(size=max(14, int(26 * scale)))
    detail_font = ImageFont.load_default(size=max(11, int(16 * scale)))
    lines = [(title, title_font, (235, 235, 235), 8 + int(6 * scale))]
    for paragraph in detail.split("\n"):
        wrapped = _wrap(draw, paragraph, detail_font, 0.7 * width)
        lines += [(line, detail_font, (170, 170, 170), 6) for line in wrapped]
        if wrapped:  # a gap after each paragraph
            lines[-1] = (*lines[-1][:3], 6 + int(8 * scale))
    heights = [draw.textbbox((0, 0), text, font=font)[3] + gap for text, font, _, gap in lines]
    y = (height - sum(heights)) / 2
    # a dark panel behind the text, so the map's circles don't run through it
    text_width = max(draw.textlength(text, font=font) for text, font, _, _ in lines)
    pad = 14 * scale
    draw.rounded_rectangle(
        [(width - text_width) / 2 - pad, y - pad, (width + text_width) / 2 + pad, y + sum(heights) + pad / 2],
        radius=8 * scale,
        fill=(24, 27, 31),
        outline=(70, 75, 82),
    )
    for (text, font, colour, _), h in zip(lines, heights):
        draw.text(((width - draw.textlength(text, font=font)) / 2, y), text, font=font, fill=colour)
        y += h
    _timestamp(image, time)
    _label(image, [label])
    return _png(image)
