"""Render simulated all-sky images."""

from __future__ import annotations

import dataclasses
import datetime
import os
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from astropy.coordinates import AltAz, SkyCoord
from astropy.io import fits
from astropy.time import Time

from pyobs_cloudcover.data import bundled_catalog_path
from pyobs_cloudcover.simulation.clouds import CloudLayer, combined_extinction
from pyobs_cloudcover.simulation.headers import build_header, make_filename
from pyobs_cloudcover.simulation.profile import CameraProfile
from pyobs_cloudcover.simulation.scene import (
    Ephemeris,
    Horizon,
    Site,
    SkyBrightness,
    SkyConditions,
    airmass,
    daylight,
    mag_to_nl,
    moon_magnitude,
    nl_to_mag,
    sky_colour,
)
from pyobs_cloudcover.simulation.truth import TRUTH_ALT, TRUTH_AZ, GroundTruth

FloatArray = npt.NDArray[np.float64]

ARCSEC2_PER_SR = (180 / np.pi * 3600) ** 2
MOON_RADIUS = np.deg2rad(0.26)  # the sun's is about the same
SUN_MAGNITUDE = -26.74
VISIBLE_SNR = 5.0
# fixed seed for synthetic faint stars, so they stay put on the celestial sphere between frames and runs
FAINT_STAR_SEED = 20240308


@dataclass
class SimulatedFrame:
    raw: npt.NDArray[Any]
    header: fits.Header
    truth: GroundTruth
    filename: str
    time: datetime.datetime

    def write(self, directory: str | Path, truth: bool = True, filename: str | None = None) -> Path:
        """Write the FITS image (atomically, so watchers never see a partial file) and the truth sidecar.

        The truth file is written first, so it exists by the time the image appears.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (filename or self.filename)
        if truth:
            self.truth.save(GroundTruth.path_for(path))
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".", suffix=".tmp")
        os.close(fd)
        try:
            fits.PrimaryHDU(self.raw, header=self.header).writeto(tmp, overwrite=True)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return path


@dataclass
class _Geometry:
    alt: FloatArray
    az: FloatArray
    theta: FloatArray
    solid_angle: FloatArray  # arcsec^2 per pixel
    in_lens: npt.NDArray[np.bool_]
    vignetting: FloatArray
    channel_response: FloatArray  # per raw pixel
    channel_index: npt.NDArray[np.intp]  # 0/1/2 = R/G/B per raw pixel (all 1 for mono)


@dataclass
class Simulator:
    """Renders all-sky images of a site for a camera profile, with ground truth.

    Args:
        profile: Camera profile (see ``presets``).
        site: Observing site.
        sky: Sky brightness and atmosphere.
        clouds: Cloud layers (empty list = clear sky).
        horizon: Local horizon obstructions, or None.
        seed: Seed for the noise; frames at different times get different noise.
        cloud_threshold: Cloud extinction (mag) from which the ground truth calls a position cloudy.
        faint_star_limit: Synthetic stars fainter than the catalogue are added down to this V magnitude.
        reference_time: Time at which clouds are at their initial position (default: first rendered frame).
        auto_exposure: Shorten the exposure when the sky would saturate (twilight, day, bright moon).
        moon: False leaves the moon out (a what-if: no disc, no moonlight); the truth then reports it below the
            horizon.
    """

    profile: CameraProfile
    site: Site = field(default_factory=Site)
    sky: SkyConditions = field(default_factory=SkyConditions)
    clouds: list[CloudLayer] = field(default_factory=list)
    horizon: Horizon | None = None
    seed: int = 0
    cloud_threshold: float = 0.5
    faint_star_limit: float = 8.5
    reference_time: datetime.datetime | None = None
    auto_exposure: bool = True
    moon: bool = True

    def __post_init__(self) -> None:
        self._observer = self.site.observer()
        self._model = self.profile.world_model()
        self._geometry: _Geometry | None = None
        self._catalog = _load_catalog()
        self._faint = _faint_stars(self.faint_star_limit)
        hot_rng = np.random.default_rng(zlib.crc32(self.profile.name.encode()))
        n_hot = int(self.profile.hot_pixel_fraction * self.profile.width * self.profile.height)
        self._hot_y = hot_rng.integers(0, self.profile.height, n_hot)
        self._hot_x = hot_rng.integers(0, self.profile.width, n_hot)
        self._hot_rate = self.profile.hot_pixel_rate * hot_rng.lognormal(0, 0.7, n_hot)

    # ------------------------------------------------------------------ geometry

    def geometry(self) -> _Geometry:
        if self._geometry is None:
            h, w = self.profile.canonical_shape
            y, x = np.mgrid[0:h, 0:w].astype(np.float64)
            alt, az = self._model.pix_to_altaz(x, y)
            alt, az = np.asarray(alt), np.asarray(az)
            theta = np.pi / 2 - alt
            in_lens = ~np.isnan(alt)
            theta0 = np.where(in_lens, theta, 0.0)
            # dOmega / dA = sin(theta) / (r dr/dtheta)
            eps = 1e-4
            with np.errstate(invalid="ignore", divide="ignore"):
                r = self._model.radius(theta0)
                dr = (self._model.radius(theta0 + eps) - self._model.radius(theta0)) / eps
                omega = np.where(theta0 > 1e-6, np.sin(theta0) / (r * dr), 1 / dr**2)
            vignetting = np.clip(1 - self.profile.vignetting * (theta0 / (np.pi / 2)) ** 2, 0.05, 1)
            self._geometry = _Geometry(
                alt=alt,
                az=az,
                theta=theta,
                solid_angle=np.where(in_lens, omega * ARCSEC2_PER_SR, 0.0),
                in_lens=in_lens,
                vignetting=vignetting,
                channel_response=self._channel_response(),
                channel_index=self._channel_index(),
            )
        return self._geometry

    def _channel_index(self) -> npt.NDArray[np.intp]:
        p = self.profile
        if p.bayer is None:
            return np.ones((p.height, p.width), dtype=np.intp)
        cell = np.array([["RGB".index(c) for c in p.bayer[:2]], ["RGB".index(c) for c in p.bayer[2:]]])
        return np.tile(cell, (p.height // 2 + 1, p.width // 2 + 1))[: p.height, : p.width]

    def _channel_response(self) -> FloatArray:
        p = self.profile
        if p.bayer is None:
            return np.ones((p.height, p.width))
        resp = dict(zip("RGB", p.channel_response))
        cell = np.array([[resp[p.bayer[0]], resp[p.bayer[1]]], [resp[p.bayer[2]], resp[p.bayer[3]]]])
        return np.tile(cell, (p.height // 2 + 1, p.width // 2 + 1))[: p.height, : p.width]

    @property
    def _brightness(self) -> SkyBrightness:
        # follows changes of ``sky`` between frames
        return SkyBrightness(self.sky)

    @property
    def world_model(self) -> Any:
        return self._model

    # ------------------------------------------------------------------ rendering

    def render(self, time: datetime.datetime, exposure: float | None = None) -> SimulatedFrame:
        if time.tzinfo is None:
            time = time.replace(tzinfo=datetime.UTC)
        if self.reference_time is None:
            self.reference_time = time
        seconds = (time - self.reference_time).total_seconds()
        p, geo = self.profile, self.geometry()
        atime = Time(time)
        eph = Ephemeris.compute(self._observer, atime)
        if not self.moon:
            eph = dataclasses.replace(eph, moon_alt=-np.pi / 2)
        k = self.sky.extinction

        on_sky = geo.in_lens & (geo.alt > 0)
        obstructed = np.zeros_like(on_sky)
        if self.horizon is not None:
            obstructed[geo.in_lens] = self.horizon.is_obstructed(geo.alt[geo.in_lens], geo.az[geo.in_lens])
        visible_sky = on_sky & ~obstructed

        # --- clouds and sky background (canonical image, e-/s)
        extinction = np.zeros(geo.alt.shape)
        extinction[on_sky] = combined_extinction(self.clouds, geo.alt[on_sky], geo.az[on_sky], seconds)
        transmission = 10 ** (-0.4 * extinction)

        sky_alt = np.where(geo.in_lens, np.clip(geo.alt, 0, None), np.pi / 2)
        sky_az = np.where(geo.in_lens, geo.az, 0.0)
        clear = self._brightness.total(sky_alt, sky_az, eph)
        cloud_light = self._cloud_brightness(sky_alt, sky_az, eph, extinction)
        surface = clear * transmission + (1 - transmission) * cloud_light
        colour = None
        if p.is_color and daylight(eph) > 0:
            # blue clear sky, white/grey clouds
            sky_rgb = sky_colour(sky_alt, sky_az, eph)
            with np.errstate(invalid="ignore", divide="ignore"):
                colour = (clear * transmission * sky_rgb + (1 - transmission) * cloud_light) / surface
            colour = np.nan_to_num(colour, nan=1.0)
        if self.horizon is not None:
            surface = np.where(obstructed, surface * self.horizon.brightness, surface)
        surface = np.where(geo.alt > 0, surface, surface * 0.02)  # ground below the horizon
        sky_rate = np.where(
            geo.in_lens, 10 ** (-0.4 * (nl_to_mag(surface) - p.zero_point)) * geo.solid_angle * geo.vignetting, 0.0
        )

        if exposure is None:
            exposure = p.exposure_time
            if self.auto_exposure:
                # like capture software: keep the median sky level at <= 15% of saturation
                level = float(np.median(sky_rate[visible_sky])) if np.any(visible_sky) else 0.0
                if level * exposure > 0.15 * p.saturation:
                    exposure = max(1e-6, 0.15 * p.saturation / level)

        sky_image = sky_rate * exposure
        image = sky_image.copy()

        # --- stars
        stars = self._star_positions(atime)
        star_alt, star_az, star_vmag = stars["alt"], stars["az"], stars["v_mag"]
        sx, sy = (np.asarray(v) for v in self._model.altaz_to_pix(star_alt, star_az))
        star_theta = np.pi / 2 - star_alt
        star_ext = combined_extinction(self.clouds, star_alt, star_az, seconds)
        star_vig = np.clip(1 - p.vignetting * (star_theta / (np.pi / 2)) ** 2, 0.05, 1)
        star_obstructed = (
            self.horizon.is_obstructed(star_alt, star_az) if self.horizon is not None else np.zeros(len(star_alt), bool)
        )
        flux = (
            10 ** (-0.4 * (star_vmag + k * airmass(star_theta) + star_ext - p.zero_point))
            * exposure
            * star_vig
            * ~star_obstructed
        )
        sigma = p.psf_fwhm / 2.3548 * (1 + p.psf_radial * (star_theta / (np.pi / 2)) ** 2)
        _add_gaussians(image, sx, sy, flux, sigma)

        # the moon itself
        if eph.moon_alt > 0:
            self._add_disc(image, eph.moon_alt, eph.moon_az, moon_magnitude(eph.moon_phase_angle), exposure, seconds)

        # the sun itself
        if eph.sun_alt > np.deg2rad(-1):
            self._add_disc(image, eph.sun_alt, eph.sun_az, SUN_MAGNITUDE, exposure, seconds)

        # --- sensor
        raw = self._sensor(self._raw_signal(image, sky_image, colour), exposure, time)

        # --- ground truth
        truth = self._truth(
            time, eph, exposure, seconds, stars, sx, sy, flux, sigma, star_ext, star_obstructed, sky_image
        )
        header = build_header(p, self.site, time, exposure)
        return SimulatedFrame(raw, header, truth, make_filename(p, self.site, time), time)

    def render_dark(self, time: datetime.datetime, exposure: float | None = None) -> SimulatedFrame:
        """A dark frame (lens covered): bias, dark current, hot pixels and noise only."""
        if time.tzinfo is None:
            time = time.replace(tzinfo=datetime.UTC)
        p = self.profile
        exposure = p.exposure_time if exposure is None else exposure
        raw = self._sensor(np.zeros((p.height, p.width)), exposure, time)
        header = build_header(p, self.site, time, exposure, image_type="Dark")
        empty = GroundTruth(np.zeros((len(TRUTH_ALT), len(TRUTH_AZ))), self.cloud_threshold, {}, {"dark": True})
        return SimulatedFrame(raw, header, empty, "dark_" + make_filename(p, self.site, time), time)

    @property
    def hot_pixels(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], FloatArray]:
        """Raw row, column and dark current (e-/s) of the simulated hot pixels."""
        return self._hot_y, self._hot_x, self._hot_rate

    def _sensor(self, signal: FloatArray, exposure: float, time: datetime.datetime) -> npt.NDArray[Any]:
        """Electrons per raw pixel -> stored ADU, with dark current, hot pixels, noise and saturation."""
        p = self.profile
        rng = np.random.default_rng([self.seed, int(time.timestamp() * 1000) % 2**63])
        dark = np.full(signal.shape, p.dark_current * exposure)
        dark[self._hot_y, self._hot_x] += self._hot_rate * exposure
        electrons = rng.poisson(np.clip(signal + dark, 0, 1e12)).astype(np.float64)
        electrons += rng.normal(0, p.read_noise, electrons.shape)
        electrons = np.clip(electrons, 0, p.full_well)
        adu = electrons / p.gain + p.bias
        if p.stretch_gamma is not None:
            adu = p.max_adu * (np.clip(adu, 0, p.max_adu) / p.max_adu) ** p.stretch_gamma
        adu = np.clip(np.rint(adu), 0, p.max_adu).astype(np.int64) << p.adu_shift
        return np.ascontiguousarray(adu.astype(p.storage_dtype))

    def _cloud_brightness(self, alt: FloatArray, az: FloatArray, eph: Ephemeris, extinction: FloatArray) -> FloatArray:
        """Surface brightness of a cloud: ground light, moonlight and sunlight it reflects. In daylight thin clouds
        are brighter than the sky around them and thick ones darker grey."""
        base = mag_to_nl(self.sky.zenith_mag) * self.sky.cloud_ground_light
        # an opaque cloud lit by the full moon (~15.6 mag/arcsec^2) is ~15x brighter than the moonlit clear sky
        moon = self._brightness.moon(alt, az, eph) * 30 * self.sky.cloud_moon_albedo
        thickness = np.clip(extinction / 4, 0, 1)
        sun = self._brightness.twilight(alt, az, eph) * (1.8 - 1.1 * thickness)
        return np.asarray(base + moon + sun)

    def _add_disc(
        self, image: FloatArray, alt: float, az: float, magnitude: float, exposure: float, seconds: float
    ) -> None:
        """The sun or moon: a blob with the body's apparent size and brightness, dimmed by clouds."""
        p = self.profile
        x, y = self._model.altaz_to_pix(alt, az)
        theta = np.array(np.pi / 2 - alt)
        plate = float(self._model.radius(theta + 1e-3) - self._model.radius(theta)) / 1e-3
        ext = float(combined_extinction(self.clouds, np.array([alt]), np.array([az]), seconds)[0])
        flux = 10 ** (-0.4 * (magnitude + self.sky.extinction * airmass(theta) + ext - p.zero_point)) * exposure
        sigma = max(p.psf_fwhm / 2.3548, MOON_RADIUS * plate)
        _add_gaussians(image, np.array([x]), np.array([y]), np.array([float(flux)]), np.array([sigma]))

    def _raw_signal(self, image: FloatArray, sky_image: FloatArray, colour: FloatArray | None) -> FloatArray:
        """Electrons per raw pixel: canonical image -> raw orientation, through the Bayer filters."""
        p, geo = self.profile, self.geometry()
        if colour is None or not p.is_color:
            return np.asarray(p.orientation.to_raw(image) * geo.channel_response)
        # stars, moon and sun are white, the sky and clouds have the colour computed for them
        point_sources = image - sky_image
        signal = np.zeros((p.height, p.width))
        for c in range(3):
            channel = p.orientation.to_raw(sky_image * colour[c] + point_sources)
            signal += np.where(geo.channel_index == c, channel, 0.0)
        return signal * geo.channel_response

    def _star_positions(self, atime: Time) -> dict[str, FloatArray]:
        cat = self._catalog
        frame = AltAz(obstime=atime, location=self._observer.location)
        altaz = SkyCoord(ra=cat["ra"], dec=cat["dec"], unit="deg").transform_to(frame)
        alt, az = altaz.alt.rad, altaz.az.rad

        # faint synthetic stars: a plain rotation by the sidereal time is plenty
        lst = atime.sidereal_time("apparent", longitude=self.site.longitude).rad
        lat = np.deg2rad(self.site.latitude)
        ha = lst - self._faint["ra"]
        dec = self._faint["dec"]
        f_alt = np.arcsin(np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(ha))
        f_az = np.mod(
            np.arctan2(-np.cos(dec) * np.sin(ha), np.sin(dec) * np.cos(lat) - np.cos(dec) * np.sin(lat) * np.cos(ha)),
            2 * np.pi,
        )

        n_cat = len(cat["sao"])
        is_cat = np.r_[np.ones(n_cat, bool), np.zeros(len(f_alt), bool)]
        result = {
            "sao": np.r_[cat["sao"], np.full(len(f_alt), -1)],
            "alt": np.r_[alt, f_alt],
            "az": np.r_[az, f_az],
            "v_mag": np.r_[cat["v_mag"], self._faint["v_mag"]],
            "catalog": is_cat,
        }
        above = result["alt"] > 0
        return {key: np.asarray(value)[above] for key, value in result.items()}

    def _truth(
        self,
        time: datetime.datetime,
        eph: Ephemeris,
        exposure: float,
        seconds: float,
        stars: dict[str, FloatArray],
        sx: FloatArray,
        sy: FloatArray,
        flux: FloatArray,
        sigma: FloatArray,
        star_ext: FloatArray,
        star_obstructed: npt.NDArray[np.bool_],
        sky_image: FloatArray,
    ) -> GroundTruth:
        p, geo = self.profile, self.geometry()
        grid_alt, grid_az = np.meshgrid(np.deg2rad(TRUTH_ALT), np.deg2rad(TRUTH_AZ), indexing="ij")
        extinction = combined_extinction(self.clouds, grid_alt, grid_az, seconds)

        h, w = sky_image.shape
        inside = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
        ix = np.clip(np.rint(np.nan_to_num(sx)).astype(int), 0, w - 1)
        iy = np.clip(np.rint(np.nan_to_num(sy)).astype(int), 0, h - 1)
        inside &= geo.in_lens[iy, ix]

        # SNR in the luminance the night pipeline sees (Bayer 2x2 superpixel -> mean response)
        response = float(np.mean(geo.channel_response[:2, :2]))
        background = sky_image[iy, ix] * response + p.dark_current * exposure
        n_eff = 4 * np.pi * sigma**2
        signal = flux * response
        # quantisation matters for 8-bit cameras with many electrons per ADU
        noise = np.sqrt(signal + n_eff * (background + p.read_noise**2 + p.gain**2 / 12))
        snr = np.where(noise > 0, signal / noise, 0.0)
        saturated = sky_image[iy, ix] * response > 0.8 * p.saturation
        visible = (snr >= VISIBLE_SNR) & inside & ~star_obstructed & ~saturated

        cat = stars["catalog"].astype(bool)
        star_table = {
            "sao": stars["sao"][cat].astype(np.int64),
            "alt": np.rad2deg(stars["alt"][cat]),
            "az": np.rad2deg(stars["az"][cat]),
            "v_mag": stars["v_mag"][cat],
            "x": sx[cat],
            "y": sy[cat],
            "flux": flux[cat],
            "extinction": star_ext[cat],
            "snr": snr[cat],
            "inside": inside[cat],
            "visible": visible[cat],
        }

        zenith = np.array([np.pi / 2]), np.array([0.0])
        metadata = {
            "time": time.astimezone(datetime.UTC).isoformat(),
            "exposure": exposure,
            "camera": dataclasses.asdict(self.profile),
            "world_model": self._model.to_config(),
            "orientation": dataclasses.asdict(p.orientation),
            "site": dataclasses.asdict(self.site),
            "sky": dataclasses.asdict(self.sky),
            "clouds": [dataclasses.asdict(c) for c in self.clouds],
            "seed": self.seed,
            "sun_alt": np.rad2deg(eph.sun_alt),
            "sun_az": np.rad2deg(eph.sun_az),
            "moon_alt": np.rad2deg(eph.moon_alt),
            "moon_az": np.rad2deg(eph.moon_az),
            "moon_phase_angle": eph.moon_phase_angle,
            "moon_illumination": eph.moon_illumination,
            "sky_zenith_mag": float(nl_to_mag(self._brightness.total(*zenith, eph))[0]),
            "cloud_seconds": seconds,
        }
        return GroundTruth(extinction, self.cloud_threshold, star_table, metadata)


def _add_gaussians(image: FloatArray, x: FloatArray, y: FloatArray, flux: FloatArray, sigma: FloatArray) -> None:
    """Add normalised circular Gaussians, vectorised in chunks."""
    h, w = image.shape
    ok = np.isfinite(x) & np.isfinite(y) & (flux > 0) & (x > -10) & (x < w + 10) & (y > -10) & (y < h + 10)
    x, y, flux, sigma = x[ok], y[ok], flux[ok], sigma[ok]
    if len(x) == 0:
        return
    # big objects (the moon) one by one, the rest in vectorised chunks
    big = sigma > 6
    for i in np.flatnonzero(big):
        half = int(np.ceil(4 * sigma[i]))
        x0, y0 = int(round(x[i])), int(round(y[i]))
        ys, xs = np.mgrid[max(0, y0 - half) : min(h, y0 + half + 1), max(0, x0 - half) : min(w, x0 + half + 1)]
        if ys.size:
            g = np.exp(-((xs - x[i]) ** 2 + (ys - y[i]) ** 2) / (2 * sigma[i] ** 2))
            image[ys, xs] += flux[i] * g / (2 * np.pi * sigma[i] ** 2)
    x, y, flux, sigma = x[~big], y[~big], flux[~big], sigma[~big]
    if len(x) == 0:
        return
    half = int(np.ceil(4 * np.max(sigma)))
    off = np.arange(-half, half + 1)
    for start in range(0, len(x), 20000):
        sl = slice(start, start + 20000)
        cx, cy, f, s = x[sl], y[sl], flux[sl], sigma[sl]
        px = np.rint(cx).astype(int)[:, None] + off[None, :]
        py = np.rint(cy).astype(int)[:, None] + off[None, :]
        gx = np.exp(-((px - cx[:, None]) ** 2) / (2 * s[:, None] ** 2))
        gy = np.exp(-((py - cy[:, None]) ** 2) / (2 * s[:, None] ** 2))
        gx /= gx.sum(axis=1, keepdims=True)
        gy /= gy.sum(axis=1, keepdims=True)
        stamp = f[:, None, None] * gy[:, :, None] * gx[:, None, :]
        rows = np.broadcast_to(py[:, :, None], stamp.shape)
        cols = np.broadcast_to(px[:, None, :], stamp.shape)
        m = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        np.add.at(image, (rows[m], cols[m]), stamp[m])


_CATALOG: dict[str, FloatArray] | None = None


def _load_catalog() -> dict[str, FloatArray]:
    global _CATALOG
    if _CATALOG is None:
        data = np.loadtxt(bundled_catalog_path(), skiprows=1, delimiter=";", comments=["#"])
        _CATALOG = {"sao": data[:, 1], "ra": data[:, 2], "dec": data[:, 3], "v_mag": data[:, 4]}
    return _CATALOG


def _faint_stars(limit: float, bright: float = 6.5) -> dict[str, FloatArray]:
    """Random stars between the catalogue limit and ``limit`` with a realistic count slope (~0.47 dex/mag)."""
    if limit <= bright:
        return {"ra": np.array([]), "dec": np.array([]), "v_mag": np.array([])}
    rng = np.random.default_rng(FAINT_STAR_SEED)
    slope = 0.47
    n = int(9000 * (10 ** (slope * (limit - bright)) - 1))
    u = rng.random(n)
    lo, hi = 10 ** (slope * bright), 10 ** (slope * limit)
    v_mag = np.log10(lo + u * (hi - lo)) / slope
    return {
        "ra": rng.uniform(0, 2 * np.pi, n),
        "dec": np.arcsin(rng.uniform(-1, 1, n)),
        "v_mag": v_mag,
    }
