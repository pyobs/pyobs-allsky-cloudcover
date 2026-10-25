"""Site, atmosphere and sky-brightness models for the simulator.

Sky brightness follows Krisciunas & Schaefer (1991, PASP 103, 1033): a dark-sky term growing with airmass,
moonlight scattered by the atmosphere, plus simple empirical twilight and light-pollution terms. All
brightnesses are V-band surface brightnesses, handled in nanoLamberts (nL) internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astroplan import Observer

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Site:
    latitude: float = -32.38  # deg, default: Sutherland
    longitude: float = 20.81  # deg east
    elevation: float = 1760.0  # m
    timezone: str = "Africa/Johannesburg"
    name: str = "Sutherland"

    def observer(self) -> Observer:
        return Observer(
            latitude=self.latitude * u.deg,
            longitude=self.longitude * u.deg,
            elevation=self.elevation * u.m,
            timezone=self.timezone,
        )


@dataclass(frozen=True)
class Horizon:
    """Local horizon (trees, buildings): altitude in degrees as a function of azimuth (degrees), linearly
    interpolated and wrapped. ``brightness`` is the obstruction's surface brightness relative to the sky at
    the same position (0 = black silhouette, ~0.3 = lit trees in daylight)."""

    azimuth: tuple[float, ...] = (0.0,)
    altitude: tuple[float, ...] = (0.0,)
    brightness: float = 0.05

    def altitude_at(self, az: FloatArray) -> FloatArray:
        azs = np.asarray(self.azimuth, dtype=np.float64)
        alts = np.asarray(self.altitude, dtype=np.float64)
        order = np.argsort(azs)
        return np.asarray(np.deg2rad(np.interp(np.rad2deg(az) % 360, azs[order], alts[order], period=360)))

    def is_obstructed(self, alt: FloatArray, az: FloatArray) -> npt.NDArray[np.bool_]:
        return np.asarray(alt < self.altitude_at(az))

    @classmethod
    def trees_and_building(cls, seed: int = 1) -> Horizon:
        """A plausible obstructed horizon: a tree line with gaps and a building in the north-east."""
        rng = np.random.default_rng(seed)
        az = np.arange(0, 360, 2.0)
        alt = np.clip(6 + 6 * np.sin(np.deg2rad(az * 3 + 40)) + rng.normal(0, 2.0, len(az)), 0, None)
        alt[(az > 30) & (az < 70)] = 22.0  # building
        alt[(az > 200) & (az < 235)] = 0.0  # gap
        return cls(tuple(az), tuple(alt))


@dataclass(frozen=True)
class SkyConditions:
    """Atmosphere and sky brightness.

    Attributes:
        zenith_mag: Dark-sky V brightness at the zenith (mag/arcsec^2), e.g. 21.8 pristine, 20 suburban.
        extinction: V-band extinction coefficient (mag per airmass).
        light_pollution: Extra zenith brightness from light pollution, as a fraction of the dark sky.
        light_dome_az: Azimuth (deg) of the brightest light dome near the horizon, or None.
        light_dome_strength: Light-dome brightness at the horizon as a multiple of the dark zenith sky.
        cloud_ground_light: Brightness of an opaque cloud lit only from the ground, relative to the dark
            zenith sky (light-polluted sites have bright clouds, dark sites black clouds).
        cloud_moon_albedo: Scales how strongly clouds reflect moonlight.
    """

    zenith_mag: float = 21.3
    extinction: float = 0.20
    light_pollution: float = 0.1
    light_dome_az: float | None = None
    light_dome_strength: float = 0.0
    cloud_ground_light: float = 0.3
    cloud_moon_albedo: float = 0.5


def airmass(zenith_angle: FloatArray) -> FloatArray:
    """Krisciunas & Schaefer eq. 3 (valid down to the horizon)."""
    return np.asarray((1 - 0.96 * np.sin(zenith_angle) ** 2) ** -0.5)


def mag_to_nl(mag: FloatArray | float) -> FloatArray:
    """V mag/arcsec^2 -> nanoLamberts (K&S eq. 1)."""
    return np.asarray(34.08 * np.exp(20.7233 - 0.92104 * np.asarray(mag)))


def nl_to_mag(nl: FloatArray | float) -> FloatArray:
    with np.errstate(divide="ignore"):
        return np.asarray((20.7233 - np.log(np.asarray(nl) / 34.08)) / 0.92104)


def angular_distance(alt1: Any, az1: Any, alt2: Any, az2: Any) -> FloatArray:
    cos_d = np.sin(alt1) * np.sin(alt2) + np.cos(alt1) * np.cos(alt2) * np.cos(np.asarray(az1) - az2)
    return np.asarray(np.arccos(np.clip(cos_d, -1, 1)))


def moon_magnitude(phase_angle_deg: float) -> float:
    """Apparent V magnitude of the moon (K&S eq. 9 as a magnitude)."""
    a = abs(phase_angle_deg)
    return -12.73 + 0.026 * a + 4e-9 * a**4


@dataclass
class Ephemeris:
    """Sun and moon for one moment (angles in rad)."""

    sun_alt: float
    sun_az: float
    moon_alt: float
    moon_az: float
    moon_phase_angle: float  # deg, 0 = full
    moon_illumination: float  # 0..1

    @classmethod
    def compute(cls, observer: Observer, time: Any) -> Ephemeris:
        sun = observer.sun_altaz(time)
        moon = observer.moon_altaz(time)
        phase = float(observer.moon_phase(time).to_value(u.deg))
        return cls(
            sun_alt=float(sun.alt.rad),
            sun_az=float(sun.az.rad),
            moon_alt=float(moon.alt.rad),
            moon_az=float(moon.az.rad),
            moon_phase_angle=phase,
            moon_illumination=float(observer.moon_illumination(time)),
        )


_TWILIGHT_SUN_ALT = (-18.0, -15.0, -12.0, -9.0, -6.0, -3.0, 0.0, 10.0, 30.0, 60.0)
_TWILIGHT_ZENITH_MAG = (23.5, 20.6, 18.6, 16.3, 13.6, 11.0, 8.5, 5.5, 4.0, 3.5)


@dataclass
class SkyBrightness:
    """Night/twilight sky surface brightness at arbitrary positions (V, nL)."""

    conditions: SkyConditions = field(default_factory=SkyConditions)

    def dark(self, alt: FloatArray, az: FloatArray) -> FloatArray:
        z = np.pi / 2 - np.clip(alt, 0, None)
        x = airmass(z)
        k = self.conditions.extinction
        b = mag_to_nl(self.conditions.zenith_mag) * (1 + self.conditions.light_pollution)
        dark = b * x * 10 ** (-0.4 * k * (x - 1))
        if self.conditions.light_dome_az is not None and self.conditions.light_dome_strength > 0:
            dist = angular_distance(0.0, np.deg2rad(self.conditions.light_dome_az), alt, az)
            dark = dark + mag_to_nl(self.conditions.zenith_mag) * self.conditions.light_dome_strength * np.exp(
                -np.clip(alt, 0, None) / np.deg2rad(12)
            ) * np.exp(-dist / np.deg2rad(40))
        return np.asarray(dark)

    def moon(self, alt: FloatArray, az: FloatArray, eph: Ephemeris) -> FloatArray:
        """Scattered moonlight (K&S eqs. 15-21)."""
        if eph.moon_alt <= 0:
            return np.zeros(np.shape(alt))
        k = self.conditions.extinction
        rho = np.rad2deg(angular_distance(alt, az, eph.moon_alt, eph.moon_az))
        a = abs(eph.moon_phase_angle)
        i_star = 10 ** (-0.4 * (3.84 + 0.026 * a + 4e-9 * a**4))
        f_rho = 10**5.36 * (1.06 + np.cos(np.deg2rad(rho)) ** 2) + 10 ** (6.15 - rho / 40)
        x = airmass(np.pi / 2 - np.clip(alt, 0, None))
        x_moon = airmass(np.array(np.pi / 2 - eph.moon_alt))
        return np.asarray(f_rho * i_star * 10 ** (-0.4 * k * x_moon) * (1 - 10 ** (-0.4 * k * x)))

    def twilight(self, alt: FloatArray, az: FloatArray, eph: Ephemeris) -> FloatArray:
        """Sunlit sky: zenith brightness interpolated from typical V values (dark at -18 deg, ~13.5 at the end
        of civil twilight, ~4 mag/arcsec^2 in daylight), brighter towards the sun and the horizon.
        Approximate; good enough to exercise the pipelines."""
        sun_alt = np.rad2deg(eph.sun_alt)
        if sun_alt < -18:
            return np.zeros(np.shape(alt))
        zenith_mag = float(np.interp(sun_alt, _TWILIGHT_SUN_ALT, _TWILIGHT_ZENITH_MAG))
        dist = angular_distance(alt, az, eph.sun_alt, eph.sun_az)
        shape = (1 + 2.5 * np.exp(-dist / np.deg2rad(35))) * airmass(np.pi / 2 - np.clip(alt, 0, None)) ** 0.5
        return np.asarray(mag_to_nl(zenith_mag) * shape)

    def total(self, alt: FloatArray, az: FloatArray, eph: Ephemeris) -> FloatArray:
        return self.dark(alt, az) + self.moon(alt, az, eph) + self.twilight(alt, az, eph)


# Raw-sensor-independent colour of the sunlit clear sky relative to green (Rayleigh: blue at the zenith and away
# from the sun, white towards the horizon and around the sun)
_BLUE_SKY = np.array([0.55, 1.0, 1.55])


def daylight(eph: Ephemeris) -> float:
    """0 at night, 1 in daylight, ramping up during twilight (sun -10 .. -2 deg)."""
    return float(np.clip((np.rad2deg(eph.sun_alt) + 10) / 8, 0, 1))


def sky_colour(alt: FloatArray, az: FloatArray, eph: Ephemeris) -> FloatArray:
    """(3, ...) R, G, B weights of the clear sky relative to its V/green brightness."""
    d = daylight(eph)
    shape = np.shape(alt)
    if d == 0:
        return np.ones((3, *shape))
    rho = angular_distance(alt, az, eph.sun_alt, eph.sun_az)
    white = np.clip(np.exp(-np.clip(alt, 0, None) / np.deg2rad(12)) + np.exp(-rho / np.deg2rad(12)), 0, 1)
    colour = _BLUE_SKY[:, None] * (1 - white.ravel()) + white.ravel()
    return np.asarray((1 + d * (colour - 1)).reshape(3, *shape))
