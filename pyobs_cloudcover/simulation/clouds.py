"""Procedural cloud layers.

A layer is a periodic fractal (power-law Gaussian random) field on a horizontal plane at ``height`` above the
camera. A sky position at zenith angle z and azimuth az sees the layer at a horizontal distance
``height * tan(z)``, so clouds shrink towards the horizon as in a real all-sky image. The field is thresholded
so that ``cover`` of the plane is cloudy, and turned into an extinction in magnitudes that ramps up from 0
at the cloud edge to ``optical_depth`` inside. The layer drifts with ``wind`` between frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.ndimage import map_coordinates

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class CloudLayer:
    """
    Attributes:
        cover: Fraction of the layer that is cloudy (0..1). The cloudy fraction of the *visible sky* differs
            somewhat because of perspective; the ground truth reports the actual value.
        optical_depth: Extinction (mag) inside the clouds: ~0.3 thin cirrus, ~1 thick cirrus, >=3 opaque.
        height: Layer height in km.
        size: Size of the periodic layer in km.
        scale: Typical cloud size in km (sets where the power spectrum turns over).
        softness: Width of the edge ramp in units of the field's standard deviation.
        wind: Drift (east, north) in m/s.
        seed: Random seed of the cloud pattern.
        resolution: Grid points per side of the layer.
    """

    cover: float = 0.5
    optical_depth: float = 3.0
    height: float = 3.0
    size: float = 80.0
    scale: float = 6.0
    softness: float = 0.3
    wind: tuple[float, float] = (8.0, 3.0)
    seed: int = 0
    resolution: int = 512

    def __post_init__(self) -> None:
        if not 0 <= self.cover <= 1:
            raise ValueError("cover must be within 0..1")

    def _field(self) -> FloatArray:
        n = self.resolution
        rng = np.random.default_rng(self.seed)
        k = np.fft.fftfreq(n, d=self.size / n)  # cycles per km
        kx, ky = np.meshgrid(k, k)
        k2 = kx**2 + ky**2
        k0 = 1.0 / self.scale
        # Kolmogorov-like power law above k0, flat below (no giant structures)
        power = (k2 + k0**2) ** (-11 / 6)
        power[0, 0] = 0
        noise = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        field = np.fft.ifft2(noise * np.sqrt(power)).real
        return np.asarray((field - field.mean()) / field.std())

    def extinction(self, alt: FloatArray, az: FloatArray, seconds: float = 0.0) -> FloatArray:
        """Cloud extinction in magnitudes at the given positions (rad), ``seconds`` after the reference time."""
        field = _field_cache(self)
        alt = np.asarray(alt, dtype=np.float64)
        az = np.asarray(az, dtype=np.float64)
        if self.cover <= 0:
            return np.zeros(alt.shape)
        threshold = np.quantile(field, 1 - self.cover) if self.cover < 1 else -np.inf

        z = np.clip(np.pi / 2 - alt, 0, np.deg2rad(88))
        rho = self.height * np.tan(z)
        east = rho * np.sin(az) - self.wind[0] * seconds / 1000.0
        north = rho * np.cos(az) - self.wind[1] * seconds / 1000.0
        n = self.resolution
        coords = np.stack([north.ravel() / self.size * n, east.ravel() / self.size * n])
        values = map_coordinates(field, coords, order=1, mode="grid-wrap").reshape(alt.shape)

        ramp = np.clip((values - threshold) / self.softness, 0, 1)
        # smoothstep keeps edges soft but the interior flat
        ramp = ramp * ramp * (3 - 2 * ramp)
        return np.asarray(self.optical_depth * ramp)


_FIELDS: dict[tuple[int, int, float, float], FloatArray] = {}


def _field_cache(layer: CloudLayer) -> FloatArray:
    # the field only depends on these; cover/depth/wind changes reuse it
    key = (layer.seed, layer.resolution, layer.size, layer.scale)
    if key not in _FIELDS:
        if len(_FIELDS) > 8:
            _FIELDS.clear()
        _FIELDS[key] = layer._field()
    return _FIELDS[key]


def combined_extinction(layers: list[CloudLayer], alt: FloatArray, az: FloatArray, seconds: float) -> FloatArray:
    total = np.zeros(np.shape(alt))
    for layer in layers:
        total = total + layer.extinction(alt, az, seconds)
    return total
