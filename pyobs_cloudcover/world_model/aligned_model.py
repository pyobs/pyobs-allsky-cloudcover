from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.world_model.world_model import WorldModel

FloatArray = npt.NDArray[np.float64]


def small_rotation(angles: FloatArray | tuple[float, float, float]) -> FloatArray:
    """Rotation of (east, north, up) vectors by ``angles`` = (about east, about north, about up), in radians."""
    a, b, c = (float(v) for v in angles)
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
    ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
    rz = np.array([[cc, -sc, 0.0], [sc, cc, 0.0], [0.0, 0.0, 1.0]])
    return np.asarray(rz @ rx @ ry)


def altaz_to_enu(alt: FloatArray, az: FloatArray) -> FloatArray:
    return np.stack([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)])


def enu_to_altaz(enu: FloatArray) -> tuple[FloatArray, FloatArray]:
    alt = np.arcsin(np.clip(enu[2], -1, 1))
    az = np.mod(np.arctan2(enu[0], enu[1]), 2 * np.pi)
    return alt, az


class AlignedModel(WorldModel):
    """Another world model, corrected by a small rotation of the sky: the camera has moved a little since the
    model was fitted (a bump, thermal flexure). The correction is updated per frame by
    :class:`~pyobs_cloudcover.pipeline.night.aligner.PointingAligner`; zero means the base model as fitted.

    A sky position ``v`` (east, north, up) is looked up in the base model as ``Q v``.
    """

    def __init__(self, model: WorldModel, angles: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self.model = model
        self.set_angles(angles)

    @property
    def angles(self) -> tuple[float, float, float]:
        return self._angles

    def set_angles(self, angles: FloatArray | tuple[float, float, float]) -> None:
        self._angles = (float(angles[0]), float(angles[1]), float(angles[2]))
        self._q = small_rotation(self._angles)

    def altaz_to_pix(
        self, alt: FloatArray | float, az: FloatArray | float
    ) -> tuple[FloatArray, FloatArray] | tuple[float, float]:
        enu = altaz_to_enu(np.asarray(alt, dtype=np.float64), np.asarray(az, dtype=np.float64))
        a2, z2 = enu_to_altaz(np.tensordot(self._q, enu, axes=1))
        if np.ndim(a2) == 0:
            return self.model.altaz_to_pix(float(a2), float(z2))
        return self.model.altaz_to_pix(a2, z2)

    def pix_to_altaz(
        self, x: FloatArray | float, y: FloatArray | float
    ) -> tuple[FloatArray, FloatArray] | tuple[float, float]:
        alt, az = self.model.pix_to_altaz(x, y)
        enu = altaz_to_enu(np.asarray(alt, dtype=np.float64), np.asarray(az, dtype=np.float64))
        a2, z2 = enu_to_altaz(np.tensordot(self._q.T, enu, axes=1))
        if np.ndim(a2) == 0:
            return float(a2), float(z2)
        return a2, z2
