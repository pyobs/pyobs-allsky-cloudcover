from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.world_model.world_model import WorldModel

FloatArray = npt.NDArray[np.float64]


class PointedModel(WorldModel):
    """A camera whose optical axis may point anywhere, with a radially symmetric lens about that axis.

    Meant for cameras that are not zenith-centred fisheyes: wide-angle (near-rectilinear) lenses such as the
    Raspberry Pi Camera Module 3 Wide, or any lens on a tilted camera. :class:`FisheyeModel` assumes the zenith is
    the centre of symmetry; this model doesn't.

    Orientation: start from a camera looking straight up, whose image +x axis points to azimuth ``rotation + 90 deg``
    and +y axis (image down) to azimuth ``rotation`` (parity 1); then tilt it by ``tilt_x`` about the image x axis
    and ``tilt_y`` about the image y axis (radians). A sky position at angle ``phi`` from the optical axis lands at
    radius

        r(phi) = f * (phi + k1 * phi**3 + k2 * phi**5)

    from the principal point ``(c_x, c_y)`` (Kannala-Brandt; a rectilinear lens is r = f tan(phi), i.e. about
    k1 = 1/3, k2 = 2/15; an equidistant fisheye is k1 = k2 = 0). ``parity=-1`` mirrors the image (y flipped).

    Args:
        c_x, c_y: Principal point (canonical full-resolution pixels).
        f: Focal length in pixels.
        rotation: Azimuth (rad) that image +y (down) points to before tilting.
        tilt_x, tilt_y: Tilt of the optical axis from the zenith (rad), about the image x and y axes.
        k1, k2: Radial distortion coefficients.
        parity: +1 or -1.
        max_field_angle: Largest angle from the optical axis (deg) the lens maps.
    """

    def __init__(
        self,
        c_x: float,
        c_y: float,
        f: float,
        rotation: float = 0.0,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
        k1: float = 0.0,
        k2: float = 0.0,
        parity: int = 1,
        max_field_angle: float = 80.0,
    ) -> None:
        if parity not in (1, -1):
            raise ValueError("parity must be 1 or -1")
        if f <= 0:
            raise ValueError("f must be positive")
        self.c_x, self.c_y, self.f = float(c_x), float(c_y), float(f)
        self.rotation, self.tilt_x, self.tilt_y = float(rotation), float(tilt_x), float(tilt_y)
        self.k1, self.k2 = float(k1), float(k2)
        self.parity = parity
        self.max_field_angle = float(max_field_angle)

        self._phi_max = np.deg2rad(self.max_field_angle)
        self._phi_table = np.linspace(0, self._phi_max, 4001)
        self._r_table = self.radius(self._phi_table)
        if np.any(np.diff(self._r_table) <= 0):
            raise ValueError("Lens radius must increase with field angle up to max_field_angle (check k1, k2)")
        self._matrix = self._rotation_matrix()

    # ------------------------------------------------------------------ geometry

    def _rotation_matrix(self) -> FloatArray:
        """Rows: the camera's x, y, z (optical axis) axes in (east, north, up) coordinates."""
        c, s = np.cos(self.rotation), np.sin(self.rotation)
        # looking up: x -> azimuth rotation + 90 deg, y -> azimuth rotation, z -> zenith
        base = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        cx, sx = np.cos(self.tilt_x), np.sin(self.tilt_x)
        cy, sy = np.cos(self.tilt_y), np.sin(self.tilt_y)
        rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        return np.asarray(rot_x @ rot_y @ base)

    @property
    def axis_altaz(self) -> tuple[float, float]:
        """Altitude and azimuth (rad) of the optical axis."""
        e, n, u = self._matrix[2]
        return float(np.arcsin(np.clip(u, -1, 1))), float(np.mod(np.arctan2(e, n), 2 * np.pi))

    def radius(self, phi: FloatArray) -> FloatArray:
        return np.asarray(self.f * (phi + self.k1 * phi**3 + self.k2 * phi**5))

    def field_angle(self, r: FloatArray) -> FloatArray:
        with np.errstate(invalid="ignore"):
            return np.asarray(np.interp(r, self._r_table, self._phi_table, right=np.nan))

    # ------------------------------------------------------------------ WorldModel

    def altaz_to_pix(
        self, alt: FloatArray | float, az: FloatArray | float
    ) -> tuple[FloatArray, FloatArray] | tuple[float, float]:
        alt_a, az_a = np.asarray(alt, dtype=np.float64), np.asarray(az, dtype=np.float64)
        enu = np.stack([np.cos(alt_a) * np.sin(az_a), np.cos(alt_a) * np.cos(az_a), np.sin(alt_a)])
        p = np.tensordot(self._matrix, enu, axes=1)
        phi = np.arccos(np.clip(p[2], -1, 1))
        psi = np.arctan2(p[1], p[0])
        r = np.where(phi <= self._phi_max, self.radius(phi), np.nan)
        x = self.c_x + r * np.cos(psi)
        y = self.c_y + self.parity * r * np.sin(psi)
        if np.ndim(x) == 0:
            return float(x), float(y)
        return x, y

    def pix_to_altaz(
        self, x: FloatArray | float, y: FloatArray | float
    ) -> tuple[FloatArray, FloatArray] | tuple[float, float]:
        dx = np.asarray(x, dtype=np.float64) - self.c_x
        dy = self.parity * (np.asarray(y, dtype=np.float64) - self.c_y)
        phi = self.field_angle(np.hypot(dx, dy))
        psi = np.arctan2(dy, dx)
        cam = np.stack([np.sin(phi) * np.cos(psi), np.sin(phi) * np.sin(psi), np.cos(phi)])
        e, n, u = np.tensordot(self._matrix.T, cam, axes=1)
        alt = np.arcsin(np.clip(u, -1, 1))
        az = np.mod(np.arctan2(e, n), 2 * np.pi)
        if np.ndim(alt) == 0:
            return float(alt), float(az)
        return alt, az

    # ------------------------------------------------------------------ helpers

    def scaled(self, factor: float) -> PointedModel:
        """The same camera on an image resampled by ``factor`` (see :meth:`FisheyeModel.scaled`)."""
        offset = 0.5 * (1 - factor)
        return PointedModel(
            c_x=self.c_x * factor - offset,
            c_y=self.c_y * factor - offset,
            f=self.f * factor,
            rotation=self.rotation,
            tilt_x=self.tilt_x,
            tilt_y=self.tilt_y,
            k1=self.k1,
            k2=self.k2,
            parity=self.parity,
            max_field_angle=self.max_field_angle,
        )

    def params(self) -> dict[str, float | int]:
        return {
            "c_x": self.c_x,
            "c_y": self.c_y,
            "f": self.f,
            "rotation": self.rotation,
            "tilt_x": self.tilt_x,
            "tilt_y": self.tilt_y,
            "k1": self.k1,
            "k2": self.k2,
            "parity": self.parity,
            "max_field_angle": self.max_field_angle,
        }

    def to_config(self) -> dict[str, object]:
        return {"class": "pyobs_cloudcover.world_model.PointedModel", **self.params()}
