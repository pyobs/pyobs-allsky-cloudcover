"""Ground truth stored next to each simulated image (``<image stem>.truth.npz``)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

TRUTH_SUFFIX = ".truth.npz"

# regular alt/az grid of the cloud/transmission maps, in degrees
TRUTH_ALT = np.arange(0.0, 90.5, 1.0)
TRUTH_AZ = np.arange(0.0, 360.0, 2.0)


@dataclass
class GroundTruth:
    """What the simulator put into an image.

    Attributes:
        extinction: Cloud extinction (mag) on the ``alt`` x ``az`` grid (degrees).
        cloud_threshold: Extinction (mag) above which a position counts as cloudy.
        stars: Catalogue stars above the horizon: ``sao, alt, az, v_mag, x, y`` (canonical full-resolution
            pixels), ``flux`` (e-), ``extinction`` (cloud, mag), ``snr`` and ``visible`` (SNR >= 5, inside the
            image and not obstructed).
        metadata: Time, site, camera, world model, sun/moon, sky brightness, seed, ...
    """

    extinction: FloatArray
    cloud_threshold: float
    stars: dict[str, npt.NDArray[Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    alt: FloatArray = field(default_factory=lambda: TRUTH_ALT.copy())
    az: FloatArray = field(default_factory=lambda: TRUTH_AZ.copy())

    @property
    def transmission(self) -> FloatArray:
        return np.asarray(10 ** (-0.4 * self.extinction))

    @property
    def cloudy(self) -> npt.NDArray[np.bool_]:
        return np.asarray(self.extinction >= self.cloud_threshold)

    def _index(self, alt: Any, az: Any) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """Nearest grid cell for positions in degrees."""
        ia = np.clip(np.rint(np.asarray(alt) - self.alt[0]).astype(np.intp), 0, len(self.alt) - 1)
        step = self.az[1] - self.az[0]
        iz = np.rint(np.mod(np.asarray(az), 360) / step).astype(np.intp) % len(self.az)
        return ia, iz

    def extinction_at(self, alt: Any, az: Any) -> FloatArray:
        ia, iz = self._index(alt, az)
        return np.asarray(self.extinction[ia, iz])

    def cloudy_at(self, alt: Any, az: Any) -> npt.NDArray[np.bool_]:
        return np.asarray(self.extinction_at(alt, az) >= self.cloud_threshold)

    def cloud_fraction(self, min_alt: float = 0.0, max_alt: float = 90.0) -> float:
        """Area-weighted cloudy fraction of the sky between two altitudes (degrees)."""
        sel = (self.alt >= min_alt) & (self.alt <= max_alt)
        weight = np.cos(np.deg2rad(self.alt[sel]))[:, None] * np.ones((1, len(self.az)))
        if np.max(self.alt[sel]) >= 90:
            # the zenith row is a single point, not a ring
            weight[self.alt[sel] >= 90] = np.cos(np.deg2rad(89.75)) / 2
        return float(np.sum(self.cloudy[sel] * weight) / np.sum(weight))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        arrays = {f"star_{k}": v for k, v in self.stars.items()}
        np.savez_compressed(
            path,
            extinction=self.extinction.astype(np.float32),
            alt=self.alt,
            az=self.az,
            cloud_threshold=np.array(self.cloud_threshold),
            metadata=np.array(json.dumps(self.metadata, default=_json_default)),
            **arrays,  # type: ignore[arg-type]
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> GroundTruth:
        with np.load(path, allow_pickle=False) as data:
            stars = {k[5:]: data[k] for k in data.files if k.startswith("star_")}
            return cls(
                extinction=data["extinction"].astype(np.float64),
                cloud_threshold=float(data["cloud_threshold"]),
                stars=stars,
                metadata=json.loads(str(data["metadata"])),
                alt=data["alt"],
                az=data["az"],
            )

    @staticmethod
    def path_for(image_path: str | Path) -> Path:
        image_path = Path(image_path)
        name = image_path.name
        for ext in (".fits.gz", ".fits", ".fit", ".fts"):
            if name.endswith(ext):
                name = name[: -len(ext)]
                break
        return image_path.with_name(name + TRUTH_SUFFIX)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Can't serialise {type(obj)}")
