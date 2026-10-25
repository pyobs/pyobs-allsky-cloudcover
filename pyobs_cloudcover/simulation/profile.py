from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pyobs_cloudcover.camera.orientation import Orientation
from pyobs_cloudcover.world_model.fisheye_model import FisheyeModel

BAYER_PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")


@dataclass(frozen=True)
class CameraProfile:
    """Everything the simulator needs to know about a camera + lens. Values are physical, not tuned.

    Attributes:
        name: Preset name.
        instrument: Value for the INSTRUME header keyword.
        width, height: Raw sensor array size in pixels.
        bit_depth: ADC bits (8, 12, 14, 16).
        adu_shift: Left shift applied when storing (ZWO stores 12-bit data as 16-bit multiples of 16).
        bayer: Bayer pattern of the raw array (row 0 first), or None for mono.
        lens: World-model parameters (``FisheyeModel`` kwargs) in canonical full-resolution pixels.
        orientation: Raw -> canonical transform.
        gain: Electrons per ADU.
        read_noise: Read noise in electrons.
        bias: Offset in ADU.
        dark_current: Electrons per second per pixel.
        full_well: Saturation in electrons.
        zero_point: Magnitude (V, top of atmosphere, zenith) giving 1 e-/s in total over the PSF, for the
            green (or mono) channel.
        channel_response: Relative response of R, G, B pixels to a star/night-sky spectrum.
        psf_fwhm: PSF FWHM in pixels at the zenith.
        psf_radial: Fractional FWHM increase at the horizon.
        vignetting: Fractional brightness loss at the horizon.
        hot_pixel_fraction: Fraction of pixels with a high dark current.
        hot_pixel_rate: Typical dark current of a hot pixel (e-/s).
        exposure_time: Default night exposure in seconds.
        day_exposure_time: Default day exposure in seconds.
        sensor_gain: Value for the GAIN header keyword.
        temperature: Sensor temperature for the header (deg C).
        header_style: Which capture software's header conventions to imitate (see ``headers.py``).
        filename_format: strftime pattern for file names (time in UTC unless ``filename_local_time``).
        filename_local_time: Use the site's local time in file names.
        stretch_gamma: If set, the output is non-linearly stretched (``x**gamma``) like cheap cameras do.
    """

    name: str
    instrument: str
    width: int
    height: int
    bit_depth: int
    lens: dict[str, Any]
    adu_shift: int = 0
    bayer: str | None = None
    orientation: Orientation = field(default_factory=Orientation)
    gain: float = 1.0
    read_noise: float = 2.0
    bias: float = 0.0
    dark_current: float = 0.05
    full_well: float = 20000.0
    zero_point: float = 8.0
    channel_response: tuple[float, float, float] = (0.6, 1.0, 0.5)
    psf_fwhm: float = 2.0
    psf_radial: float = 0.5
    vignetting: float = 0.3
    hot_pixel_fraction: float = 2e-4
    hot_pixel_rate: float = 50.0
    exposure_time: float = 30.0
    day_exposure_time: float = 0.0005
    sensor_gain: float = 100.0
    temperature: float = 20.0
    header_style: str = "indi"
    filename_format: str = "image_%Y%m%d_%H%M%S.fits"
    filename_local_time: bool = False
    stretch_gamma: float | None = None

    def __post_init__(self) -> None:
        if self.bayer is not None and self.bayer not in BAYER_PATTERNS:
            raise ValueError(f"Unknown Bayer pattern {self.bayer}")
        if self.bit_depth + self.adu_shift > 16:
            raise ValueError("bit_depth + adu_shift must fit into 16 bits")

    @property
    def is_color(self) -> bool:
        return self.bayer is not None

    @property
    def storage_dtype(self) -> type[np.unsignedinteger[Any]]:
        return np.uint8 if self.bit_depth + self.adu_shift <= 8 else np.uint16

    @property
    def max_adu(self) -> int:
        """Largest value the ADC can produce (before ``adu_shift``)."""
        return int(2**self.bit_depth - 1)

    @property
    def saturation(self) -> float:
        """Electrons at which a pixel saturates: full well or the end of the ADC range, whichever comes first."""
        return min(self.full_well, (self.max_adu - self.bias) * self.gain)

    @property
    def canonical_shape(self) -> tuple[int, int]:
        return self.orientation.canonical_shape(self.height, self.width)

    def world_model(self) -> FisheyeModel:
        return FisheyeModel(**self.lens)

    def scaled(self, factor: float) -> CameraProfile:
        """A smaller (or larger) version of the camera for fast tests: same field of view and sky.

        A scaled pixel behaves like a binned one: full well, gain and dark current grow with the pixel area,
        read noise with its square root, so star and sky statistics match a binned full-size image. Sizes stay
        even so Bayer 2x2 cells stay intact.
        """
        width = max(2, int(round(self.width * factor / 2)) * 2)
        height = max(2, int(round(self.height * factor / 2)) * 2)
        # use the exact ratio actually achieved for the lens
        exact = width / self.width
        lens = FisheyeModel(**self.lens).scaled(exact).to_config()
        lens.pop("class")
        return dataclasses.replace(
            self,
            name=f"{self.name}@{factor:g}",
            width=width,
            height=height,
            lens=lens,
            psf_fwhm=max(0.8, self.psf_fwhm * exact),
            full_well=self.full_well / exact**2,
            gain=self.gain / exact**2,
            dark_current=self.dark_current / exact**2,
            read_noise=self.read_noise / exact,
        )


PRESETS: dict[str, CameraProfile] = {
    # Raspberry Pi HQ camera (Sony IMX477) in its 2x2-binned 2028x1520 raw mode, 12-bit, stored unshifted.
    # ~1.55 mm M12 fisheye (equidistant); the image circle is slightly cut off at top and bottom.
    # Header as written by typical Pi all-sky scripts: no Bayer keyword, whole-second timestamps.
    "rpi-hq": CameraProfile(
        name="rpi-hq",
        instrument="IMX477",
        width=2028,
        height=1520,
        bit_depth=12,
        bayer="BGGR",
        lens={
            "c_x": 1013.5,
            "c_y": 759.5,
            "f": 480.0,
            "a0": np.deg2rad(-90.0),
            "projection": "equidistant",
            "parity": -1,
        },
        gain=1.0,
        read_noise=3.0,
        bias=256,
        dark_current=0.4,
        full_well=8000,
        zero_point=8.3,
        channel_response=(0.55, 1.0, 0.45),
        psf_fwhm=2.2,
        hot_pixel_fraction=5e-4,
        exposure_time=30.0,
        sensor_gain=16.0,
        temperature=45.0,
        header_style="allsky",
        filename_format="image-%Y%m%d%H%M%S.fits",
    ),
    # ZWO ASI224MC with a 1.25 mm fisheye; 12-bit ADC stored as 16-bit (multiples of 16); INDI headers.
    "zwo-asi224mc": CameraProfile(
        name="zwo-asi224mc",
        instrument="ZWO CCD ASI224MC",
        width=1304,
        height=976,
        bit_depth=12,
        adu_shift=4,
        bayer="RGGB",
        lens={"c_x": 651.5, "c_y": 487.5, "f": 330.0, "a0": np.deg2rad(90.0), "projection": "equisolid", "parity": 1},
        gain=0.8,
        read_noise=1.5,
        bias=20,
        dark_current=0.1,
        full_well=19000,
        zero_point=8.5,
        channel_response=(0.65, 1.0, 0.55),
        psf_fwhm=1.8,
        exposure_time=30.0,
        sensor_gain=250.0,
        temperature=25.0,
        header_style="indi",
        filename_format="ccd1_%Y%m%dT%H%M%S.fits",
    ),
    # ZWO ASI178MM (mono, 14-bit stored as 16-bit), equidistant 1.4 mm lens, camera mounted rotated by
    # 90 degrees. N.I.N.A.-style headers with DATE-LOC and millisecond DATE-OBS.
    "zwo-asi178mm": CameraProfile(
        name="zwo-asi178mm",
        instrument="ZWO ASI178MM",
        width=3096,
        height=2080,
        bit_depth=14,
        adu_shift=2,
        bayer=None,
        orientation=Orientation(rotate=90),
        lens={
            "c_x": 1039.5,
            "c_y": 1547.5,
            "f": 640.0,
            "a0": np.deg2rad(180.0),
            "projection": "equidistant",
            "parity": -1,
        },
        gain=0.9,
        read_noise=2.2,
        bias=40,
        dark_current=0.1,
        full_well=15000,
        zero_point=8.6,
        channel_response=(1.0, 1.0, 1.0),
        psf_fwhm=2.6,
        exposure_time=20.0,
        sensor_gain=200.0,
        temperature=18.0,
        header_style="nina",
        filename_format="%Y-%m-%d_%H-%M-%S_LIGHT.fits",
        filename_local_time=True,
    ),
    # A low-cost 8-bit colour USB camera with a cheap lens (strong barrel distortion, modelled as a
    # polynomial), mirrored image, gamma-stretched output and no timestamp in the header: the time is only
    # in the (local-time) file name.
    "budget-8bit": CameraProfile(
        name="budget-8bit",
        instrument="USB2.0 Camera",
        width=1280,
        height=960,
        bit_depth=8,
        bayer="GRBG",
        orientation=Orientation(flip_x=True),
        lens={
            "c_x": 643.0,
            "c_y": 476.0,
            "projection": "polynomial",
            "coefficients": [330.0, 0.0, -28.0],
            "a0": np.deg2rad(10.0),
            "parity": 1,
        },
        gain=12.0,
        read_noise=4.0,
        bias=4,
        dark_current=1.0,
        full_well=4500,
        zero_point=8.0,
        channel_response=(0.6, 1.0, 0.5),
        psf_fwhm=2.4,
        psf_radial=0.8,
        vignetting=0.45,
        hot_pixel_fraction=1e-3,
        exposure_time=10.0,
        sensor_gain=1.0,
        temperature=30.0,
        header_style="minimal",
        filename_format="allsky_%Y%m%d-%H%M%S.fits",
        filename_local_time=True,
        stretch_gamma=0.6,
    ),
}


def get_preset(name: str, scale: float = 1.0) -> CameraProfile:
    try:
        profile = PRESETS[name]
    except KeyError:
        raise KeyError(f"Unknown camera preset {name!r}; available: {', '.join(PRESETS)}") from None
    return profile.scaled(scale) if scale != 1.0 else profile
