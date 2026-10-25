import logging

import numpy as np
from numpy import typing as npt

from pyobs_cloudcover.camera.bayer import debayer
from pyobs_cloudcover.camera.profile import Frame
from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.cloud_info_calculator import CoverageInfoCalculator
from pyobs_cloudcover.pipeline.day.altaz_map_generator import AltAzMapGenerator
from pyobs_cloudcover.pipeline.day.color_ratio_calculation import calc_color_ratio
from pyobs_cloudcover.pipeline.day.sun_masker import SunMasker
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator import CloudMapGenerator
from pyobs_cloudcover.pipeline.night.preprocessor.image_binner import ImageBinner
from pyobs_cloudcover.pipeline.pipeline import Pipeline
from pyobs_cloudcover.sky_query import SkyGrid, SkyPixelQuery

log = logging.getLogger(__name__)


class DayPipeline(Pipeline):
    """Cloud detection by colour: blue sky has a high blue/red + blue/green ratio, clouds are grey/white.

    ``mask`` (True = sky) is in full-resolution canonical pixels; the colour-ratio map is binned by
    ``bin_size`` before thresholding. Mono images can't be analysed by colour and give no result.
    """

    def __init__(
        self,
        mask: npt.NDArray[np.bool_] | None,
        alt_az_generator: AltAzMapGenerator,
        cloud_map_generator: CloudMapGenerator,
        sun_masker: SunMasker,
        coverage_info_calculator: CoverageInfoCalculator,
        bin_size: int = 1,
        default_bayer: str | None = None,
    ) -> None:
        self._mask = mask
        self._alt_az_generator = alt_az_generator
        self._cloud_map_generator = cloud_map_generator
        self._sun_masker = sun_masker
        self._coverage_info_calculator = coverage_info_calculator
        self._binner = ImageBinner(bin_size)
        self._default_bayer = default_bayer
        self._warned_mono = False

        self._grid: SkyGrid | None = None
        self._alt_az_mask: npt.NDArray[np.bool_] | None = None

    def _rgb(self, frame: Frame) -> npt.NDArray[np.float64] | None:
        if frame.is_rgb:
            return np.asarray(frame.data, dtype=np.float64)
        pattern = frame.bayer or self._default_bayer
        if pattern is None:
            if not self._warned_mono:
                self._warned_mono = True
                log.warning("The day pipeline needs colour images (Bayer or RGB); mono frames are skipped.")
            return None
        return debayer(frame.data, pattern).astype(np.float64)

    def __call__(self, frame: Frame) -> CloudCoverageInfo | None:
        rgb = self._rgb(frame)
        if rgb is None:
            return None
        lo, hi = float(np.min(rgb)), float(np.max(rgb))
        normed = (rgb - lo) / (hi - lo) if hi > lo else np.zeros_like(rgb)
        ratio = self._binner(calc_color_ratio(normed))
        cloud_map = self._cloud_map_generator(ratio)

        if self._grid is None or self._alt_az_mask is None:
            alt, az, on_sky = self._alt_az_generator(*ratio.shape)
            if self._mask is not None:
                on_sky &= self._binner(self._mask.astype(np.float64))[: ratio.shape[0], : ratio.shape[1]] >= 0.5
            self._alt_az_mask = on_sky
            self._grid = SkyGrid(alt[on_sky], az[on_sky])

        sky_query = self._sun_masker(SkyPixelQuery(self._grid, cloud_map[self._alt_az_mask]), frame.obs_time)
        ratio_query = self._sun_masker(SkyPixelQuery(self._grid, ratio[self._alt_az_mask]), frame.obs_time)

        return self._coverage_info_calculator(sky_query, frame.obs_time, ratio_query, "color_ratio", "day")
