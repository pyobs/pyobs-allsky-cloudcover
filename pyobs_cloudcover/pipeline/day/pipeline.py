import datetime

import numpy as np
from numpy import typing as npt

from cloudmap_rs import SkyPixelQuery

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.cloud_info_calculator import CoverageInfoCalculator
from pyobs_cloudcover.pipeline.day.color_ratio_calculation import calc_color_ratio
from pyobs_cloudcover.pipeline.day.debayer_image import debayer_image
from pyobs_cloudcover.pipeline.day.sun_masker import SunMasker
from pyobs_cloudcover.pipeline.night.altaz_grid_generator.altaz_map_generator import AltAzMapGenerator
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator import CloudMapGenerator
from pyobs_cloudcover.pipeline.pipeline import Pipeline


class DayPipeline(Pipeline):
    def __init__(self, mask: npt.NDArray[np.bool_], alt_az_generator: AltAzMapGenerator, cloud_map_generator: CloudMapGenerator, sun_masker: SunMasker, coverage_info_calculator: CoverageInfoCalculator) -> None:
        self._mask = mask
        self._alt_az_generator = alt_az_generator
        self._cloud_map_generator = cloud_map_generator
        self._sun_masker = sun_masker
        self._coverage_info_calculator = coverage_info_calculator

    def __call__(self, image: npt.NDArray[np.float_], obs_time: datetime.datetime) -> CloudCoverageInfo:
        debayered_image = debayer_image(image)
        color_ratio_image = calc_color_ratio(debayered_image)
        cloud_map = self._cloud_map_generator(color_ratio_image.flatten())

        alt_az_points = np.array(self._alt_az_generator(*color_ratio_image.shape)).flatten()

        sky_query = SkyPixelQuery(cloud_map[self._mask], alt_az_points[self._mask])
        sky_query = self._sun_masker(sky_query, obs_time)

        return self._coverage_info_calculator(sky_query, obs_time)
