import datetime
from copy import copy
from itertools import compress
from typing import List, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy import typing as npt

from cloudmap_rs import SkyPixelQuery

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.cloud_info_calculator import CoverageInfoCalculator
from pyobs_cloudcover.cloud_plot.cloud_plotter import plot_clouds, render_plot
from pyobs_cloudcover.pipeline.day.color_ratio_calculation import calc_color_ratio
from pyobs_cloudcover.pipeline.day.debayer_image import debayer_image
from pyobs_cloudcover.pipeline.day.sun_masker import SunMasker
from pyobs_cloudcover.pipeline.day.altaz_map_generator import AltAzMapGenerator
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator import CloudMapGenerator
from pyobs_cloudcover.pipeline.night.preprocessor.image_binner import ImageBinner
from pyobs_cloudcover.pipeline.pipeline import Pipeline


class DayPipeline(Pipeline):
    def __init__(self, mask: npt.NDArray[np.bool_], alt_az_generator: AltAzMapGenerator, cloud_map_generator: CloudMapGenerator, sun_masker: SunMasker, coverage_info_calculator: CoverageInfoCalculator) -> None:
        self._mask = mask
        self._alt_az_generator = alt_az_generator
        self._cloud_map_generator = cloud_map_generator
        self._sun_masker = sun_masker
        self._coverage_info_calculator = coverage_info_calculator

        self._sky_query: Optional[SkyPixelQuery] = None
        self._alt_az_mask: Optional[npt.NDArray[np.bool_]] = None

        self._image_binner = ImageBinner(2)

    def __call__(self, image: npt.NDArray[np.float_], obs_time: datetime.datetime) -> CloudCoverageInfo:
        debayered_image = debayer_image(image)
        color_ratio_image = calc_color_ratio(debayered_image)#.flatten()
        cloud_map = self._image_binner(self._cloud_map_generator(color_ratio_image))


        if self._sky_query is None or self._alt_az_mask is None:
            alt_az_points = self.flatten(self._alt_az_generator(*cloud_map.shape))
            self._alt_az_mask = self._mask.flatten().astype(np.bool_) & np.array([x is not None for x in alt_az_points]).astype(np.bool_)
            flat_cloud_map =  self._to_python_type(cloud_map)
            self._sky_query = SkyPixelQuery(self.filter(alt_az_points, self._alt_az_mask), self.filter(flat_cloud_map, self._alt_az_mask)) # type: ignore
        else:
            self._sky_query.set_pixels(cloud_map[self._alt_az_mask])

        sky_query = self._sun_masker(self._sky_query, obs_time)

        cloud_plot = cloud_map.astype(np.float_).flatten()
        cloud_plot[~self._alt_az_mask] = np.nan
        plt.imshow(cloud_plot.reshape(cloud_map.shape))

        return self._coverage_info_calculator(sky_query, render_plot(),obs_time)


    def _to_python_type(self, cloud_map: npt.NDArray[np.float_]) -> List[bool]:
        flat_map = self.flatten(list(cloud_map > 0.5))
        return list(map(bool, flat_map))

    @staticmethod
    def flatten(map_2d: List[List[Any]]) -> List[Any]:
        return [entry for row in map_2d for entry in row]

    @staticmethod
    def filter(map_1d: List[Any], predicate: List[bool]) -> List[Any]:
        return list(compress(map_1d, predicate))
