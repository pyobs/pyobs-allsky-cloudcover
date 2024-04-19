from typing import List

import numpy as np
import numpy.typing as npt
from cloudmap_rs import Entry, gen_cloud_map, Average

from pyobs_cloudcover.pipeline.night.catalog.pixel_catalog import PixelCatalog


class CloudMapGenerator(object):
    def __init__(self, radius: float):
        self._radius = radius

    def __call__(self, catalog: PixelCatalog, matches: List[bool], height: int, width: int) -> npt.NDArray[np.float_]:
        std_entries = [Entry(*entry, True) for entry in zip(catalog.sao, catalog.px, catalog.py, catalog.v_mag)]
        av_std_vis_map = gen_cloud_map(std_entries, self._radius, width, height)
        std_vis_map = self._convert_average_to_value(av_std_vis_map)

        match_entries = [Entry(*entry) for entry in zip(catalog.sao, catalog.px, catalog.py, catalog.v_mag, matches)]
        av_vis_map = gen_cloud_map(match_entries, self._radius, width, height)
        vis_map = self._convert_average_to_value(av_vis_map)

        return std_vis_map - vis_map

    @staticmethod
    def _convert_average_to_value(average_map: List[List[Average]]) -> npt.NDArray[np.float_]:
        return np.array(
            [[x.value if x is not None else np.nan for x in row] for row in average_map]
        )
