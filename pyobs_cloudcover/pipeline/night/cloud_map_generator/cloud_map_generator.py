from typing import List, Optional

import numpy as np
import numpy.typing as npt
from cloudmap_rs import Entry, Average, MagnitudeMapGenerator, AltAzCoord

from pyobs_cloudcover.pipeline.night.catalog.pixel_catalog import PixelCatalog


class CloudMapGenerator(object):
    def __init__(self, radius: float):
        self._radius = radius

    def __call__(self, catalog: PixelCatalog, matches: List[bool], alt_az_image_list: List[List[Optional[AltAzCoord]]]) -> npt.NDArray[np.float_]:
        star_coords = [AltAzCoord(np.deg2rad(alt), np.deg2rad(az)) for alt, az in zip(catalog.alt, catalog.az)]

        map_generator = MagnitudeMapGenerator(star_coords, alt_az_image_list, np.deg2rad(self._radius))

        match_entries = [Entry(v_mag, found) for v_mag, found in zip(catalog.v_mag, matches)]
        av_vis_map = map_generator.gen_cloud_map(match_entries)
        vis_map = self._convert_average_to_value(av_vis_map)

        return vis_map

    @staticmethod
    def _convert_average_to_value(average_map: List[List[Average]]) -> npt.NDArray[np.float_]:
        return np.array(
            [[x.value if x is not None else np.nan for x in row] for row in average_map]
        )
