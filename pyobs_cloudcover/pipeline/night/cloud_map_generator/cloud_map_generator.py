from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from cloudmap_rs import Entry, Average, MagnitudeMapGenerator, AltAzCoord

from pyobs_cloudcover.pipeline.night.catalog.pixel_catalog import PixelCatalog


class CloudMapGenerator(object):
    def __init__(self, radius: float, integration_length: int = 1):
        self._radius = radius
        self._integration_frame: deque[Tuple[PixelCatalog, List[bool]]] = deque([], integration_length)

    def __call__(self, catalog: PixelCatalog, matches: List[bool], alt_az_image_list: List[List[Optional[AltAzCoord]]]) -> npt.NDArray[np.float_]:
        self._update_integration_frame(catalog, matches)
        integrated_catalog, integrated_matches = self._get_integrated_frame()

        star_coords = [AltAzCoord(np.deg2rad(alt), np.deg2rad(az)) for alt, az in zip(integrated_catalog.alt, integrated_catalog.az)]

        map_generator = MagnitudeMapGenerator(star_coords, alt_az_image_list, np.deg2rad(self._radius))

        match_entries = [Entry(v_mag, found) for v_mag, found in zip(integrated_catalog.v_mag, integrated_matches)]
        av_vis_map = map_generator.gen_cloud_map(match_entries)
        vis_map = self._convert_average_to_value(av_vis_map)

        return vis_map

    def _update_integration_frame(self, catalog: PixelCatalog, matches: List[bool]) -> None:
        self._integration_frame.append((catalog, matches))

    def _get_integrated_frame(self) -> Tuple[PixelCatalog, List[bool]]:
        entries = list(self._integration_frame)
        catalogs: List[PixelCatalog] = list(map(lambda x: x[0], entries))
        matches: List[List[bool]] = list(map(lambda x: x[1], entries))

        integrated_catalog: PixelCatalog = sum(catalogs, PixelCatalog.default())
        integrated_matches = sum(matches, [])

        return integrated_catalog, integrated_matches


    @staticmethod
    def _convert_average_to_value(average_map: List[List[Average]]) -> npt.NDArray[np.float_]:
        return np.array(
            [[x.value if x is not None else np.nan for x in row] for row in average_map]
        )
