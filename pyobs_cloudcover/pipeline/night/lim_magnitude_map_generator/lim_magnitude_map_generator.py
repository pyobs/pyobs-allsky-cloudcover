from collections import deque, defaultdict
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from cloudmap_rs import Star, Average, MagnitudeMapGenerator, AltAzCoord

from pyobs_cloudcover.pipeline.night.catalog.pixel_catalog import PixelCatalog


class LimMagnitudeMapGenerator(object):
    def __init__(self, radius: float, integration_length: int = 1):
        self._radius = radius
        self._integration_frame: deque[Tuple[PixelCatalog, List[bool]]] = deque([], integration_length)

    def __call__(self, catalog: PixelCatalog, matches: List[bool], alt_az_image_list: List[Optional[AltAzCoord]]) -> npt.NDArray[np.float_]:
        self._update_integration_frame(catalog, matches)
        integrated_matches = self._get_integrated_frame()

        star_coords = [AltAzCoord(np.deg2rad(alt), np.deg2rad(az)) for alt, az in zip(catalog.alt, catalog.az)]
        match_entries = [Star(v_mag, found) for v_mag, found in zip(catalog.v_mag, integrated_matches)]

        map_generator = MagnitudeMapGenerator(star_coords, match_entries)

        av_vis_map = map_generator.query_many(alt_az_image_list, np.deg2rad(self._radius))
        vis_map = self._convert_average_to_value(av_vis_map)

        return vis_map

    def _update_integration_frame(self, catalog: PixelCatalog, matches: List[bool]) -> None:
        self._integration_frame.append((catalog, matches))

    def _get_integrated_frame(self) -> List[bool]:
        entries = list(self._integration_frame)
        catalogs: List[PixelCatalog] = list(map(lambda x: x[0], entries))
        match_lists: List[List[bool]] = list(map(lambda x: x[1], entries))

        match_dict = defaultdict(list)

        for catalog, match_list in zip(catalogs, match_lists):
            for sao, match in zip(catalog.sao, match_list):
                match_dict[sao].append(match)

        return [all(match_dict[sao]) for sao in catalogs[-1].sao]

    @staticmethod
    def _convert_average_to_value(average_map: List[Average]) -> npt.NDArray[np.float_]:
        return np.array(
            [x.value if x is not None else np.nan for x in average_map]
        )
