from collections import defaultdict, deque

import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.pipeline.night.catalog.pixel_catalog import PixelCatalog
from pyobs_cloudcover.pipeline.night.lim_magnitude_map_generator.border_magnitude import border_magnitude
from pyobs_cloudcover.sky_query import SkyGrid


class LimMagnitudeMapGenerator:
    def __init__(self, radius: float, integration_length: int = 1):
        self._radius = radius
        self._integration_frame: deque[tuple[PixelCatalog, list[bool]]] = deque([], integration_length)

    def __call__(self, catalog: PixelCatalog, matches: list[bool], grid: SkyGrid) -> npt.NDArray[np.float64]:
        self._update_integration_frame(catalog, matches)
        integrated_matches = self._get_integrated_frame()

        return border_magnitude(
            grid,
            np.deg2rad(np.asarray(catalog.alt, dtype=np.float64)),
            np.deg2rad(np.asarray(catalog.az, dtype=np.float64)),
            np.asarray(catalog.v_mag, dtype=np.float64),
            np.asarray(integrated_matches, dtype=np.bool_),
            np.deg2rad(self._radius),
        )

    def _update_integration_frame(self, catalog: PixelCatalog, matches: list[bool]) -> None:
        self._integration_frame.append((catalog, matches))

    def _get_integrated_frame(self) -> list[bool]:
        entries = list(self._integration_frame)
        catalogs: list[PixelCatalog] = list(map(lambda x: x[0], entries))
        match_lists: list[list[bool]] = list(map(lambda x: x[1], entries))

        match_dict = defaultdict(list)

        for catalog, match_list in zip(catalogs, match_lists):
            for sao, match in zip(catalog.sao, match_list):
                match_dict[sao].append(match)

        return [all(match_dict[sao]) for sao in catalogs[-1].sao]
