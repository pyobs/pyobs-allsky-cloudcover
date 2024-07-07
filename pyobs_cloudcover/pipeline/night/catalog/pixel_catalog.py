from __future__ import annotations

import datetime
from typing import cast, List, Tuple, Set

import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.pipeline.night.catalog.altaz_catalog import AltAzCatalog
from pyobs_cloudcover.world_model import WorldModel


class PixelCatalog(object):
    def __init__(self, sao: npt.NDArray[np.int_], alt: npt.NDArray[np.float_], az: npt.NDArray[np.float_], px: npt.NDArray[np.float_], py: npt.NDArray[np.float_],
                 v_mag: npt.NDArray[np.float_]) -> None:
        self.sao = sao
        self.alt = alt
        self.az = az
        self.px = px
        self.py = py
        self.v_mag = v_mag

    def _filter(self, condition: npt.NDArray[np.bool_]) -> None:
        self.sao = self.sao[condition]
        self.alt = self.alt[condition]
        self.az = self.az[condition]
        self.px = self.px[condition]
        self.py = self.py[condition]
        self.v_mag = self.v_mag[condition]

    def filter_close(self, distance: float) -> None:
        indices = np.arange(len(self.sao))
        neighbourhood = [
                indices[self._in_neighbourhood(x, y, self.px, self.py, distance)] for x, y in zip(self.px, self.py)
             ]

        neighbors = list(map(tuple, (filter(lambda x: len(x) > 1, neighbourhood))))

        if len(neighbors) == 0:
            return

        clusters = self._find_cluster(neighbors)

        tuple_indices_to_delete = [self._get_faintest_indices(group) for group in clusters]

        stars_to_delete = [entry for group, delete in zip(clusters, tuple_indices_to_delete) for entry in group[delete]]

        indices_to_delete = np.isin(indices, stars_to_delete)
        self._filter(~indices_to_delete)

    @staticmethod
    def _in_neighbourhood(x: float, y: float, px: npt.NDArray[np.float_], py: npt.NDArray[np.float_], d: float) -> npt.NDArray[np.bool_]:
        in_neighbourhood: npt.NDArray[np.bool_] = np.sqrt(np.square(x - px) + np.square(y - py)) <= d
        return in_neighbourhood

    @staticmethod
    def _find_cluster(neighbours: List[Tuple[np.int_, ...]]) -> List[npt.NDArray[np.int_]]:
        """
        Finds clusters in list of neighbours
        """

        unique_neighbour_pairs = list(set(neighbours))
        clusters: List[Set[np.int_]] = [set(unique_neighbour_pairs[0])]

        for pair in unique_neighbour_pairs[1:]:
            for group in clusters:
                if any([x in group for x in pair]):
                    group.update(set(pair))
                    break
            else:
                clusters.append(set(pair))

        cluster_list: List[npt.NDArray[np.int_]] = list(map(lambda x: np.array(list(x)), clusters))

        return cluster_list

    def _get_faintest_indices(self, cluster: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        return np.argsort(self.v_mag[cluster])[1:]

    def filter_window_size(self, height: int, width: int) -> None:
        inside_x_axis = (0 < self.px) & (self.px < width)
        inside_y_axis = (0 < self.py) & (self.py < height)

        self._filter(inside_x_axis & inside_y_axis)

    @classmethod
    def from_altaz_catalog(cls, altaz_catalog: AltAzCatalog, model: WorldModel) -> PixelCatalog:
        px, py = model.altaz_to_pix(np.deg2rad(altaz_catalog.alt), np.deg2rad(altaz_catalog.az))

        return PixelCatalog(
            altaz_catalog.sao,
            altaz_catalog.alt, altaz_catalog.az,
            cast(npt.NDArray[np.float_], px), cast(npt.NDArray[np.float_], py),
            altaz_catalog.v_mag
            )

    @classmethod
    def default(cls) -> PixelCatalog:
        return PixelCatalog(np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))

    def __add__(self, other: PixelCatalog) -> PixelCatalog:
        return PixelCatalog(
            np.concatenate([self.sao, other.sao]),
            np.concatenate([self.alt, other.alt]),
            np.concatenate([self.az, other.az]),
            np.concatenate([self.px, other.px]),
            np.concatenate([self.py, other.py]),
            np.concatenate([self.v_mag, other.v_mag]),
        )