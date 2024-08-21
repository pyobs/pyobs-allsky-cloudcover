import datetime

import numpy as np
import numpy.typing as npt

from cloudmap_rs import SkyPixelQuery

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.pipeline.night.altaz_grid_generator.spherical_alt_az_generator import SphericalAltAzGenerator
from pyobs_cloudcover.pipeline.night.catalog.catalog_constructor import CatalogConstructor
from pyobs_cloudcover.cloud_info_calculator import CoverageInfoCalculator
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator import CloudMapGenerator
from pyobs_cloudcover.pipeline.night.lim_magnitude_map_generator.lim_magnitude_map_generator import LimMagnitudeMapGenerator
from pyobs_cloudcover.pipeline.night.preprocessor.preprocessor import Preprocessor
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher import StarReverseMatcher
from pyobs_cloudcover.pipeline.night.moon_masker import MoonMasker
from pyobs_cloudcover.pipeline.pipeline import Pipeline


class NightPipeline(Pipeline):
    def __init__(self,
                 preprocess: Preprocessor,
                 catalog_constructor: CatalogConstructor,
                 alt_az_list_generator: SphericalAltAzGenerator,
                 star_reverse_matcher: StarReverseMatcher,
                 lim_magnitude_map_generator: LimMagnitudeMapGenerator,
                 cloud_map_generator: CloudMapGenerator,
                 moon_masker: MoonMasker,
                 coverage_info_calculator: CoverageInfoCalculator) -> None:

        self._preprocess = preprocess
        self._catalog_constructor = catalog_constructor
        self._star_reverse_matcher = star_reverse_matcher
        self._alt_az_list_generator = alt_az_list_generator
        self._lim_magnitude_map_generator = lim_magnitude_map_generator
        self._cloud_map_generator = cloud_map_generator
        self._moon_masker = moon_masker
        self._coverage_info_calculator = coverage_info_calculator

    def __call__(self, image: npt.NDArray[np.float_], obs_time: datetime.datetime) -> CloudCoverageInfo:
        preprocessed_image = self._preprocess(image)
        img_height, img_width = preprocessed_image.shape

        catalog = self._catalog_constructor(obs_time, img_height, img_width)
        matches = self._star_reverse_matcher(preprocessed_image, catalog)

        alt_az_list = self._alt_az_list_generator()

        lim_mag_map = self._lim_magnitude_map_generator(catalog, matches, alt_az_list)
        cloud_map = self._cloud_map_generator(lim_mag_map)
        sky_query = SkyPixelQuery(alt_az_list, cloud_map)
        sky_query = self._moon_masker(sky_query, obs_time)

        return self._coverage_info_calculator(sky_query, obs_time)
