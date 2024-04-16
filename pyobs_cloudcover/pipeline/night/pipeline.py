import datetime

import numpy as np
import numpy.typing as npt
from cloudmap_rs.cloudmap_rs import Entry

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.pipeline.night.catalog.catalog_constructor import CatalogConstructor
from pyobs_cloudcover.pipeline.night.catalog.pixel_catalog import PixelCatalog
from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.coverage_info_calculator import CoverageInfoCalculator
from pyobs_cloudcover.pipeline.night.cloud_map_generator import CloudMapGenerator
from pyobs_cloudcover.pipeline.night.preprocessor.preprocessor import Preprocessor
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher import StareReverseMatcher


class NightPipeline(object):
    def __init__(self,
                 preprocess: Preprocessor,
                 catalog_constructor: CatalogConstructor,
                 star_reverse_matcher: StareReverseMatcher,
                 cloud_map_generator: CloudMapGenerator,
                 coverage_info_calculator: CoverageInfoCalculator) -> None:

        self._preprocess = preprocess
        self._catalog_constructor = catalog_constructor
        self._star_reverse_matcher = star_reverse_matcher
        self._cloud_map_generator = cloud_map_generator
        self._coverage_info_calculator = coverage_info_calculator

    def __call__(self, image: npt.NDArray[np.float_], obstime: datetime.datetime) -> CloudCoverageInfo:
        preprocessed_image = self._preprocess(image)
        catalog = self._catalog_constructor(obstime)
        matches = self._star_reverse_matcher(preprocessed_image, catalog)
        cloud_map = self._cloud_map_generator(catalog, matches, *image.shape)
        return self._coverage_info_calculator(cloud_map)