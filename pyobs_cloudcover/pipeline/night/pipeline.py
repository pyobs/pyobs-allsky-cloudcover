import datetime

import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.pipeline.night.altaz_map_generator.altaz_map_generator import AltAzMapGenerator
from pyobs_cloudcover.pipeline.night.catalog.catalog_constructor import CatalogConstructor
from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.coverage_info_calculator import CoverageInfoCalculator
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator import CloudMapGenerator
from pyobs_cloudcover.pipeline.night.preprocessor.preprocessor import Preprocessor
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher import StarReverseMatcher
from pyobs_cloudcover.pipeline.pipeline import Pipeline


class NightPipeline(Pipeline):
    def __init__(self,
                 preprocess: Preprocessor,
                 catalog_constructor: CatalogConstructor,
                 alt_az_list_generator: AltAzMapGenerator,
                 star_reverse_matcher: StarReverseMatcher,
                 cloud_map_generator: CloudMapGenerator,
                 coverage_info_calculator: CoverageInfoCalculator) -> None:

        self._preprocess = preprocess
        self._catalog_constructor = catalog_constructor
        self._star_reverse_matcher = star_reverse_matcher
        self._alt_az_list_generator = alt_az_list_generator
        self._cloud_map_generator = cloud_map_generator
        self._coverage_info_calculator = coverage_info_calculator

    def __call__(self, image: npt.NDArray[np.float_], obs_time: datetime.datetime) -> CloudCoverageInfo:
        preprocessed_image = self._preprocess(image)
        img_height, img_width = preprocessed_image.shape

        catalog = self._catalog_constructor(obs_time, img_height, img_width)
        matches = self._star_reverse_matcher(preprocessed_image, catalog)

        '''
        import matplotlib.pyplot as plt
        from astropy.visualization import PercentileInterval
        plt.imshow(PercentileInterval(99)(preprocessed_image), cmap="gray")
        plt.scatter(catalog.px[matches], catalog.py[matches], alpha=0.2, c="green", s=6)
        plt.scatter(catalog.px[~np.array(matches)], catalog.py[~np.array(matches)], alpha=0.2, c="red", s=6)
        plt.show()
        '''

        alt_az_list = self._alt_az_list_generator(img_height, img_width)

        cloud_map = self._cloud_map_generator(catalog, matches, alt_az_list)
        return self._coverage_info_calculator(cloud_map, obs_time, alt_az_list)
