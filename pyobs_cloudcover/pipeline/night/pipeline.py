import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.camera.profile import Frame
from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.cloud_info_calculator import CoverageInfoCalculator
from pyobs_cloudcover.pipeline.night.altaz_grid_generator.spherical_alt_az_generator import SphericalAltAzGenerator
from pyobs_cloudcover.pipeline.night.catalog.catalog_constructor import CatalogConstructor
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator import CloudMapGenerator
from pyobs_cloudcover.pipeline.night.lim_magnitude_map_generator.lim_magnitude_map_generator import (
    LimMagnitudeMapGenerator,
)
from pyobs_cloudcover.pipeline.night.moon_masker import MoonMasker
from pyobs_cloudcover.pipeline.night.preprocessor.image_binner import ImageBinner
from pyobs_cloudcover.pipeline.night.preprocessor.preprocessor import Preprocessor
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.aperture_matcher import ApertureMatcher
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher import StarReverseMatcher
from pyobs_cloudcover.pipeline.pipeline import Pipeline
from pyobs_cloudcover.sky_query import SkyPixelQuery


class NightPipeline(Pipeline):
    def __init__(self,
                 preprocess: Preprocessor,
                 catalog_constructor: CatalogConstructor,
                 alt_az_list_generator: SphericalAltAzGenerator,
                 star_reverse_matcher: StarReverseMatcher | ApertureMatcher,
                 lim_magnitude_map_generator: LimMagnitudeMapGenerator,
                 cloud_map_generator: CloudMapGenerator,
                 moon_masker: MoonMasker,
                 coverage_info_calculator: CoverageInfoCalculator,
                 sky_mask: npt.NDArray[np.bool_] | None = None) -> None:
        """``sky_mask``: full-resolution canonical mask (True = sky); stars outside it are ignored."""

        self._preprocess = preprocess
        self._catalog_constructor = catalog_constructor
        self._star_reverse_matcher = star_reverse_matcher
        self._alt_az_list_generator = alt_az_list_generator
        self._lim_magnitude_map_generator = lim_magnitude_map_generator
        self._cloud_map_generator = cloud_map_generator
        self._moon_masker = moon_masker
        self._coverage_info_calculator = coverage_info_calculator
        self._sky_mask = sky_mask
        self._binned_mask: npt.NDArray[np.bool_] | None = None

    def _mask(self) -> npt.NDArray[np.bool_] | None:
        if self._sky_mask is not None and self._binned_mask is None:
            # a binned pixel is sky if most of it is
            self._binned_mask = ImageBinner(self._preprocess.bin_size)(self._sky_mask.astype(np.float64)) >= 0.5
        return self._binned_mask

    def __call__(self, frame: Frame) -> CloudCoverageInfo:
        obs_time = frame.obs_time
        preprocessed_image = self._preprocess(frame)
        img_height, img_width = preprocessed_image.shape

        catalog = self._catalog_constructor(obs_time, img_height, img_width)
        mask = self._mask()
        if mask is not None:
            catalog.filter_mask(mask)
        matches = self._star_reverse_matcher(preprocessed_image, catalog, frame.saturation)

        grid = self._alt_az_list_generator()

        lim_mag_map = self._lim_magnitude_map_generator(catalog, matches, grid)
        cloud_map = self._cloud_map_generator(lim_mag_map)
        sky_query = self._moon_masker(SkyPixelQuery(grid, cloud_map), obs_time)
        lim_mag_query = self._moon_masker(SkyPixelQuery(grid, lim_mag_map), obs_time)

        return self._coverage_info_calculator(sky_query, obs_time, lim_mag_query, "limiting_magnitude", "night")
