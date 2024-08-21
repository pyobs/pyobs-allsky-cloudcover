import datetime

import astropy.units as u
import numpy as np
from astroplan import Observer

from pyobs_cloudcover.pipeline.night.altaz_grid_generator.spherical_alt_az_generator import SphericalAltAzGenerator
from pyobs_cloudcover.pipeline.night.catalog.altaz_catalog_loader import AltAzCatalogLoader
from pyobs_cloudcover.pipeline.night.catalog.catalog_constructor import CatalogConstructor
from pyobs_cloudcover.cloud_info_calculator.coverage_change_calculator import CoverageChangeCalculator
from pyobs_cloudcover.cloud_info_calculator import CoverageInfoCalculator
from pyobs_cloudcover.cloud_info_calculator import ZenithCloudCoverageCalculator
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator import CloudMapGenerator
from pyobs_cloudcover.pipeline.night.lim_magnitude_map_generator.lim_magnitude_map_generator import \
    LimMagnitudeMapGenerator
from pyobs_cloudcover.pipeline.night.moon_masker import MoonMasker
from pyobs_cloudcover.pipeline.night.pipeline import NightPipeline
from pyobs_cloudcover.pipeline.night.preprocessor.image_binner import ImageBinner
from pyobs_cloudcover.pipeline.night.preprocessor.image_masker import ImageMasker
from pyobs_cloudcover.pipeline.night.preprocessor.preprocessor import Preprocessor

from pyobs_cloudcover.pipeline.night.star_reverse_matcher.detector.sigma_treshhold_detector import \
    SigmaThresholdDetector
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher import StarReverseMatcher
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.window import ImageWindow
from pyobs_cloudcover.world_model.simple_model import SimpleModel


def test_night_pipeline() -> None:
    observer = Observer(latitude=51.559299 * u.deg, longitude=9.945472 * u.deg, elevation=201 * u.m)
    obs_time = datetime.datetime(2024, 3, 9, 1, 48, 48, 297970)

    model_parameters = [4.81426598e-03, 2.00000000e+00, 1.06352627e+03, 7.57115607e+02, 5.11194838e+02]
    model = SimpleModel(*model_parameters)

    stars = np.loadtxt('tests/integration/matches_small_20240308.csv', delimiter=",")

    image = np.zeros((2*1040, 2*1548))

    for star in stars:
        image[int(star[2]), int(star[1])] = 10

    mask = ImageMasker(np.ones((2*1040, 2*1548)).astype(np.bool_))
    binner = ImageBinner(2)
    preprocessor = Preprocessor(mask, binner)

    altaz_catalog_loader = AltAzCatalogLoader.from_csv("tests/integration/catalog.csv")
    catalog_constructor = CatalogConstructor(altaz_catalog_loader, model, observer, 0.0, 3.0, 0.0)

    altaz_list_generator = SphericalAltAzGenerator(100, 30.0)

    reverse_matcher = StarReverseMatcher(SigmaThresholdDetector(3.0, 4.0, 8e3), ImageWindow(6.0))

    cloud_map_gen = CloudMapGenerator(0.5)
    lim_mag_map_generator = LimMagnitudeMapGenerator(50.0)

    moon_masker = MoonMasker(10, observer)

    coverage_change_calculator = CoverageChangeCalculator()
    cloud_coverage_info_calculator = CoverageInfoCalculator(coverage_change_calculator)

    pipeline = NightPipeline(preprocessor, catalog_constructor, altaz_list_generator, reverse_matcher, lim_mag_map_generator, cloud_map_gen, moon_masker, cloud_coverage_info_calculator)

    pipeline(image, obs_time)
