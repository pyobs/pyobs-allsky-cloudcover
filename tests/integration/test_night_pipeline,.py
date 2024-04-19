import datetime

import numpy as np
from astroplan import Observer

import astropy.units as u

from pyobs_cloudcover.pipeline.night.catalog.altaz_catalog_loader import AltAzCatalogLoader
from pyobs_cloudcover.pipeline.night.catalog.catalog_constructor import CatalogConstructor
from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.coverage_calculator import CoverageCalculator
from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.coverage_change_calculator import \
    CoverageChangeCalculator
from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.coverage_info_calculator import CoverageInfoCalculator
from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.zenith_masker import ZenithMasker
from pyobs_cloudcover.pipeline.night.cloud_map_generator import CloudMapGenerator
from pyobs_cloudcover.pipeline.night.pipeline import NightPipeline
from pyobs_cloudcover.pipeline.night.preprocessor.image_masker import ImageMasker
from pyobs_cloudcover.pipeline.night.preprocessor.preprocessor import Preprocessor
from pyobs_cloudcover.pipeline.night.preprocessor.image_binner import ImageBinner
from pyobs_cloudcover.pipeline.night.preprocessor.background_remover import BackgroundRemover
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.detector.sigma_treshhold_detector import \
    SigmaThresholdDetector
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher import StareReverseMatcher
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.window import ImageWindow
from pyobs_cloudcover.world_model.simple_model import SimpleModel


def test_night_pipeline():
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
    background_remover = BackgroundRemover()
    preprocessor = Preprocessor(mask, binner, background_remover)

    altaz_catalog_loader = AltAzCatalogLoader.from_csv("tests/integration/catalog.csv")
    catalog_constructor = CatalogConstructor(altaz_catalog_loader, model, observer, 0.0, 3.0, 0.0)

    reverse_matcher = StareReverseMatcher(SigmaThresholdDetector(3.0), ImageWindow(6.0))

    cloud_map_gem = CloudMapGenerator(50.0)

    coverage_calculator = CoverageCalculator(0.5)
    coverage_change_calculator = CoverageChangeCalculator()
    zenith_masker = ZenithMasker(80, model)
    cloud_coverage_info_calculator = CoverageInfoCalculator(coverage_calculator, coverage_change_calculator, zenith_masker)

    pipeline = NightPipeline(preprocessor, catalog_constructor, reverse_matcher, cloud_map_gem, cloud_coverage_info_calculator)

    pipeline(image, obs_time)
