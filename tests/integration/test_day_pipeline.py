import datetime

import astropy.units as u
import numpy as np
from astroplan import Observer

from pyobs_cloudcover.pipeline.day.pipeline import DayPipeline
from pyobs_cloudcover.pipeline.day.sun_masker import SunMasker
from pyobs_cloudcover.pipeline.day.altaz_map_generator import AltAzMapGenerator
from pyobs_cloudcover.cloud_info_calculator.coverage_change_calculator import CoverageChangeCalculator
from pyobs_cloudcover.cloud_info_calculator import CoverageInfoCalculator
from pyobs_cloudcover.cloud_info_calculator import ZenithCloudCoverageCalculator
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator import CloudMapGenerator
from pyobs_cloudcover.world_model.simple_model import SimpleModel


def test_day_pipeline() -> None:
    observer = Observer(latitude=51.559299 * u.deg, longitude=9.945472 * u.deg, elevation=201 * u.m)
    obs_time = datetime.datetime(2024, 3, 9, 1, 48, 48, 297970)

    model_parameters = [4.81426598e-03, 2.00000000e+00, 1.06352627e+03, 7.57115607e+01, 5.11194838e+01]
    model = SimpleModel(*model_parameters)

    image = np.ones((104, 154)).astype(np.uint16) * 1
    image[104//2, 154//2] = 0
    mask = np.ones((104, 154)).astype(bool)

    alt_az_generator = AltAzMapGenerator(model, 20)
    cloud_map_gen = CloudMapGenerator(3.0)
    sun_masker = SunMasker(0.54, observer)

    coverage_change_calculator = CoverageChangeCalculator()
    cloud_coverage_info_calculator = CoverageInfoCalculator(coverage_change_calculator)

    pipeline = DayPipeline(mask, alt_az_generator, cloud_map_gen, sun_masker, cloud_coverage_info_calculator)

    result = pipeline(image, obs_time)

    assert result.change is None

