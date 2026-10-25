"""The night pipeline assembled by hand from its components, on a simulated frame."""

import numpy as np

from pyobs_cloudcover.cloud_info_calculator import CoverageInfoCalculator
from pyobs_cloudcover.cloud_info_calculator.coverage_change_calculator import CoverageChangeCalculator
from pyobs_cloudcover.data import bundled_catalog_path
from pyobs_cloudcover.pipeline.night.altaz_grid_generator.spherical_alt_az_generator import SphericalAltAzGenerator
from pyobs_cloudcover.pipeline.night.catalog.altaz_catalog_loader import AltAzCatalogLoader
from pyobs_cloudcover.pipeline.night.catalog.catalog_constructor import CatalogConstructor
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator import CloudMapGenerator
from pyobs_cloudcover.pipeline.night.lim_magnitude_map_generator.lim_magnitude_map_generator import (
    LimMagnitudeMapGenerator,
)
from pyobs_cloudcover.pipeline.night.moon_masker import MoonMasker
from pyobs_cloudcover.pipeline.night.pipeline import NightPipeline
from pyobs_cloudcover.pipeline.night.preprocessor.preprocessor import Preprocessor
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.aperture_matcher import ApertureMatcher
from pyobs_cloudcover.world_model import ScaledModel

from .conftest import NIGHT, SimulatedCamera


def test_night_pipeline(camera: SimulatedCamera) -> None:
    model = ScaledModel(camera.sim.world_model, 2)
    pipeline = NightPipeline(
        Preprocessor(bin_size=2),
        CatalogConstructor(AltAzCatalogLoader.from_csv(bundled_catalog_path()), model, camera.observer, 30, 7, 0),
        SphericalAltAzGenerator(20000, 30.0),
        ApertureMatcher(snr=3),
        LimMagnitudeMapGenerator(7.0, estimator="robust"),
        CloudMapGenerator(4.0),
        MoonMasker(10, camera.observer),
        CoverageInfoCalculator(CoverageChangeCalculator()),
    )
    frame, simulated = camera.frame(NIGHT)
    result = pipeline(frame)

    assert result.pipeline == "night" and result.change is None
    assert result.value_query is not None and result.value_name == "limiting_magnitude"
    lim = result.value_query.get_pixels()
    assert 2 < np.nanmedian(lim) < 7
    # cloudy points have a lower limiting magnitude than clear ones
    grid = result.value_query.grid
    cloudy = simulated.truth.cloudy_at(np.rad2deg(grid.alt), np.rad2deg(grid.az))
    assert np.nanmedian(lim[cloudy]) < np.nanmedian(lim[~cloudy])
