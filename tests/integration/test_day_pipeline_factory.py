"""The day pipeline built from a config dict, as the module does."""

import numpy as np

from pyobs_cloudcover.pipeline.day.pipeline_factory import DayPipelineFactory
from pyobs_cloudcover.pipeline.day.pipeline_options import DayPipelineOptions

from .conftest import DAY, SimulatedCamera


def test_day_pipeline_from_config(camera: SimulatedCamera) -> None:
    options = DayPipelineOptions.from_dict({"bin_size": 2, "sun_apparent_size": 5.0})
    pipeline = DayPipelineFactory(camera.observer, camera.profile)(options)
    frame, simulated = camera.frame(DAY)
    result = pipeline(frame)

    assert result is not None and result.pipeline == "day" and result.value_name == "color_ratio"
    measured = result.cloud_cover_query.query_radius(np.pi / 2, 0, np.deg2rad(70))
    assert measured is not None
    assert abs(measured - simulated.truth.cloud_fraction(20)) < 0.15
    # the sun is masked
    sun_alt, sun_az = np.deg2rad(simulated.truth.metadata["sun_alt"]), np.deg2rad(simulated.truth.metadata["sun_az"])
    assert result.cloud_cover_query.query_radius(sun_alt, sun_az, np.deg2rad(4)) is None
