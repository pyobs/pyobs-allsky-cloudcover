"""The night pipeline built from a config dict, as the module does."""

import datetime

from pyobs_cloudcover.pipeline.night.pipeline_factory import NightPipelineFactory
from pyobs_cloudcover.pipeline.night.pipeline_options import NightPipelineOptions

from .conftest import NIGHT, SimulatedCamera


def test_night_pipeline_from_config(camera: SimulatedCamera) -> None:
    options = NightPipelineOptions.from_dict(
        {"bin_size": 2, "altaz_grid": {"point_number": 20000}, "cloud_map": {"threshold": 4.0}}
    )
    pipeline = NightPipelineFactory(camera.observer, camera.profile)(options)

    first = pipeline(camera.frame(NIGHT)[0])
    second = pipeline(camera.frame(NIGHT + datetime.timedelta(minutes=2))[0])

    assert first.change is None
    # clouds drift a little in two minutes (plus detection noise, which is considerable on small images)
    assert second.change is not None and 0 < second.change < 0.5
    assert first.cloud_cover_query.query_radius(1.5, 0, 1.0) is not None
