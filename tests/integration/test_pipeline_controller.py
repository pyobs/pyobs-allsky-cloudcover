"""The pipeline controller picks the pipeline by solar altitude."""

from pyobs_cloudcover.pipeline.pipeline_controller_factory import PipelineControllerFactory

from .conftest import DAY, NIGHT, TWILIGHT, SimulatedCamera


def test_pipeline_by_sun_altitude(camera: SimulatedCamera) -> None:
    controller = PipelineControllerFactory(camera.observer, camera.profile)(
        {
            "night": {"alt_interval": {"end": -18}, "options": {"altaz_grid": {"point_number": 20000}}},
            "day": {"alt_interval": {"start": 18}, "options": {}},
        }
    )
    night = controller(camera.frame(NIGHT)[0])
    day = controller(camera.frame(DAY)[0])
    twilight = controller(camera.frame(TWILIGHT)[0])

    assert night is not None and night.pipeline == "night"
    assert day is not None and day.pipeline == "day"
    assert twilight is None
